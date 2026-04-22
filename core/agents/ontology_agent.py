"""constructs a dynamic ontology from a user question via llm calls."""

import os
from typing import Any, Dict, List, Tuple

from core.agents.base_agent import BaseAgent
from core.utils.data_models import (
    AttributeValuePair,
    DynamicOntology,
    LogicalRelationship,
    QuestionProfile,
)


class OntologyConstructionAgent(BaseAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/ontology"):
        super().__init__("OntologyConstructionAgent", llm)
        self.prompt_dir = prompt_dir

    def load_prompt(self, filename: str) -> str:
        filepath = os.path.join(self.prompt_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def process(self, question: str, include_relationships: bool = True) -> DynamicOntology:
        av_pairs = self._extract_attribute_value_pairs(question)
        self.logger.info(f"extracted {len(av_pairs)} attribute-value pairs")

        relationships: List[LogicalRelationship] = []
        if include_relationships:
            relationships = self._define_logical_relationships(av_pairs, question)
            self.logger.info(f"defined {len(relationships)} logical relationships")

        return DynamicOntology(
            attribute_value_pairs=av_pairs,
            logical_relationships=relationships,
            source_query=question,
            should_use_ontology=True,
            include_relationships=include_relationships,
        )

    # ------------------------------------------------------------------
    # private llm helpers
    # ------------------------------------------------------------------

    def _extract_attribute_value_pairs(self, question: str) -> List[AttributeValuePair]:
        template = self.load_prompt("extract_attributes_slim.txt")
        prompt = template.replace("{question}", question)

        try:
            response = self.llm.invoke(prompt, force_json=True)
            if isinstance(response, dict) and "pairs" in response:
                return [
                    AttributeValuePair(
                        attribute=p.get("attribute", ""),
                        value=p.get("value", ""),
                        description=p.get("description", ""),
                        centrality=p.get("centrality", 0.5),
                    )
                    for p in response["pairs"]
                ]
            self.logger.error(f"unexpected av-pair response type: {type(response)}")
            return []
        except Exception as e:
            self.logger.error(f"attribute extraction failed: {e}")
            return []

    # ------------------------------------------------------------------
    # adaptive pipeline (set5) -- fused av-pair + profile extraction
    # ------------------------------------------------------------------

    def process_with_profile(
        self, question: str, include_relationships: bool = False
    ):
        """superset of process(): fuses av-pair extraction and profile
        extraction into a single llm call. falls back to process() plus a
        null-confidence profile on any error -- the null confidence then
        triggers the safety net in core.policy.router.

        include_relationships defaults to False because the adaptive
        pipeline routes before refinement, so relationships add latency
        without informing routing.
        """
        template = self.load_prompt("extract_profile.txt")
        prompt = template.replace("{question}", question)

        try:
            resp = self.llm.invoke(prompt, force_json=True)
            if not isinstance(resp, dict):
                raise ValueError(f"unexpected response type: {type(resp)}")

            av_pairs = [
                AttributeValuePair(
                    attribute=p.get("attribute", ""),
                    value=p.get("value", ""),
                    description=p.get("description", ""),
                    centrality=p.get("centrality", 0.5),
                )
                for p in resp.get("pairs", [])
            ]
            relationships: List[LogicalRelationship] = []
            if include_relationships:
                relationships = self._define_logical_relationships(av_pairs, question)

            ontology = DynamicOntology(
                attribute_value_pairs=av_pairs,
                logical_relationships=relationships,
                source_query=question,
                should_use_ontology=True,
                include_relationships=include_relationships,
            )

            raw = resp.get("profile", {}) or {}
            # normalise question_type: strip whitespace and lowercase to guard
            # against LLM responses like "Mechanism" or "mechanism " that would
            # silently fail the {in: [...]} check in rules.yaml and cause
            # tier-3 to never fire.
            raw_qt = (raw.get("question_type") or "continuous").strip().lower()
            profile = QuestionProfile(
                identity=question,
                one_line_summary=raw.get("one_line_summary", ""),
                question_type=raw_qt,
                complexity=float(raw.get("complexity", 0.5)),
                quantitativity=float(raw.get("quantitativity", 0.3)),
                spatial_specificity=float(raw.get("spatial_specificity", 0.1)),
                temporal_specificity=float(raw.get("temporal_specificity", 0.1)),
                methodological_depth=float(raw.get("methodological_depth", 0.1)),
                needs_numeric_emphasis=bool(raw.get("needs_numeric_emphasis", False)),
                confidence=raw.get("confidence", None),
            )

            self.logger.info(
                f"profile: type={profile.question_type} "
                f"complexity={profile.complexity:.2f} "
                f"quant={profile.quantitativity:.2f} "
                f"conf={profile.confidence}"
            )
            return ontology, profile

        except Exception as e:
            self.logger.error(f"process_with_profile failed: {e} -- falling back")
            ontology = self.process(question, include_relationships)
            profile = QuestionProfile(
                identity=question,
                one_line_summary="",
                question_type="continuous",
                complexity=0.5,
                quantitativity=0.3,
                spatial_specificity=0.1,
                temporal_specificity=0.1,
                methodological_depth=0.1,
                confidence=None,  # triggers safety-tier3 in router
            )
            return ontology, profile

    # ------------------------------------------------------------------
    # document relevance filter (second LLM pass)
    # ------------------------------------------------------------------

    def filter_documents(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
        profile: "QuestionProfile",
        question: str,
        min_keep: int = 6,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """LLM-based document relevance filter.

        Classifies every document in `docs` as one of:
          "full"     -- fetch PDF + excerpts
          "abstract" -- abstract only (no PDF fetch)
          "drop"     -- exclude from refinement entirely

        The filter is complexity-aware: simple/definitional questions are
        steered toward fewer "full" fetches since their abstracts typically
        suffice. Complex mechanistic or comparison questions may use more.

        Returns (full_docs, abstract_docs, drop_docs).
        On any error (including LLM 503/timeout) falls back to returning
        all docs as full_docs so the pipeline degrades gracefully.
        """
        if not docs:
            return [], [], []

        # top-5 ontology concepts by centrality, for the filter prompt
        pairs = sorted(
            getattr(ontology, "attribute_value_pairs", []),
            key=lambda av: getattr(av, "centrality", 0.0),
            reverse=True,
        )[:5]
        top_concepts = "\n".join(
            f"- {getattr(av, 'attribute', '')}: {getattr(av, 'value', '')}"
            for av in pairs
        )

        n = len(docs)
        # simple questions (definition / factual) need fewer full-text fetches
        is_simple = getattr(profile, "question_type", "mechanism") in ("definition", "factual")
        min_full  = max(2, n // 6) if is_simple else max(3, n // 4)
        min_total = max(4, n // 4) if is_simple else max(min_keep, n // 2)

        doc_list = "\n".join(
            f"{i+1}. {docs[i].get('title', 'Unknown')}" for i in range(n)
        )

        prompt = (
            f"You are a document relevance filter for a scientific RAG pipeline.\n\n"
            f"Question: {question}\n"
            f"Key ontology concepts (most important first):\n{top_concepts}\n\n"
            f"For each document below, classify it as exactly one of:\n"
            f'- "full"     : clearly relevant to the ontology concepts; worth reading in depth\n'
            f'- "abstract" : potentially relevant or provides supporting context; '
            f'abstract only is sufficient\n'
            f'- "drop"     : clearly unrelated to both the question and ontology concepts\n\n'
            f"Rules:\n"
            f"- First, judge the question's complexity: does it require deep methodological\n"
            f"  detail from multiple sources, or can it be answered well with 2-3 strong papers?\n"
            f'- Only mark "full" for documents that are CENTRAL to answering the question —\n'
            f"  i.e. they likely contain the specific methods, results, or measurements needed.\n"
            f'  If the abstract alone is sufficient to support a claim, mark "abstract".\n'
            f"- A simple or definitional question rarely needs more than 3 \"full\" documents.\n"
            f"  A complex mechanistic or comparison question may need up to 6. Be honest\n"
            f"  about which this is.\n"
            f'- "drop" only when there is no plausible connection to the question or ontology.\n'
            f'- You MUST return at least {min_full} "full" classifications.\n'
            f'- You MUST return at least {min_total} combined "full"+"abstract" '
            f"classifications.\n\n"
            f"Documents:\n{doc_list}\n\n"
            f'Return ONLY valid JSON: {{"classifications": ["full", "abstract", "drop", ...]}}\n'
            f"No prose. The list length must equal the number of documents ({n})."
        )

        try:
            raw = self.llm.invoke(prompt, force_json=True)

            # explicit None check: invoke() returns None when all retries are
            # exhausted (503, timeout, etc.). treat as transient -- fall back.
            if raw is None:
                raise ValueError("LLM unavailable after retries (returned None)")

            if isinstance(raw, str):
                import json as _json
                data = _json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                raise ValueError(f"unexpected llm response type: {type(raw)}")

            classifications = data.get("classifications", [])
            if not classifications:
                raise ValueError("LLM returned empty classifications list")
            if len(classifications) != n:
                raise ValueError(
                    f"classification count {len(classifications)} != doc count {n}"
                )

            n_full  = classifications.count("full")
            n_total = n_full + classifications.count("abstract")
            if n_full < min_full or n_total < min_total:
                raise ValueError(
                    f"guardrail violated: n_full={n_full} (min {min_full}), "
                    f"n_total={n_total} (min {min_total})"
                )

        except Exception as exc:
            self.logger.warning(
                "filter_documents: %s \u2014 falling back to full set", exc
            )
            return list(docs), [], []

        full_docs     = [docs[i] for i, c in enumerate(classifications) if c == "full"]
        abstract_docs = [docs[i] for i, c in enumerate(classifications) if c == "abstract"]
        drop_docs     = [docs[i] for i, c in enumerate(classifications) if c == "drop"]

        self.logger.info(
            "filter_documents: %d full, %d abstract, %d dropped (of %d)",
            len(full_docs), len(abstract_docs), len(drop_docs), n,
        )
        return full_docs, abstract_docs, drop_docs
