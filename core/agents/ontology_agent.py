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
            profile = QuestionProfile(
                identity=question,
                one_line_summary=raw.get("one_line_summary", ""),
                question_type=raw.get("question_type", "continuous"),
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

    def filter_documents(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
        profile: "QuestionProfile",
        question: str,
        min_keep: int = 6,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        if not docs:
            return [], [], []

        # build top concepts from ontology av_pairs sorted by centrality
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
        min_full = max(3, n // 4)
        min_total = max(min_keep, n // 2)

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
            f'- Be GENEROUS. When uncertain, prefer "abstract" over "drop", '
            f'"full" over "abstract".\n'
            f'- "full" documents will have their complete PDF text fetched — only assign '
            f'"full" if the document directly addresses the main topic or a closely '
            f"related method or variable.\n"
            f'- "drop" only when there is no plausible connection to the question or ontology.\n'
            f'- You MUST return at least {min_full} "full" classifications.\n'
            f'- You MUST return at least {min_total} combined "full"+"abstract" '
            f"classifications.\n\n"
            f"Documents:\n{doc_list}\n\n"
            f'Return ONLY valid JSON: {{"classifications": ["full", "abstract", "drop", ...]}}\n'
            f"No prose. The list length must equal the number of documents ({n})."
        )

        # follow the exact same llm invocation pattern used in process_with_profile()
        try:
            raw = self.llm.invoke(prompt, force_json=True)
            if isinstance(raw, str):
                import json as _json
                data = _json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                raise ValueError(f"unexpected llm response type: {type(raw)}")

            classifications = data.get("classifications", [])

            if len(classifications) != n:
                raise ValueError(
                    f"classification count {len(classifications)} != doc count {n}"
                )
            n_full = classifications.count("full")
            n_total = n_full + classifications.count("abstract")
            if n_full < min_full or n_total < min_total:
                raise ValueError(
                    f"guardrail violated: n_full={n_full} (min {min_full}), "
                    f"n_total={n_total} (min {min_total})"
                )

        except Exception as exc:
            # fail-safe: keep all docs, never silently discard evidence
            self.logger.warning(
                "filter_documents failed (%s) — falling back to full set", exc
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

    def _define_logical_relationships(
        self, av_pairs: List[AttributeValuePair], question: str
    ) -> List[LogicalRelationship]:
        if not av_pairs:
            return []

        attrs_text = "\n".join(f"- {av.attribute}: {av.value}" for av in av_pairs)
        template = self.load_prompt("extract_relationships.txt")
        prompt = (
            template.replace("{question}", question).replace("{attributes}", attrs_text)
        )

        try:
            response = self.llm.invoke(prompt, force_json=True)
            if isinstance(response, dict) and "relationships" in response:
                return [
                    LogicalRelationship(
                        source_attribute=r.get("source_attribute", ""),
                        source_value=r.get("source_value", ""),
                        target_attribute=r.get("target_attribute", ""),
                        target_value=r.get("target_value", ""),
                        relationship_type=r.get("relationship_type", "relates_to"),
                        logical_constraint=r.get("logical_constraint", ""),
                    )
                    for r in response["relationships"]
                ]
            self.logger.error(f"unexpected relationship response type: {type(response)}")
            return []
        except Exception as e:
            self.logger.error(f"relationship extraction failed: {e}")
            return []
