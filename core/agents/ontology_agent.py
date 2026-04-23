"""constructs a dynamic ontology from a user question via llm calls."""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.agents.base_agent import BaseAgent
from core.utils.data_models import (
    AttributeValuePair,
    DynamicOntology,
    LogicalRelationship,
    QuestionProfile,
)


class OntologyAgent(BaseAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/ontology"):
        super().__init__("OntologyAgent", llm)
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

    def process_with_profile(
        self, question: str, include_relationships: bool = False
    ):
        """superset of process(): fuses av-pair extraction and profile
        extraction into a single llm call. falls back to process() plus a
        null-confidence profile on any error.
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
                confidence=None,
            )
            return ontology, profile

    def _lexical_score(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
    ) -> List[Tuple[float, int, Dict]]:
        """Score docs by keyword overlap with ontology AV-pairs."""
        word_re = re.compile(r"[a-z][a-z0-9\-]{1,}")

        kws: List[str] = []
        lits: List[str] = []
        for av in getattr(ontology, "attribute_value_pairs", []):
            for field in ("attribute", "value", "description"):
                val = getattr(av, field, None)
                if val and isinstance(val, str):
                    kws.extend(word_re.findall(val.lower()))
                    short = val.strip().lower()
                    if short and len(short) <= 40:
                        lits.append(short)
        seen: set = set()
        kws = [k for k in kws if len(k) >= 3 and not (k in seen or seen.add(k))]

        scored: List[Tuple[float, int, Dict]] = []
        for idx, doc in enumerate(docs):
            text = ((doc.get("abstract") or "") + " " + (doc.get("title") or "")).lower()
            token_set = set(word_re.findall(text))
            kw_hits  = sum(1 for k in kws if k in token_set)
            lit_hits = sum(1 for l in lits if l and l in text)
            score    = kw_hits + 0.5 * lit_hits
            scored.append((score, idx, doc))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored

    def _lexical_fallback(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
        min_full: int,
        min_total: int,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Rank docs by keyword overlap and split into full/abstract/drop."""
        scored = self._lexical_score(docs, ontology)

        all_zero = all(s == 0 for s, _, _ in scored)
        if all_zero:
            self.logger.warning(
                "filter_documents lexical_fallback: all scores are 0 "
                "(empty ontology?) — using document order as tiebreak"
            )

        ranked = [doc for _, _, doc in scored]
        full_docs     = ranked[:min_full]
        abstract_docs = ranked[min_full:min_total]
        drop_docs     = ranked[min_total:]

        self.logger.info(
            "filter_documents lexical_fallback: %d full, %d abstract, %d drop (of %d)",
            len(full_docs), len(abstract_docs), len(drop_docs), len(docs),
        )
        return full_docs, abstract_docs, drop_docs

    def _abstracts_topup(
        self,
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
        drop_docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
        min_keep: int,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """top up the kept pool to min_keep by promoting best-scoring dropped docs."""
        kept = len(full_docs) + len(abstract_docs)
        needed = min_keep - kept
        if needed <= 0 or not drop_docs:
            return full_docs, abstract_docs, drop_docs

        scored = self._lexical_score(drop_docs, ontology)
        promote = [doc for _, _, doc in scored[:needed]]
        remaining_drop = [doc for _, _, doc in scored[needed:]]

        self.logger.info(
            "filter_documents abstracts_topup: promoting %d dropped docs "
            "to abstract (kept was %d, target %d)",
            len(promote), kept, min_keep,
        )
        return full_docs, abstract_docs + promote, remaining_drop

    def filter_documents(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
        profile: "QuestionProfile",
        question: str,
        min_keep: int = 6,
        evidence_mode: str = "excerpts_narrow",
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """LLM-based document relevance filter.

        Classifies every document as 'full', 'abstract', or 'drop'.
        Guardrail floors ensure minimum kept docs. Falls back to lexical
        ranking on LLM failure.
        """
        if not docs:
            return [], [], []

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
        is_simple = getattr(profile, "question_type", "mechanism") in ("definition", "factual")
        min_full  = 2 if is_simple else 3
        min_total = max(min_keep, n // 3)

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
                "filter_documents: %s — falling back to lexical ranking", exc
            )
            full_docs, abstract_docs, drop_docs = self._lexical_fallback(
                docs, ontology, min_full, min_total
            )
            if evidence_mode == "abstracts":
                full_docs, abstract_docs, drop_docs = self._abstracts_topup(
                    full_docs, abstract_docs, drop_docs, ontology, min_keep
                )
            return full_docs, abstract_docs, drop_docs

        full_docs     = [docs[i] for i, c in enumerate(classifications) if c == "full"]
        abstract_docs = [docs[i] for i, c in enumerate(classifications) if c == "abstract"]
        drop_docs     = [docs[i] for i, c in enumerate(classifications) if c == "drop"]

        self.logger.info(
            "filter_documents: %d full, %d abstract, %d dropped (of %d)",
            len(full_docs), len(abstract_docs), len(drop_docs), n,
        )

        if evidence_mode == "abstracts":
            full_docs, abstract_docs, drop_docs = self._abstracts_topup(
                full_docs, abstract_docs, drop_docs, ontology, min_keep
            )

        return full_docs, abstract_docs, drop_docs


# backward-compat alias
OntologyConstructionAgent = OntologyAgent
