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


# ---------------------------------------------------------------------------
# tier descriptions used by filter_documents prompt
# each entry: (label, brief_description)
# ---------------------------------------------------------------------------
_TIER_GUIDANCE: Dict[str, Tuple[str, str]] = {
    "tier-1": (
        "simple / factual",
        "A straightforward question that can usually be answered from one or two "
        "well-matched abstracts. Full PDFs rarely add value here — reserve 'full' "
        "only for papers that directly define or measure the concept being asked about.",
    ),
    "tier-1-def": (
        "definitional",
        "A pure definition question. One strong foundational paper is typically "
        "enough. Prefer 'abstract' broadly and only mark 'full' if the paper is "
        "the canonical source for the definition itself.",
    ),
    "tier-m": (
        "moderate / continuous",
        "A question requiring some cross-document synthesis. A mix of 'full' and "
        "'abstract' works well. Prioritise papers that cover the core concept with "
        "supporting data or context; general background papers are fine as 'abstract'.",
    ),
    "tier-2a": (
        "mechanistic",
        "A question about how something works or what drives it. Papers describing "
        "specific mechanisms, processes, or causal relationships are strong 'full' "
        "candidates. Supporting context papers can stay 'abstract'.",
    ),
    "tier-2b": (
        "comparative / multi-entity",
        "A comparison or multi-variable question. Papers covering each entity being "
        "compared, or that present side-by-side measurements, are worth 'full'. "
        "Partial-overlap papers (cover one side well) can be 'abstract'.",
    ),
    "tier-3": (
        "complex synthesis",
        "A high-complexity synthesis question spanning multiple concepts or domains. "
        "Be generous with 'full' for well-scored papers — this question benefits from "
        "rich, specific evidence. Even moderately scored papers may be worth 'full' "
        "if they address a distinct facet of the question.",
    ),
    "safety-tier3": (
        "uncertain complexity (safety fallback)",
        "The routing confidence was low, so this question was escalated. Treat it "
        "like a tier-2 question: collect solid evidence without over-fetching. "
        "Lexical and centrality scores are the most reliable signal here.",
    ),
}

_TIER_GUIDANCE_DEFAULT = (
    "moderate",
    "Use centrality and lexical scores as the primary signal. Prefer 'abstract' "
    "over 'drop' unless the paper is clearly off-topic.",
)


class OntologyConstructionAgent(BaseAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/ontology"):
        super().__init__("OntologyConstructionAgent", llm)
        self.prompt_dir = prompt_dir

    def load_prompt(self, filename: str) -> str:
        filepath = os.path.join(self.prompt_dir, filename)
        with open(filepath, "r", encoding="utf-8\") as f:
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

    def _lexical_score(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
    ) -> List[Tuple[float, int, Dict]]:
        """Score docs by keyword overlap with ontology AV-pairs.

        Returns a list of (score, original_index, doc) sorted by score DESC,
        with original index as stable tiebreak. Used by both _lexical_fallback
        and _abstracts_topup.
        """
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

    def _semantic_score(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
        profile: "QuestionProfile",
    ) -> List[Tuple[Dict[str, float], int, Dict]]:
        """Compute a 3-signal relevance score per doc against the ontology + profile.

        Signals (all 0-1):
          lex  -- normalised keyword overlap with AV-pair tokens (via _lexical_score)
          dim  -- profile-dimension alignment; only dimensions with a profile score
                  >= 0.2 are considered active, so low-weight dims don't add noise
          cen  -- centrality-weighted AV overlap (high-centrality pair hits count more)

        Returns list of (scores_dict, original_index, doc), in original doc order.
        The list is NOT sorted — callers rebuild sorted order themselves so the
        index mapping to the original docs list is stable.
        """
        word_re = re.compile(r"[a-z][a-z0-9\-]{1,}")

        # lex: reuse existing scorer, normalise to [0,1]
        raw_lex = self._lexical_score(docs, ontology)
        max_lex = max((s for s, _, _ in raw_lex), default=1.0) or 1.0
        lex_by_idx = {idx: s / max_lex for s, idx, _ in raw_lex}

        # dim: dimension-specific signal keywords
        _DIM_KWS: Dict[str, List[str]] = {
            "spatial": [
                "region", "area", "spatial", "geographic", "location",
                "site", "basin", "watershed", "arctic", "tropical",
                "global", "local", "scale",
            ],
            "temporal": [
                "trend", "decadal", "annual", "seasonal", "interannual",
                "long-term", "time-series", "period", "historic",
                "variability", "change",
            ],
            "quant": [
                "measurement", "estimate", "rate", "flux", "concentration",
                "data", "dataset", "observation", "satellite", "sensor",
                "numeric", "quantitative", "model", "simulation",
            ],
            "method": [
                "method", "approach", "algorithm", "retrieval", "technique",
                "framework", "validation", "calibration", "accuracy",
                "uncertainty", "assessment",
            ],
        }
        # only use dimensions that are actually meaningful for this question
        _DIM_THRESHOLD = 0.2
        dim_weights: Dict[str, float] = {}
        raw_weights = {
            "spatial":  getattr(profile, "spatial_specificity", 0.1),
            "temporal": getattr(profile, "temporal_specificity", 0.1),
            "quant":    getattr(profile, "quantitativity", 0.3),
            "method":   getattr(profile, "methodological_depth", 0.1),
        }
        for dim, w in raw_weights.items():
            if w >= _DIM_THRESHOLD:
                dim_weights[dim] = w
        total_dim_weight = sum(dim_weights.values()) or 1.0

        # cen: centrality-weighted AV overlap
        av_pairs = getattr(ontology, "attribute_value_pairs", [])
        total_centrality = sum(getattr(av, "centrality", 0.5) for av in av_pairs) or 1.0

        results: List[Tuple[Dict[str, float], int, Dict]] = []
        for idx, doc in enumerate(docs):
            text = (
                (doc.get("abstract") or "") + " " + (doc.get("title") or "")
            ).lower()
            tokens = set(word_re.findall(text))

            # dim score (only over active dimensions)
            if dim_weights:
                dim_score = 0.0
                for dim, weight in dim_weights.items():
                    hits = sum(1 for kw in _DIM_KWS[dim] if kw in tokens)
                    presence = min(hits / 3.0, 1.0)  # saturates at 3 hits
                    dim_score += weight * presence
                dim_score /= total_dim_weight
            else:
                dim_score = 0.0  # no active dimensions for this question

            # cen score
            cen_score = 0.0
            for av in av_pairs:
                cent = getattr(av, "centrality", 0.5)
                val  = (getattr(av, "value", "") or "").strip().lower()
                attr = (getattr(av, "attribute", "") or "").strip().lower()
                hit = (val and val in text) or (attr and attr in tokens)
                if hit:
                    cen_score += cent
            cen_score = min(cen_score / total_centrality, 1.0)

            scores = {
                "lex": round(lex_by_idx.get(idx, 0.0), 2),
                "dim": round(dim_score, 2),
                "cen": round(cen_score, 2),
            }
            results.append((scores, idx, doc))

        return results

    def _lexical_fallback(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
        min_full: int,
        min_total: int,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Rank docs by keyword overlap with ontology AV-pairs and split into
        full / abstract / drop using min_full and min_total as cut-points.

        Used as a fallback when the LLM filter fails or its response violates
        the guardrail. Avoids the previous behaviour of returning all docs as
        full, which defeats the purpose of filtering entirely.

        If all scores are 0 (empty ontology / no AV-pairs), the ranking is
        order-stable (first N docs become full), which is no worse than the
        old all-full fallback but respects the budget.
        """
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
        """Top up the kept pool (full + abstract) to min_keep by promoting
        the highest-scoring dropped docs back into abstract_docs.

        Called only in abstracts evidence mode, where the full/abstract split
        is irrelevant (no PDF fetch either way) and the only thing that matters
        is the total number of docs available for refinement.

        The promoted docs go into abstract_docs (not full_docs) so that the
        rest of the pipeline never tries to fetch their PDFs even if evidence
        mode changes later.
        """
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
        rule_hit: str = "fallback",
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Hybrid semantic + LLM document relevance filter.

        Each document is scored on three pre-computed signals (lex, dim, cen)
        derived from the ontology and question profile.  These scores are
        forwarded to the LLM as orientation — the LLM reasons about what the
        question genuinely needs and classifies each doc as:

          "full"     -- fetch PDF + excerpts (LLM is confident this doc
                        contains specific evidence the question needs)
          "abstract" -- abstract only; default when relevance is plausible
                        but depth is uncertain
          "drop"     -- clearly off-topic; would introduce noise

        The LLM is steered by a tier description (from rule_hit) that
        characterises how much evidence depth this question type typically
        needs.  It must justify each drop with a one-sentence reason.

        Guardrail floors (safety net, invisible to LLM):
          min_full  : 2 for simple/definitional, 3 for all others
          min_total : always max(min_keep, n//3)

        Abstracts top-up (evidence_mode == "abstracts" only):
          After LLM or lexical path resolves, if kept pool < min_keep,
          best-scoring dropped docs are re-promoted via lexical ranking.

        Returns (full_docs, abstract_docs, drop_docs).
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
            f"  {getattr(av, 'attribute', '')}: {getattr(av, 'value', '')} "
            f"(centrality={getattr(av, 'centrality', 0.5):.2f})"
            for av in pairs
        )

        n = len(docs)
        is_simple = getattr(profile, "question_type", "mechanism") in ("definition", "factual")
        min_full  = 2 if is_simple else 3
        min_total = max(min_keep, n // 3)

        # semantic scores per doc
        sem_scores = self._semantic_score(docs, ontology, profile)
        # build doc list with scores, in original index order
        doc_lines: List[str] = []
        for scores, idx, doc in sorted(sem_scores, key=lambda t: t[1]):
            title = docs[idx].get("title", "Unknown")
            doc_lines.append(
                f"{idx + 1}. {title}\n"
                f"   scores: lex={scores['lex']:.2f}  "
                f"dim={scores['dim']:.2f}  cen={scores['cen']:.2f}"
            )
        doc_list = "\n".join(doc_lines)

        # active profile dimensions (only those >= 0.2, to avoid noise)
        _active_dims = {
            "spatial":      getattr(profile, "spatial_specificity", 0.1),
            "temporal":     getattr(profile, "temporal_specificity", 0.1),
            "quantitative": getattr(profile, "quantitativity", 0.3),
            "methodological": getattr(profile, "methodological_depth", 0.1),
        }
        active_dim_parts = [
            f"{dim}={val:.2f}" for dim, val in _active_dims.items() if val >= 0.2
        ]
        profile_dim_line = (
            "Active question dimensions (≥0.2): " + ", ".join(active_dim_parts)
            if active_dim_parts
            else "No strongly active dimensions — treat as a general conceptual question."
        )

        # tier description
        tier_label, tier_desc = _TIER_GUIDANCE.get(rule_hit, _TIER_GUIDANCE_DEFAULT)

        prompt = (
            f"You are a document relevance filter for a scientific RAG pipeline.\n\n"
            f"Question: {question}\n\n"
            f"Key ontology concepts (by centrality):\n{top_concepts}\n\n"
            f"{profile_dim_line}\n\n"
            f"Routing tier: {rule_hit} ({tier_label})\n"
            f"{tier_desc}\n\n"
            f"Three pre-computed relevance signals are shown for each document (all 0–1):\n"
            f"  lex — vocabulary overlap between title/abstract and the ontology concepts above\n"
            f"  dim — how well the abstract addresses the question's active dimensions\n"
            f"        (spatial coverage, temporal scope, quantitative data, methodology)\n"
            f"  cen — how strongly the abstract addresses the highest-centrality concepts\n\n"
            f"These scores are orientation, not verdicts. Use them to form a hypothesis\n"
            f"about each paper, then apply your judgment:\n"
            f"  - A paper can score low yet be foundational to the question.\n"
            f"  - A paper can score high on shared vocabulary while being off-topic.\n\n"
            f"Classify each document as exactly one of:\n"
            f'  "abstract" — DEFAULT. Use when the paper is plausibly relevant or provides\n'
            f"               supporting context, but you are not confident the full text\n"
            f"               would add specific evidence beyond what the abstract reveals.\n"
            f'  "full"     — Use only when you are confident that:\n'
            f"               (a) this paper likely contains specific data, methods, or\n"
            f"                   arguments that directly address what the question asks, AND\n"
            f"               (b) that depth of specificity is what this question's tier needs.\n"
            f"               High lex + cen together is a strong signal; tier-1/def questions\n"
            f"               need very few full papers.\n"
            f'  "drop"     — Use only when the paper would introduce noise: it shares\n'
            f"               vocabulary with the question but addresses a clearly different\n"
            f"               phenomenon, domain, or application context.\n"
            f"               When in doubt between 'drop' and 'abstract', choose 'abstract'.\n\n"
            f"For each document, provide a one-sentence reason for your classification.\n"
            f"This is used for pipeline debugging and thesis evaluation — be specific.\n\n"
            f"Hard constraints (safety net — do not target these as goals):\n"
            f'  - At least {min_full} "full" classifications required.\n'
            f'  - At least {min_total} combined "full"+"abstract" required.\n\n'
            f"Documents (with pre-computed scores):\n{doc_list}\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{"documents": [{{"classification": "full"|"abstract"|"drop", '
            f'"reason": "<one sentence>"}}, ...]}}\n'
            f"No prose outside the JSON. The list length must equal {n}."
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

            # support both new format {"documents": [...]} and legacy {"classifications": [...]}
            if "documents" in data:
                entries = data["documents"]
                if not entries:
                    raise ValueError("LLM returned empty documents list")
                if len(entries) != n:
                    raise ValueError(
                        f"documents count {len(entries)} != doc count {n}"
                    )
                classifications = []
                for e in entries:
                    c = e.get("classification", "abstract")
                    if c not in ("full", "abstract", "drop"):
                        c = "abstract"
                    classifications.append(c)
                    reason = e.get("reason", "")
                    self.logger.debug(
                        "filter doc[%d] -> %s | %s",
                        entries.index(e) + 1, c, reason,
                    )
            elif "classifications" in data:
                # legacy flat format fallback
                classifications = data["classifications"]
                if not classifications:
                    raise ValueError("LLM returned empty classifications list")
                if len(classifications) != n:
                    raise ValueError(
                        f"classification count {len(classifications)} != doc count {n}"
                    )
            else:
                raise ValueError("LLM response missing both 'documents' and 'classifications' keys")

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


# backward-compat alias used by pipeline.py
OntologyAgent = OntologyConstructionAgent
