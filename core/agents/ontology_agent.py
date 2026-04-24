"""constructs a dynamic ontology from a user question via llm calls."""

import math
import os
import re
from collections import Counter
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
        "high quantitativity",
        "A quantitative question requiring numeric evidence. Papers with measured "
        "data, estimates, or observations for the specific variable are strong 'full' "
        "candidates. Supporting context papers can stay 'abstract'.",
    ),
    "tier-2b": (
        "quantitative focal",
        "A quantitative question with strong spatial or temporal focus. Papers "
        "covering the specific region/period with measured data are worth 'full'. "
        "Partial-overlap papers (cover one dimension well) can be 'abstract'.",
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
        "Similarity score is the most reliable signal here.",
    ),
    "fallback": (
        "conservative fallback",
        "No routing rule matched confidently. Prefer 'abstract' over 'drop' unless "
        "the paper is clearly off-topic.",
    ),
}

_TIER_GUIDANCE_DEFAULT = (
    "moderate",
    "Prefer 'abstract' over 'drop' unless the paper is clearly off-topic.",
)

# rule_hit values that indicate a simple/definitional routing outcome.
# used to set min_full=2 instead of 3 (fewer pdfs needed for shallow questions).
_SIMPLE_RULE_HITS = frozenset({"tier-1", "tier-1-def"})

_COMPARISON_TYPES = frozenset({"comparison", "method_eval"})

# centrality threshold above which an av-pair value/attribute match in a doc
# title is flagged as [title-match] in the prompt to help the llm decide full.
# lowered from 0.65 → 0.50 so that moderately-central terms (e.g. "sea ice"
# in a cryosphere carbon-flux question) also trigger the flag.
_TITLE_MATCH_CENTRALITY = 0.50

# words that signal adversarial/null-result framing in a doc title.
# docs matching any of these are flagged [retain-adversarial] before the llm
# sees them — the flag overrides the llm's title-level prior so the model
# cannot silently drop contrarian-framed papers that are primary evidence.
_ADVERSARIAL_TITLE_WORDS = frozenset({
    "inexplicable", "implausible", "no evidence", "absence of",
    "refutes", "unlikely", "challenges", "contradicts",
})

# minimum number of question content-words (length >= 5, non-stopword) that
# must appear verbatim in a doc title to fire the [question-term-match] flag.
_QUESTION_TERM_MATCH_MIN = 2

# common function words filtered out before question-term matching so that
# generic terms like "earth" or "about" do not inflate hit counts.
_QUESTION_STOPWORDS = frozenset({
    "what", "which", "where", "when", "does", "this", "that", "with",
    "from", "have", "been", "their", "there", "about", "over", "into",
    "under", "between", "through", "during", "earth", "these", "those",
    "other", "such", "more", "also", "both", "each", "than", "then",
})

# phrases that indicate the question specifies a geological or long timescale
# as the scope of inquiry. when detected, a mandatory constraint block is
# injected into the filter prompt to prevent the LLM from using the timescale
# qualifier as a drop criterion for mechanistically relevant papers.
_TIMESCALE_SCOPE_PHRASES = frozenset({
    "geological timescale", "geological timescales",
    "geologic timescale", "geologic timescales",
    "over long timescales", "over long time scales",
    "over millions of years", "over geological time",
    "on geological timescales", "on geologic timescales",
})


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

    def _define_logical_relationships(
        self,
        av_pairs: List[AttributeValuePair],
        question: str,
    ) -> List[LogicalRelationship]:
        """extract logical relationships between av-pairs via llm.

        only called when include_relationships=True (non-default). the adaptive
        pipeline always passes include_relationships=False so this path is
        effectively unused at runtime — it exists for the legacy
        1_pass_with_ontology pipeline type in core/main.py.
        """
        if not av_pairs:
            return []
        try:
            template = self.load_prompt("define_relationships.txt")
        except FileNotFoundError:
            self.logger.warning(
                "define_relationships.txt not found — skipping relationship extraction"
            )
            return []

        pairs_text = "\n".join(
            f"  {av.attribute}: {av.value}" for av in av_pairs
        )
        prompt = template.replace("{question}", question).replace(
            "{pairs}", pairs_text
        )
        try:
            resp = self.llm.invoke(prompt, force_json=True)
            if not isinstance(resp, dict):
                return []
            return [
                LogicalRelationship(
                    source_attribute=r.get("source_attribute", ""),
                    source_value=r.get("source_value", ""),
                    target_attribute=r.get("target_attribute", ""),
                    target_value=r.get("target_value", ""),
                    relationship_type=r.get("relationship_type", "influences"),
                    logical_constraint=r.get("logical_constraint", ""),
                )
                for r in resp.get("relationships", [])
            ]
        except Exception as e:
            self.logger.error(f"relationship extraction failed: {e}")
            return []

    # ------------------------------------------------------------------
    # adaptive pipeline -- fused av-pair + profile extraction
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
            # against llm responses like "Mechanism" or "mechanism " that would
            # silently fail the {in: [...]} check in rules.yaml.
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

    def _tokenise(self, text: str) -> List[str]:
        """lowercase word tokens, length >= 2, no punctuation."""
        return re.findall(r"[a-z][a-z0-9\-]{1,}", text.lower())

    def _question_sim(
        self,
        question: str,
        docs: List[Dict[str, Any]],
    ) -> List[Tuple[float, int, Dict]]:
        """tf-idf cosine similarity of each doc's title+abstract against the question.

        returns list of (sim_score, original_index, doc) in original doc order.
        scores are normalised to [0, 1] by dividing by the per-doc tf-idf magnitude
        and the question tf-idf magnitude (standard cosine). the question is treated
        as a pseudo-document so rare question terms get higher idf weight.

        uses stdlib only (math, re, collections) -- no external dependencies.
        """
        corpus = [
            ((doc.get("abstract") or "") + " " + (doc.get("title") or ""))
            for doc in docs
        ]
        q_tokens = self._tokenise(question)

        # build idf over corpus + question as one extra doc
        df: Counter = Counter()
        doc_token_lists = [self._tokenise(c) for c in corpus]
        all_docs = doc_token_lists + [q_tokens]
        for toks in all_docs:
            for t in set(toks):
                df[t] += 1

        n_docs = len(all_docs)

        def idf(term: str) -> float:
            return math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1.0

        def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
            tf = Counter(tokens)
            total = len(tokens) or 1
            return {t: (c / total) * idf(t) for t, c in tf.items()}

        q_vec = tfidf_vec(q_tokens)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        results: List[Tuple[float, int, Dict]] = []
        for idx, (doc, toks) in enumerate(zip(docs, doc_token_lists)):
            d_vec = tfidf_vec(toks)
            dot = sum(q_vec.get(t, 0.0) * v for t, v in d_vec.items())
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            sim = dot / (q_norm * d_norm)
            results.append((round(sim, 3), idx, doc))

        return results

    def _lexical_score(
        self,
        docs: List[Dict[str, Any]],
        ontology: "DynamicOntology",
    ) -> List[Tuple[float, int, Dict]]:
        """score docs by keyword overlap with ontology av-pairs.

        returns a list of (score, original_index, doc) sorted by score desc,
        with original index as stable tiebreak. used only by _lexical_fallback
        and _abstracts_topup (safety-net paths).
        """
        word_re = re.compile(r"[a-z][a-z0-9\-]{1,}")

        kws: List[str] = []
        lits: List[str] = []
        for av in getattr(ontology, "attribute_value_pairs", []):
            for field_name in ("attribute", "value", "description"):
                val = getattr(av, field_name, None)
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
            lit_hits = sum(1 for lit in lits if lit and lit in text)
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
        """rank docs by keyword overlap and split into full/abstract/drop.

        used as fallback when the llm filter fails or violates guardrails.
        if all scores are 0 (empty ontology), document order is used as
        tiebreak -- no worse than the old all-full fallback but respects budget.
        """
        scored = self._lexical_score(docs, ontology)

        all_zero = all(s == 0 for s, _, _ in scored)
        if all_zero:
            self.logger.warning(
                "filter_documents lexical_fallback: all scores are 0 "
                "(empty ontology?) -- using document order as tiebreak"
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
        """top up the kept pool (full + abstract) to min_keep by promoting
        the highest-scoring dropped docs back into abstract_docs.

        called only in abstracts evidence mode, where the full/abstract split
        is irrelevant (no pdf fetch either way) and the only thing that matters
        is the total number of docs available for refinement.
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
        """llm-based document relevance filter with ontology-grounded reasoning.

        each document receives a tf-idf cosine similarity score against the
        question text (unbiased orientation signal), then the llm reasons over
        the full ontology av-pairs + question profile to classify each doc as:

          "full"     -- fetch pdf + excerpts
          "abstract" -- abstract only; default when relevance is plausible
          "drop"     -- clearly off-topic; would introduce noise

        guardrail floors (safety net, not shown as targets to the llm):
          min_full  : 2 for simple/definitional tiers, 3 for all others
          min_total : always max(min_keep, n//3)

        abstracts top-up (evidence_mode == "abstracts" only):
          after filter resolves, if kept pool < min_keep, best-scoring
          dropped docs are re-promoted via lexical ranking.

        returns (full_docs, abstract_docs, drop_docs).
        """
        if not docs:
            return [], [], []

        n = len(docs)
        is_simple = rule_hit in _SIMPLE_RULE_HITS
        min_full  = 2 if is_simple else 3
        min_total = max(min_keep, n // 3)

        # --- similarity scores (question text as reference, no ontology bias) ---
        sim_scores = self._question_sim(question, docs)
        sim_by_idx = {idx: sim for sim, idx, _ in sim_scores}

        # --- full ontology block for the prompt ---
        av_pairs = sorted(
            getattr(ontology, "attribute_value_pairs", []),
            key=lambda av: getattr(av, "centrality", 0.0),
            reverse=True,
        )
        ontology_lines = "\n".join(
            f"  [{i+1}] {getattr(av, 'attribute', '')}: {getattr(av, 'value', '')} "
            f"— {getattr(av, 'description', '')} "
            f"(centrality={getattr(av, 'centrality', 0.5):.2f})"
            for i, av in enumerate(av_pairs)
        )
        if not ontology_lines:
            ontology_lines = "  (no av-pairs extracted)"

        # --- active profile dimensions ---
        _active_dims = {
            "spatial":        getattr(profile, "spatial_specificity", 0.1),
            "temporal":       getattr(profile, "temporal_specificity", 0.1),
            "quantitative":   getattr(profile, "quantitativity", 0.3),
            "methodological": getattr(profile, "methodological_depth", 0.1),
        }
        active_dim_parts = [
            f"{dim}={val:.2f}" for dim, val in _active_dims.items() if val >= 0.2
        ]
        profile_dim_line = (
            "Active question dimensions (>=0.2): " + ", ".join(active_dim_parts)
            if active_dim_parts
            else "No strongly active dimensions — treat as a general conceptual question."
        )

        # --- build title-match set for high-centrality av-pairs ---
        # a doc gets [title-match] in the listing if a high-centrality pair's
        # value or attribute appears verbatim (case-insensitive) in its title.
        # threshold lowered to 0.50 so moderately-central terms also fire.
        high_cent_terms = [
            t.strip().lower()
            for av in av_pairs
            if getattr(av, "centrality", 0.0) >= _TITLE_MATCH_CENTRALITY
            for t in (getattr(av, "value", ""), getattr(av, "attribute", ""))
            if t and len(t.strip()) >= 4
        ]

        # --- pre-compute question content-words for [question-term-match] ---
        q_content_words = [
            w for w in re.findall(r"[a-z]{5,}", question.lower())
            if w not in _QUESTION_STOPWORDS
        ]

        # --- document list with sim scores + multi-flag injection ---
        doc_lines: List[str] = []
        for idx, doc in enumerate(docs):
            title = doc.get("title", "Unknown")
            sim = sim_by_idx.get(idx, 0.0)
            title_lower = title.lower()

            flags: List[str] = []

            # [title-match]: high-centrality ontology term in title
            if any(term in title_lower for term in high_cent_terms):
                flags.append("[title-match]")

            # [retain-adversarial]: contrarian/null-result framing in title
            if any(word in title_lower for word in _ADVERSARIAL_TITLE_WORDS):
                flags.append("[retain-adversarial]")

            # [question-term-match]: >=2 question content-words in title
            qtm_hits = sum(1 for w in q_content_words if w in title_lower)
            if qtm_hits >= _QUESTION_TERM_MATCH_MIN:
                flags.append("[question-term-match]")

            flag_str = (" " + " ".join(flags)) if flags else ""
            doc_lines.append(f"{idx + 1}. {title}{flag_str}\n   sim={sim:.3f}")
        doc_list = "\n".join(doc_lines)

        # --- tier guidance ---
        tier_label, tier_desc = _TIER_GUIDANCE.get(rule_hit, _TIER_GUIDANCE_DEFAULT)

        # --- comparison constraint block (fires for comparison / method_eval) ---
        q_type = getattr(profile, "question_type", "")
        comparison_block = ""
        if q_type in _COMPARISON_TYPES:
            comparison_block = (
                "\n⚠ COMPARISON / METHOD-EVAL QUESTION — MANDATORY CONSTRAINT:\n"
                "This question explicitly compares or contrasts two or more distinct "
                "domains, methods, or indicator sets.\n"
                "You MUST retain evidence from EACH named side of the comparison in "
                "your kept set (full + abstract).\n"
                "A kept set that covers only one side is a filter failure.\n"
                "Low sim scores on papers from the secondary domain are expected "
                "when the question vocabulary is weighted toward the primary domain — "
                "do NOT use a low sim score alone as justification for 'drop'.\n"
                "When uncertain whether a paper addresses a named comparison target, "
                "always choose 'abstract' over 'drop'.\n"
            )

        # --- timescale scope block (fires when question names a geological/long
        #     timescale as context). prevents the LLM from using the timescale
        #     qualifier as a drop criterion for mechanistically relevant papers. ---
        q_lower = question.lower()
        timescale_block = ""
        if any(phrase in q_lower for phrase in _TIMESCALE_SCOPE_PHRASES):
            timescale_block = (
                "\n⚠ TIMESCALE-SCOPED QUESTION — MANDATORY CONSTRAINT:\n"
                "This question mentions a geological or long timescale "
                "(e.g. 'geological timescales') as the context for the phenomenon.\n"
                "This is background framing, NOT a filter on paper timescale.\n"
                "Papers that study the named mechanisms (e.g. tectonic activity, "
                "geomagnetic field variation, earthquake coupling) at any timescale "
                "— including historical, instrumental, or event-scale — are directly "
                "relevant and must NOT be dropped on timescale grounds alone.\n"
                "Classify such papers as 'abstract' or 'full', never 'drop'.\n"
            )

        prompt = (
            f"You are a document relevance filter for a scientific RAG pipeline.\n\n"
            f"Question: {question}\n\n"
            f"Question type: {q_type or 'unspecified'}\n"
            f"Routing tier: {rule_hit} ({tier_label})\n"
            f"{tier_desc}\n"
            f"{comparison_block}"
            f"{timescale_block}\n"
            f"Ontology AV-pairs (all pairs, ranked by centrality):\n{ontology_lines}\n\n"
            f"{profile_dim_line}\n\n"
            f"Each document has a pre-computed similarity score (sim, 0-1) and optional\n"
            f"flags computed before this prompt:\n"
            f"  sim                    -- tf-idf cosine similarity of title+abstract vs\n"
            f"                            question text. Unbiased starting hypothesis.\n"
            f"                            Use it to form an initial view, then reason\n"
            f"                            against the full ontology and question intent.\n"
            f"  [title-match]          -- a high-centrality ontology concept appears\n"
            f"                            verbatim in this title. Strong prior for 'full'.\n"
            f"  [retain-adversarial]   -- title uses contrarian/null-result framing\n"
            f"                            (e.g. 'inexplicable', 'no evidence', 'refutes').\n"
            f"                            This paper is primary evidence — classify as\n"
            f"                            'abstract' or 'full', NOT 'drop'.\n"
            f"  [question-term-match]  -- ≥2 content-words from the question appear\n"
            f"                            verbatim in this title. Do not drop this paper\n"
            f"                            without a very specific, concrete reason.\n"
            f"  - A paper can score low on sim yet be conceptually central to the question.\n"
            f"  - A paper can score high on sim due to shared surface vocabulary while\n"
            f"    addressing a clearly different phenomenon.\n\n"
            f"Classify each document as exactly one of:\n"
            f'  "abstract" -- DEFAULT. Use when the paper is plausibly relevant or provides\n'
            f"               supporting context, but the full text is unlikely to add\n"
            f"               specific evidence beyond what the abstract reveals.\n"
            f"               This includes papers that test a mechanism named in the\n"
            f"               question and report a null, negative, or contradictory result\n"
            f"               — these are evidence, not noise.\n"
            f'  "full"     -- Use when you are confident that:\n'
            f"               (a) this paper likely contains specific data, methods, or\n"
            f"                   arguments that directly address what the question asks, AND\n"
            f"               (b) that depth is what this question's tier needs.\n"
            f"               A [title-match] flag is a strong prior for 'full'.\n"
            f"               A paper that directly measures, models, or empirically tests\n"
            f"               any mechanism or process named in the question qualifies for\n"
            f"               'full' even if its primary timescale, scale, or framing differs\n"
            f"               from the question's — mechanistic evidence is always relevant.\n"
            f"               A stated timescale or spatial scope in the question (e.g.\n"
            f"               'geological timescales', 'permafrost regions') defines the\n"
            f"               question's focus but does NOT exclude papers providing\n"
            f"               mechanistic evidence at shorter timescales or adjacent regions\n"
            f"               — classify them as 'abstract' or 'full', not 'drop'.\n"
            f'  "drop"     -- Use ONLY when ALL of the following are true:\n'
            f"               (a) the paper does not address ANY concept or domain explicitly\n"
            f"                   named in the question,\n"
            f"               (b) you can state a specific, concrete reason it is off-topic.\n"
            f"               Sharing a domain label with one side of the question does NOT\n"
            f"               justify drop if that domain is named in the question.\n"
            f"               A [retain-adversarial]-flagged paper uses contrarian framing —\n"
            f"               it is primary evidence; do NOT drop it.\n"
            f"               A stated timescale or spatial scope in the question does NOT\n"
            f"               justify dropping a paper that provides mechanistic evidence at\n"
            f"               a different timescale or adjacent region — use 'abstract'.\n"
            f"               Do not drop a paper because its conclusion differs from the\n"
            f"               question's implied premise.\n"
            f"               When in doubt between 'drop' and 'abstract', choose 'abstract'.\n\n"
            f"For each document provide a one-sentence reason for your classification.\n"
            f"This is used for pipeline debugging and thesis evaluation — be specific.\n\n"
            f"Hard constraints (safety net — do not target these as goals):\n"
            f'  - At least {min_full} "full" classifications required.\n'
            f'  - At least {min_total} combined "full"+"abstract" required.\n\n'
            f"Documents:\n{doc_list}\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{\"documents\": [{{\"classification\": \"full\"|\"abstract\"|\"drop\", '
            f'\"reason\": \"<one sentence>\"}}, ...]}}\n'
            f"No prose outside the JSON. The list length must equal {n}."
        )

        self.logger.debug("filter_documents prompt (q=%s...):\n%s", question[:60], prompt)

        try:
            raw = self.llm.invoke(prompt, force_json=True)
            self.logger.debug("filter_documents raw llm response: %s", raw)

            if raw is None:
                raise ValueError("llm unavailable after retries (returned None)")

            if isinstance(raw, str):
                import json as _json
                data = _json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                raise ValueError(f"unexpected llm response type: {type(raw)}")

            # support new format {"documents": [...]} and legacy {"classifications": [...]}
            if "documents" in data:
                entries = data["documents"]
                if not entries:
                    raise ValueError("llm returned empty documents list")
                if len(entries) != n:
                    raise ValueError(
                        f"documents count {len(entries)} != doc count {n}"
                    )
                classifications = []
                for i, e in enumerate(entries):
                    c = e.get("classification", "abstract")
                    if c not in ("full", "abstract", "drop"):
                        c = "abstract"
                    classifications.append(c)
                    self.logger.info(
                        "filter doc[%d/%d] -> %-8s | sim=%.3f | %s",
                        i + 1, n, c,
                        sim_by_idx.get(i, 0.0),
                        e.get("reason", "<no reason>"),
                    )
            elif "classifications" in data:
                classifications = data["classifications"]
                if not classifications:
                    raise ValueError("llm returned empty classifications list")
                if len(classifications) != n:
                    raise ValueError(
                        f"classification count {len(classifications)} != doc count {n}"
                    )
            else:
                raise ValueError(
                    "llm response missing both 'documents' and 'classifications' keys"
                )

            n_full  = classifications.count("full")
            n_total = n_full + classifications.count("abstract")
            self.logger.info(
                "filter_documents guardrail: n_full=%d (min %d), n_total=%d (min %d)",
                n_full, min_full, n_total, min_total,
            )
            if n_full < min_full or n_total < min_total:
                raise ValueError(
                    f"guardrail violated: n_full={n_full} (min {min_full}), "
                    f"n_total={n_total} (min {min_total})"
                )

        except Exception as exc:
            self.logger.warning(
                "filter_documents: %s -- falling back to lexical ranking", exc
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
            "filter_documents [llm]: %d full, %d abstract, %d dropped (of %d) | "
            "q_type=%s tier=%s",
            len(full_docs), len(abstract_docs), len(drop_docs), n, q_type, rule_hit,
        )

        if evidence_mode == "abstracts":
            full_docs, abstract_docs, drop_docs = self._abstracts_topup(
                full_docs, abstract_docs, drop_docs, ontology, min_keep
            )

        return full_docs, abstract_docs, drop_docs


# backward-compat alias (pre-refactor name, still referenced in some test files)
OntologyConstructionAgent = OntologyAgent
