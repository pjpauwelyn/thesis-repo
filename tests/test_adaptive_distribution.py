# manual adaptiveness check -- confirms rule distribution is non-degenerate
# across the 70 real DLR questions using heuristic profile synthesis.
#
# this is a manual/CI-lite test. it does NOT call the LLM; instead we build a
# plausible QuestionProfile per question based on keyword heuristics and feed
# it through Router to verify the rule histogram is healthy.

from __future__ import annotations

import csv
import re
from pathlib import Path

from core.policy.router import Router
from core.utils.data_models import QuestionProfile

csv.field_size_limit(int(1e8))

_DLR_CSV = Path("data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv")


def _heuristic_profile(q: str) -> QuestionProfile:
    """build a plausible profile from surface features of the question text.
    this stands in for a real llm profiler for distribution testing only.
    """
    ql = q.lower()
    words = ql.split()

    # question type
    if re.search(r"\b(what is|define|definition of|meaning of)\b", ql):
        qtype = "definition"
    elif re.search(r"\b(how does|how do|mechanism|why does|explain why)\b", ql) \
            and any(w in ql for w in ["work", "affect", "cause", "influence", "interact", "drive", "trigger"]):
        qtype = "mechanism"
    elif re.search(r"\b(compare|versus|vs\.?|differ|difference between|contrast)\b", ql):
        qtype = "comparison"
    elif re.search(r"\b(how much|how many|what fraction|percentage|percent|rate of|by how|trend|over the past|between \d{4}|since \d{4}|by \d{4})\b", ql):
        qtype = "quantitative"
    elif re.search(r"\b(how effective|how accurate|evaluate|performance|precision|accuracy|validate|compare methods)\b", ql):
        qtype = "method_eval"
    elif re.search(r"\b(applied|application|used in|use case|used for)\b", ql):
        qtype = "application"
    else:
        qtype = "continuous"

    # quantitativity: numbers, percentages, years, quantitative phrasing
    quant_hits = 0
    if re.search(r"\b\d{2,4}\b", ql): quant_hits += 1
    if re.search(r"%|percent|percentage|fraction|rate|trend|magnitude|amount|volume", ql): quant_hits += 1
    if qtype == "quantitative": quant_hits += 1
    quantitativity = min(1.0, 0.15 + 0.25 * quant_hits)

    # spatial specificity: named regions
    spatial_hits = sum(
        1 for kw in ["arctic", "antarctic", "europe", "africa", "asia", "amazon",
                     "mediterranean", "pacific", "atlantic", "sahel", "tropics",
                     "north", "south", "region", "country"] if kw in ql
    )
    spatial = min(1.0, 0.1 + 0.25 * spatial_hits)

    # temporal specificity: year ranges, decades
    temp = 0.1
    if re.search(r"\b(19|20)\d{2}\b", ql): temp += 0.4
    if re.search(r"\b(decade|past \d+ years|since|between \d{4})\b", ql): temp += 0.3
    temp = min(1.0, temp)

    # complexity: length, conjunctions, mechanism/method markers
    n_conj = sum(ql.count(kw) for kw in [" and ", " or ", " while ", " whereas ", " however ", " but "])
    complexity = min(1.0, 0.25 + 0.015 * len(words) + 0.1 * n_conj)

    # methodological depth
    meth = 0.1
    if any(kw in ql for kw in ["method", "approach", "technique", "algorithm", "model", "framework"]):
        meth = 0.5
    if any(kw in ql for kw in ["validate", "validation", "uncertainty", "error", "calibration"]):
        meth = min(1.0, meth + 0.3)

    return QuestionProfile(
        identity=q[:80],
        one_line_summary=q[:120],
        question_type=qtype,
        complexity=round(complexity, 2),
        quantitativity=round(quantitativity, 2),
        spatial_specificity=round(spatial, 2),
        temporal_specificity=round(temp, 2),
        methodological_depth=round(meth, 2),
        confidence=0.85,
    )


def _load_questions() -> list[str]:
    with open(_DLR_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def test_dlr_rule_distribution_is_non_degenerate():
    """over 70 real dlr questions, the router must not collapse to a single rule
    and must not route the majority to fallback or safety-tier3.
    """
    router = Router("core/policy/rules.yaml")
    questions = _load_questions()
    assert len(questions) >= 60, f"expected ~70 dlr questions, got {len(questions)}"

    hist: dict[str, int] = {}
    for q in questions:
        prof = _heuristic_profile(q)
        cfg = router.select(prof)
        hist[cfg.rule_hit] = hist.get(cfg.rule_hit, 0) + 1

    print("\nrule histogram over 70 dlr questions:")
    for rule, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {rule:30s} {n}")

    # non-degenerate: at least 3 distinct rules fire
    assert len(hist) >= 3, f"distribution collapsed to {len(hist)} rule(s): {hist}"

    # fallback must not dominate
    total = sum(hist.values())
    assert hist.get("fallback", 0) < 0.5 * total, \
        f"fallback dominates: {hist}"

    # at least two non-fallback, non-safety rules must fire on at least one q
    meaningful = [r for r in hist if r not in ("fallback", "safety_tier3")]
    assert len(meaningful) >= 2, \
        f"fewer than 2 meaningful rules fired: {hist}"


def test_varied_profiles_produce_varied_configs():
    """a grid of varied profiles must produce >= 4 distinct configs."""
    router = Router("core/policy/rules.yaml")

    scenarios = [
        # simple definition
        dict(identity="d", question_type="definition", complexity=0.2,
             quantitativity=0.1, confidence=0.9),
        # quantitative global trend
        dict(identity="q", question_type="quantitative", complexity=0.5,
             quantitativity=0.85, spatial_specificity=0.2, confidence=0.9),
        # quantitative regional
        dict(identity="qr", question_type="quantitative", complexity=0.6,
             quantitativity=0.8, spatial_specificity=0.7, confidence=0.9),
        # mechanism heavy
        dict(identity="m", question_type="mechanism", complexity=0.85,
             methodological_depth=0.6, confidence=0.9),
        # medium conceptual
        dict(identity="c", question_type="definition", complexity=0.55,
             confidence=0.9),
        # comparison
        dict(identity="cm", question_type="comparison", complexity=0.7,
             confidence=0.9),
        # low confidence -> safety
        dict(identity="s", question_type="continuous", complexity=0.5,
             confidence=0.3),
        # no confidence (parse fail) -> safety
        dict(identity="n", question_type="continuous", complexity=0.5,
             confidence=None),
    ]

    hits: set[str] = set()
    for s in scenarios:
        prof = QuestionProfile(**s)
        cfg = router.select(prof)
        hits.add(cfg.rule_hit)

    print("\ndistinct rules from varied grid:", sorted(hits))
    assert len(hits) >= 4, f"expected >=4 distinct rules, got {hits}"
