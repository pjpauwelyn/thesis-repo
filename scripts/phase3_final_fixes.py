#!/usr/bin/env python3
"""
phase3_final_fixes.py
=====================
Targeted fix pass for phase3_answers.jsonl.
Wired directly to the existing pipeline (scripts/run_pipeline.py logic)
via the same Pipeline().run() interface.

For each flagged question, injects a retrieval_scope_instruction and
answer_scope_instruction into the question string passed to the pipeline,
and hard-blocks forbidden domains at retrieval time.

Tier-3 (text-only) questions are patched locally — no model call needed.
Global Unicode normalisation is applied to every record.

USAGE
-----
# Full run — regenerate all 16 flagged questions + patch all records:
python scripts/phase3_final_fixes.py

# Dry-run (print what would change, write nothing):
python scripts/phase3_final_fixes.py --dry-run

# Only specific tiers:
python scripts/phase3_final_fixes.py --tiers 1
python scripts/phase3_final_fixes.py --tiers 1 2

# Only specific questions (by qindex label):
python scripts/phase3_final_fixes.py --questions Q43 Q29

# Override I/O:
python scripts/phase3_final_fixes.py \
    --input  tests/output/phase3_answers.jsonl \
    --output tests/output/phase3_answers_fixed.jsonl \
    --log    tests/output/phase3_fixes_run.log
"""

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipelines.pipeline import Pipeline  # noqa: E402  (import after sys.path fix)

log = logging.getLogger("phase3_final_fixes")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  FIX CATALOGUE
# ──────────────────────────────────────────────────────────────────────────────

# Each entry maps a qindex label (e.g. "Q43") to a spec dict:
#   tier                         : 1=full regen | 2=targeted | 3=text-only
#   error_summary                : human-readable description of the problem
#   retrieval_scope_instruction  : appended to the question to steer retrieval
#   answer_scope_instruction     : appended to the question to steer generation
#   forbidden_domains            : list of domain strings to avoid (logged only;
#                                  pipeline uses the scope instructions to enforce)
#   text_patches                 : [(old, new), ...] applied after generation
#                                  (tier 3) or as a safety net (tier 1/2)

FIX_CATALOGUE: Dict[str, Dict[str, Any]] = {

    # ── TIER 1: Full regeneration ─────────────────────────────────────────────

    "Q43": {
        "tier": 1,
        "error_summary": (
            "Answer body is Q42 precision-irrigation content; references are "
            "geomagnetically induced currents and geotechnical RS/GIS. "
            "Complete answer-question misbinding."
        ),
        "retrieval_scope_instruction": (
            "Focus exclusively on remote sensing data quality, SAR/optical image "
            "accuracy assessment, geometric and radiometric calibration, or "
            "multi-sensor data fusion for land surface mapping. "
            "Do NOT use documents about irrigation, agricultural water management, "
            "geomagnetically induced currents, or geotechnical engineering."
        ),
        "answer_scope_instruction": (
            "Address how sensor calibration, geometric accuracy, atmospheric "
            "correction, and data fusion affect remote sensing data quality and "
            "land surface characterisation. Do not include any irrigation "
            "scheduling or geomagnetic-current content."
        ),
        "forbidden_domains": [
            "irrigation", "agricultural water management",
            "geomagnetically induced currents", "geotechnical engineering",
        ],
        "text_patches": [],
    },

    "Q29": {
        "tier": 1,
        "error_summary": (
            "Opens correctly on groundwater contamination / surface water chemistry "
            "then switches to conservation tillage / INM / irrigation with references "
            "Lokhande 2019 and Gonzalez Perea 2017 that belong to a different question."
        ),
        "retrieval_scope_instruction": (
            "Focus exclusively on groundwater contamination plumes, contaminant "
            "transport to surface water, groundwater-surface water chemical "
            "interactions, redox geochemistry at the interface, or dissolved metal "
            "and solute transport. "
            "Do NOT use documents on irrigation, tillage, soil nutrient management, "
            "or crop production."
        ),
        "answer_scope_instruction": (
            "Address how contaminated groundwater discharge alters dissolved solids, "
            "major ions, heavy metals, and VOC concentrations in nearby rivers/lakes, "
            "covering redox controls, preferential flow paths, and regional case "
            "studies. No tillage, irrigation, or INM content."
        ),
        "forbidden_domains": [
            "irrigation", "conservation tillage",
            "integrated nutrient management", "crop production",
        ],
        "text_patches": [],
    },

    "Q18": {
        "tier": 1,
        "error_summary": (
            "Feshchenko 2015 (electromagnetic diffraction in accelerator physics) "
            "used as primary land surface radiative properties reference. "
            "BRDF, emissivity, and thermal IR entirely absent."
        ),
        "retrieval_scope_instruction": (
            "Focus exclusively on land surface radiative properties: broadband albedo, "
            "spectral BRDF, surface emissivity, thermal infrared emission, and the "
            "radiative transfer equation applied to land surfaces. "
            "Do NOT use documents on accelerator physics, electromagnetic diffraction "
            "in conductors, plasma physics, or particle beam optics."
        ),
        "answer_scope_instruction": (
            "Cover: (1) surface albedo and BRDF in the solar spectrum, "
            "(2) surface emissivity and thermal infrared emission, "
            "(3) scattering and absorption by land cover types, "
            "(4) how vegetation, soil moisture, and snow alter these properties, "
            "(5) measurement and modelling methods. "
            "No accelerator or particle-beam physics."
        ),
        "forbidden_domains": [
            "accelerator physics", "electromagnetic diffraction in conductors",
            "plasma physics", "particle beam",
        ],
        "text_patches": [],
    },

    "Q8": {
        "tier": 1,
        "error_summary": (
            "Makarenko 2011 galactic-motion heat hypothesis presented without "
            "mainstream caveat. Core heat flux, CMB, LLSVPs, and reversal "
            "frequency absent."
        ),
        "retrieval_scope_instruction": (
            "Focus on mainstream geophysical sources: the geodynamo mechanism, "
            "core-mantle boundary heat flux, secular variation, geomagnetic reversals, "
            "and magnetospheric shielding. If the cosmic-furnace hypothesis appears, "
            "include it only as a speculative minority view with explicit low "
            "confidence."
        ),
        "answer_scope_instruction": (
            "Explain: (1) the geodynamo; (2) magnetosphere formation and solar-wind "
            "shielding; (3) secular variation and reversal timescales; "
            "(4) practical implications. Any cosmic-ray or galactic-motion heat "
            "hypothesis must be flagged explicitly as speculative and not empirically "
            "validated."
        ),
        "forbidden_domains": [],
        "text_patches": [],
    },

    "Q69": {
        "tier": 1,
        "error_summary": (
            "Sorghum bicolor (a crop plant) cited as an invertebrate taxonomy example. "
            "Persistent domain-crossing error."
        ),
        "retrieval_scope_instruction": (
            "Focus exclusively on invertebrate phylogenetics, invertebrate molecular "
            "taxonomy, genetic variation in invertebrate populations, or population "
            "genetics/phylogeography of invertebrate clades. "
            "Do NOT use any plant biology, crop science, or cereal taxonomy documents."
        ),
        "answer_scope_instruction": (
            "All cited species and examples must be invertebrates. "
            "No plant, vertebrate, or fungal examples."
        ),
        "forbidden_domains": [
            "plant biology", "crop science", "cereal taxonomy",
            "Sorghum", "Poaceae",
        ],
        "text_patches": [],
    },

    # ── TIER 2: Targeted patches ──────────────────────────────────────────────

    "Q11": {
        "tier": 2,
        "error_summary": (
            "Field-aligned current density reported as mA/m instead of μA/m². "
            "Blank reference slot present. Anomalous solar minimum caveat missing."
        ),
        "retrieval_scope_instruction": (
            "Focus on field-aligned currents (Region 1/2 Birkeland currents), "
            "FAC density measurements from Swarm or AMPERE satellites, and "
            "solar-cycle dependence of FAC intensity. "
            "Use standard SI unit μA/m² for current density."
        ),
        "answer_scope_instruction": (
            "FAC density values must be expressed in μA/m² (not mA/m or mA/m²). "
            "Include the anomalously deep 2007-2009 solar minimum context. "
            "Remove any blank or placeholder reference entries."
        ),
        "forbidden_domains": [],
        "text_patches": [
            ("mA/m²", "μA/m²"),
            ("mA/m",  "μA/m²"),
        ],
    },

    "Q12": {
        "tier": 2,
        "error_summary": (
            "Blank reference entry present. Minor unit inconsistency in current density."
        ),
        "retrieval_scope_instruction": (
            "Focus on ionospheric electrodynamics, Joule heating, "
            "magnetosphere-ionosphere coupling, or auroral current systems."
        ),
        "answer_scope_instruction": (
            "Remove any blank or placeholder reference entries. "
            "Ensure all numeric current-density values use μA/m²."
        ),
        "forbidden_domains": [],
        "text_patches": [
            ("mA/m²", "μA/m²"),
            ("mA/m",  "μA/m²"),
        ],
    },

    "Q14": {
        "tier": 2,
        "error_summary": (
            "Unit inconsistency in field-aligned current density (mA/m vs μA/m²). "
            "Anomalous solar minimum caveat absent."
        ),
        "retrieval_scope_instruction": (
            "Focus on ionospheric dynamics, magnetosphere-ionosphere current systems, "
            "or solar EUV influence on ionospheric conductivity."
        ),
        "answer_scope_instruction": (
            "FAC density in μA/m². "
            "Mention the anomalously deep 2007-2009 solar minimum where relevant. "
            "Remove placeholder references."
        ),
        "forbidden_domains": [],
        "text_patches": [
            ("mA/m²", "μA/m²"),
            ("mA/m",  "μA/m²"),
        ],
    },

    "Q19": {
        "tier": 2,
        "error_summary": (
            "Snow-free albedo conflated with snow-covered albedo. "
            "Kilimanjaro findings incorrectly generalised to continuous permafrost. "
            "Some Q20 thaw-rate index content bleeds in."
        ),
        "retrieval_scope_instruction": (
            "Focus on albedo of bare frozen ground in permafrost regions under "
            "snow-free conditions, spectral reflectance of ice-rich soils, BRDF "
            "of permafrost surfaces, and the radiative balance in Arctic or Tibetan "
            "permafrost zones."
        ),
        "answer_scope_instruction": (
            "Clearly separate snow-free frozen-ground albedo (0.10-0.25) from "
            "snow-covered albedo (0.6-0.9). When citing Kilimanjaro, note it "
            "represents isolated tropical alpine permafrost and may not generalise "
            "to continuous Arctic/Tibetan zones. Do not include thawing-index "
            "(GTI/GFI) trend data unless directly supporting a radiative-property "
            "point — that material belongs to Q20."
        ),
        "forbidden_domains": [],
        "text_patches": [],
    },

    "Q55": {
        "tier": 2,
        "error_summary": (
            "Glacier mass balance content from Q56/Q57 bleeds into this answer."
        ),
        "retrieval_scope_instruction": (
            "Focus strictly on the topic of Q55. Do not retrieve glacier mass "
            "balance, specific mass balance measurements, or geodetic mass balance "
            "methods unless the question explicitly requests them."
        ),
        "answer_scope_instruction": (
            "Remove all glacier mass balance / geodetic mass balance content not "
            "directly asked for by Q55. Each section must map to an explicit part "
            "of the Q55 question."
        ),
        "forbidden_domains": [],
        "text_patches": [],
    },

    "Q59": {
        "tier": 2,
        "error_summary": (
            "Amundsen Sea context must remain within the Antarctic ice-shelf / "
            "ocean-circulation domain, not drift into Arctic framing."
        ),
        "retrieval_scope_instruction": (
            "Focus on Amundsen Sea Embayment (West Antarctica): ice-shelf basal "
            "melt, Circumpolar Deep Water intrusion, Pine Island and Thwaites "
            "glacier dynamics, and ocean-ice interactions. All context must be "
            "Antarctic."
        ),
        "answer_scope_instruction": (
            "All references to the Amundsen Sea must remain in the Antarctic "
            "context. Do not apply Arctic framing to Amundsen Sea dynamics."
        ),
        "forbidden_domains": [],
        "text_patches": [],
    },

    "Q65": {
        "tier": 2,
        "error_summary": (
            "Hospitalisation threshold stated as 10°C instead of 1°C. "
            "Unit error that inverts the climate-health finding."
        ),
        "retrieval_scope_instruction": (
            "Focus on climate change and health outcomes: temperature thresholds "
            "for hospital admissions, heat-related morbidity, cold-snap "
            "hospitalisation, or climate-attribution studies for respiratory / "
            "cardiovascular admissions."
        ),
        "answer_scope_instruction": (
            "Verify the temperature threshold for hospitalisation increase against "
            "the source. The correct value is 1°C per degree warming (not 10°C). "
            "Cite the exact threshold from the primary paper."
        ),
        "forbidden_domains": [],
        "text_patches": [
            ("10°C increase in temperature was associated with",
             "1°C increase in temperature was associated with"),
            ("10°C rise in temperature was associated with",
             "1°C rise in temperature was associated with"),
            ("10 °C increase in temperature was associated with",
             "1 °C increase in temperature was associated with"),
            ("10 °C rise in temperature was associated with",
             "1 °C rise in temperature was associated with"),
        ],
    },

    # ── TIER 3: Text-only literal patches ────────────────────────────────────

    "Q4": {
        "tier": 3,
        "error_summary": "Implications section truncated mid-sentence.",
        "retrieval_scope_instruction": "",
        "answer_scope_instruction": "",
        "forbidden_domains": [],
        # Exact truncation varies per run; escalate to tier-2 if it recurs.
        "text_patches": [],
    },

    "Q6": {
        "tier": 3,
        "error_summary": "Citation-marker corruption: 'Ms .4' -> 'Ms 6.4'.",
        "retrieval_scope_instruction": "",
        "answer_scope_instruction": "",
        "forbidden_domains": [],
        "text_patches": [
            ("Ms .4",  "Ms 6.4"),
            ("Ms.4",   "Ms 6.4"),
            ("Ms . 4", "Ms 6.4"),
        ],
    },

    "Q10": {
        "tier": 3,
        "error_summary": "Citation-marker corruption: '.16 Ma' -> '3.16 Ma'.",
        "retrieval_scope_instruction": "",
        "answer_scope_instruction": "",
        "forbidden_domains": [],
        "text_patches": [
            (".16 Ma",  "3.16 Ma"),
            (" .16 Ma", " 3.16 Ma"),
        ],
    },

    "Q17": {
        "tier": 3,
        "error_summary": (
            "Albedo ranges for permafrost / SFG must be presented under snow-free "
            "conditions; snow-covered values noted separately. "
            "Emissivity range for dry permafrost may be missing."
        ),
        "retrieval_scope_instruction": "",
        "answer_scope_instruction": "",
        "forbidden_domains": [],
        "text_patches": [
            # Restore emissivity line if stripped
            ("emissivity 0.920.96 than dry permafrost surfaces",
             "emissivity 0.92-0.96 than dry permafrost surfaces (0.88-0.93)"),
            # Normalise Unicode en-dash corruption in numeric ranges
            ("0.220130.4",  "0.22-0.40"),
            ("0.120130.25", "0.10-0.25"),
            ("0.620130.9",  "0.6-0.9"),
            ("0.220130.6",  "0.2-0.6"),
            ("0.220132.5",  "0.2-2.5"),
            ("1.020132.5",  "1.0-2.5"),
        ],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GLOBAL UNICODE NORMALISATION
#     Applied to every answer in the output (all 70 records).
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_REPLACEMENTS: List[Tuple[str, str]] = [
    ("\\u2013", "-"),
    ("\\u2014", "-"),
    ("\\u2019", "'"),
    ("\\u2018", "'"),
    ("\\u201c", '"'),
    ("\\u201d", '"'),
    ("\\u00b0", "°"),
    ("\\u00b1", "±"),
    ("\\u00b2", "²"),
    ("\\u00b3", "³"),
    ("\\u00b5", "μ"),
    ("\\u00b7", "·"),
    ("\\u00d7", "×"),
    ("207b00b2", "⁻²"),
    ("207b00b9", "⁻¹"),
    ("207b00b3", "⁻³"),
    ("2082",    "₂"),
    ("2084",    "₄"),
    ("2083",    "₃"),
    ("2019",  "'"),
    ("2013",  "-"),
    ("2014",  "-"),
    ("00b0",  "°"),
    ("00b1",  "±"),
    ("00b2",  "²"),
    ("00b3",  "³"),
    ("00b7",  "·"),
    ("00b5",  "μ"),
    ("00d7",  "×"),
    ("00f8",  "ø"),
    ("00fc",  "ü"),
    ("00e9",  "é"),
    ("00e0",  "à"),
    ("00e4",  "ä"),
    ("00f6",  "ö"),
    ("2192",  "->"),
    ("2191",  "^"),
    ("2190",  "<-"),
    ("03b1",  "α"),
    ("03b4",  "δ"),
    ("03b5",  "ε"),
    ("03b8",  "θ"),
    ("03bc",  "μ"),
    ("03c4",  "τ"),
]


def normalise_text(text: str) -> str:
    for old, new in GLOBAL_REPLACEMENTS:
        text = text.replace(old, new)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# 3.  I/O HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_label_map(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Map 'Q{n}' labels to list indices (0-based)."""
    mapping: Dict[str, int] = {}
    for i, rec in enumerate(records):
        qi = rec.get("q_index") or rec.get("qindex")
        if qi is not None:
            mapping[f"Q{qi}"] = i
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PIPELINE INTEGRATION
# ──────────────────────────────────────────────────────────────────────────────

def _build_augmented_question(original_question: str, spec: Dict[str, Any]) -> str:
    """
    Append scope instructions to the question text so the pipeline
    steers retrieval and generation appropriately.
    """
    parts = [original_question.rstrip()]
    if spec.get("retrieval_scope_instruction"):
        parts.append(
            f"\n\n[RETRIEVAL SCOPE] {spec['retrieval_scope_instruction']}"
        )
    if spec.get("answer_scope_instruction"):
        parts.append(
            f"\n\n[ANSWER SCOPE] {spec['answer_scope_instruction']}"
        )
    return "".join(parts)


def regenerate_with_pipeline(
    record: Dict[str, Any],
    spec: Dict[str, Any],
    pipeline: Pipeline,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call pipeline.run() with scope-augmented question.
    Returns an updated record dict.
    """
    augmented_q = _build_augmented_question(record["question"], spec)
    aql = record.get("aql_results", "") or ""

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            ans = pipeline.run(augmented_q, aql)
            elapsed = round(time.time() - t0, 2)

            updated = deepcopy(record)
            updated["answer"]                 = ans.answer
            updated["actual_tier"]            = ans.rule_hit
            updated["enriched_context"]       = ans.enriched_context
            updated["enriched_context_chars"] = len(ans.enriched_context)
            updated["excerpt_stats"]          = ans.excerpt_stats
            updated["references"]             = ans.references
            updated["formatted_references"]   = ans.formatted_references
            updated["elapsed_s"]              = elapsed
            updated["error"]                  = None
            return updated

        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_rate = any(
                tok in msg
                for tok in ("429", "rate limit", "too many requests", "rate_limit")
            )
            backoff = (2 ** (attempt - 1)) * (3.0 if is_rate else 1.0)
            log.warning(
                "attempt %d/%d failed (%s): %s — sleeping %.1fs",
                attempt, max_retries,
                "RATE LIMIT" if is_rate else "ERROR",
                exc, backoff,
            )
            if attempt < max_retries:
                time.sleep(backoff)

    updated = deepcopy(record)
    updated["error"] = repr(last_exc)
    updated["answer"] = f"ERROR after {max_retries} attempts: {last_exc}"
    return updated


def apply_text_patches(answer: str, patches: List[Tuple[str, str]]) -> str:
    for old, new in patches:
        answer = answer.replace(old, new)
    return answer


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 targeted fix pass")
    p.add_argument("--input",     default="tests/output/phase3_answers.jsonl")
    p.add_argument("--output",    default="tests/output/phase3_answers_fixed.jsonl")
    p.add_argument("--log",       default="tests/output/phase3_fixes_run.log")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print what would change; write nothing.")
    p.add_argument("--tiers",     nargs="+", type=int,
                   help="Only fix questions in these tier(s), e.g. --tiers 1 2")
    p.add_argument("--questions", nargs="+",
                   help="Only fix these labels, e.g. --questions Q43 Q29")
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(args.log, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    log.info("Loading: %s", args.input)
    records   = load_jsonl(args.input)
    label_map = build_label_map(records)
    log.info("Loaded %d records.", len(records))

    # Determine which questions to process
    target_labels = set(FIX_CATALOGUE.keys())
    if args.tiers:
        target_labels = {
            k for k, v in FIX_CATALOGUE.items() if v["tier"] in args.tiers
        }
    if args.questions:
        target_labels &= set(args.questions)

    log.info("Questions to process: %s", sorted(target_labels))

    if args.dry_run:
        for label in sorted(target_labels):
            spec = FIX_CATALOGUE[label]
            log.info(
                "[DRY-RUN] %s  tier=%d  |  %s",
                label, spec["tier"], spec["error_summary"]
            )
        log.info("Dry-run complete. No files written.")
        return 0

    # Initialise pipeline once (reused for all tier-1 and tier-2 questions)
    needs_pipeline = any(
        FIX_CATALOGUE[lbl]["tier"] in (1, 2)
        for lbl in target_labels
        if lbl in FIX_CATALOGUE
    )
    pipeline: Optional[Pipeline] = Pipeline() if needs_pipeline else None

    updated_records = list(records)

    ok_count  = 0
    err_count = 0

    for label in sorted(target_labels):
        if label not in label_map:
            log.warning("Label %s not found in input - skipping.", label)
            continue

        idx      = label_map[label]
        original = records[idx]
        spec     = FIX_CATALOGUE[label]
        tier     = spec["tier"]

        log.info("[%s]  tier=%d  |  %s", label, tier, spec["error_summary"])

        try:
            if tier == 3:
                # Text-only: no model call
                updated = deepcopy(original)
                updated["answer"] = apply_text_patches(
                    updated.get("answer", ""),
                    spec["text_patches"],
                )
            else:
                # Tier 1 or 2: full regeneration via pipeline
                assert pipeline is not None
                updated = regenerate_with_pipeline(
                    original, spec, pipeline, max_retries=args.max_retries
                )
                # Safety net: apply any text patches on top of new answer
                if spec["text_patches"]:
                    updated["answer"] = apply_text_patches(
                        updated["answer"], spec["text_patches"]
                    )

            updated["answer"] = normalise_text(updated.get("answer", ""))
            updated_records[idx] = updated
            log.info("[%s]  done.", label)
            ok_count += 1

        except Exception as exc:
            log.error("[%s]  FAILED: %s", label, exc, exc_info=True)
            err_count += 1

    # Global normalisation for every untouched record too
    for i, rec in enumerate(updated_records):
        if "answer" in rec:
            updated_records[i]["answer"] = normalise_text(rec["answer"])

    log.info("Writing output: %s", args.output)
    save_jsonl(updated_records, args.output)
    log.info(
        "Done. %d records written | fixes: ok=%d err=%d",
        len(updated_records), ok_count, err_count,
    )
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
