#!/usr/bin/env python3
"""diag_cache_filter.py — two-phase diagnostic for the adaptive pipeline.

Phase 1 — Cache health audit
  Walks cache/fulltext/{openalex,pdfs,text} and reports:
    - meta / PDF / text counts and .fail marker breakdown
    - zero-byte PDFs (should be 0 after the extractor's auto-delete fix)
    - .fail files: top-10 failure reasons
    - text JSONs where extraction_ok=False (failed silently)
    - cross-check: text cached but PDF missing (orphan), PDF present but no text

Phase 2 — Filter + refinement handoff probe
  Picks 3 questions from the DLR CSV: one that will likely hit a tier requiring
  doc-filter (>=6 docs) and one shorter one.  For each:
    1. profile_and_route_with_filter()   — exercises filter_documents LLM pass
    2. Manually calls _render_documents_block() + set_documents_block()
       then calls refinement_agent.process_context()
    3. Prints a structured report: filter split (full/abstract/drop),
       documents_block head/tail, and refined context stats
  No generation step — we stop before the generation LLM call.

Usage:
    python scripts/diag_cache_filter.py                  # full run
    python scripts/diag_cache_filter.py --phase 1        # cache audit only
    python scripts/diag_cache_filter.py --phase 2        # filter probe only
    python scripts/diag_cache_filter.py --phase 2 --row 42   # specific CSV row
    python scripts/diag_cache_filter.py --cache-dir /custom/path
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# logging — INFO to stdout so tee-ing to a file is easy
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("diag")

# ---------------------------------------------------------------------------
# paths (can be overridden via CLI)
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path("cache/fulltext")
DEFAULT_CSV = Path("data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv")

# ---------------------------------------------------------------------------
# PHASE 1 — cache health
# ---------------------------------------------------------------------------

def phase1_cache_audit(cache_dir: Path) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("PHASE 1 — Full-text cache health audit")
    print(f"  cache_dir : {cache_dir.resolve()}")
    print(sep)

    meta_dir = cache_dir / "openalex"
    pdf_dir  = cache_dir / "pdfs"
    text_dir = cache_dir / "text"

    # --- existence check ---
    for d, label in [(meta_dir, "openalex"), (pdf_dir, "pdfs"), (text_dir, "text")]:
        if not d.exists():
            log.warning("directory missing: %s", d)

    n_meta  = len(list(meta_dir.glob("*.json")))  if meta_dir.exists()  else 0
    n_pdf   = len(list(pdf_dir.glob("*.pdf")))    if pdf_dir.exists()   else 0
    n_fail  = len(list(pdf_dir.glob("*.fail")))   if pdf_dir.exists()   else 0
    n_text  = len(list(text_dir.glob("*.json")))  if text_dir.exists()  else 0

    print(f"\n{'':4}{'Subdirectory':<20} {'Count':>8}")
    print(f"{'':4}{'-'*30}")
    print(f"{'':4}{'openalex/ meta':<20} {n_meta:>8}")
    print(f"{'':4}{'pdfs/ PDFs':<20} {n_pdf:>8}")
    print(f"{'':4}{'pdfs/ .fail markers':<20} {n_fail:>8}")
    print(f"{'':4}{'text/ extracted':<20} {n_text:>8}")
    print(f"{'':4}{'PDF+fail total':<20} {n_pdf+n_fail:>8}")

    # --- zero-byte PDFs ---
    zero_byte: List[Path] = []
    if pdf_dir.exists():
        for p in pdf_dir.glob("*.pdf"):
            if p.stat().st_size == 0:
                zero_byte.append(p)
    if zero_byte:
        print(f"\n  ⚠  Zero-byte PDFs ({len(zero_byte)}) — should be 0 after extractor auto-delete fix:")
        for p in zero_byte[:10]:
            print(f"       {p.name}")
        if len(zero_byte) > 10:
            print(f"       ... and {len(zero_byte)-10} more")
    else:
        print("\n  ✓  Zero-byte PDFs: 0")

    # --- .fail reason breakdown ---
    if n_fail and pdf_dir.exists():
        reason_counts: Dict[str, int] = {}
        for fp in pdf_dir.glob("*.fail"):
            try:
                reason = fp.read_text(encoding="utf-8", errors="ignore").strip()
                # bucket by first token
                bucket = reason[:60] if reason else "empty"
            except Exception:
                bucket = "unreadable"
            reason_counts[bucket] = reason_counts.get(bucket, 0) + 1
        print(f"\n  .fail breakdown (top 10):")
        for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {cnt:4d}x  {reason}")

    # --- text JSONs with extraction_ok=False ---
    n_extract_failed = 0
    extract_errors: Dict[str, int] = {}
    if text_dir.exists():
        for tp in text_dir.glob("*.json"):
            try:
                data = json.loads(tp.read_text(encoding="utf-8"))
                if not data.get("extraction_ok", True):
                    n_extract_failed += 1
                    err = (data.get("error") or "unknown")[:50]
                    extract_errors[err] = extract_errors.get(err, 0) + 1
            except Exception:
                pass
    if n_extract_failed:
        print(f"\n  ⚠  Text JSONs with extraction_ok=False: {n_extract_failed}")
        for err, cnt in sorted(extract_errors.items(), key=lambda x: -x[1])[:10]:
            print(f"    {cnt:4d}x  {err}")
    else:
        print(f"\n  ✓  Text JSONs with extraction_ok=False: 0")

    # --- cross-check: text present but PDF missing ---
    orphan_text: List[str] = []
    if text_dir.exists() and pdf_dir.exists():
        for tp in text_dir.glob("*.json"):
            wid = tp.stem
            if not (pdf_dir / f"{wid}.pdf").exists():
                orphan_text.append(wid)
    if orphan_text:
        print(f"\n  ⚠  Orphan text JSONs (text without PDF): {len(orphan_text)}")
        for wid in orphan_text[:5]:
            print(f"       {wid}")
    else:
        print(f"\n  ✓  Orphan text JSONs: 0")

    # --- PDFs without text (not yet extracted or extraction failed) ---
    pdf_no_text: List[str] = []
    if pdf_dir.exists() and text_dir.exists():
        for pp in pdf_dir.glob("*.pdf"):
            wid = pp.stem
            if not (text_dir / f"{wid}.json").exists():
                pdf_no_text.append(wid)
    if pdf_no_text:
        print(f"\n  ⚠  PDFs without extracted text: {len(pdf_no_text)}")
        print(f"       (first 5: {', '.join(pdf_no_text[:5])})")
    else:
        print(f"\n  ✓  PDFs without extracted text: 0")

    # --- summary line ---
    pct_full = 100.0 * n_text / (n_text + n_fail) if (n_text + n_fail) > 0 else 0
    print(f"\n  Summary: {n_text} fully cached, {n_fail} failed "
          f"({pct_full:.1f}% success rate)")
    print(sep)


# ---------------------------------------------------------------------------
# PHASE 2 — filter + refinement probe
# ---------------------------------------------------------------------------

def _load_csv_rows(csv_path: Path, n: int = 200) -> List[Dict[str, Any]]:
    """Load up to n rows from the DLR CSV."""
    csv.field_size_limit(int(1e8))
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
            if len(rows) >= n:
                break
    return rows


def _parse_docs(raw: str) -> List[Dict[str, Any]]:
    if not raw or not raw.strip():
        return []
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return []


def _pick_probe_rows(
    rows: List[Dict[str, Any]], specific_row: Optional[int]
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Select probe rows that cover interesting filter scenarios.
    Returns a list of (original_row_index, row) tuples.
    """
    if specific_row is not None:
        if specific_row < len(rows):
            return [(specific_row, rows[specific_row])]
        log.error("--row %d out of range (CSV has %d rows)", specific_row, len(rows))
        sys.exit(1)

    # Strategy: pick rows with >=6 docs (filter will actually fire) and
    # <=3 docs (filter is skipped), sampling spread across the CSV.
    big: List[Tuple[int, Dict]] = []
    small: List[Tuple[int, Dict]] = []
    for i, row in enumerate(rows):
        docs = _parse_docs(row.get("aql_results", ""))
        n = len(docs)
        if n >= 6 and len(big) < 2:
            big.append((i, row))
        elif 1 <= n <= 3 and len(small) < 1:
            small.append((i, row))
        if len(big) == 2 and len(small) == 1:
            break

    chosen = big + small
    if not chosen:
        # fallback: first 3 rows
        chosen = list(enumerate(rows[:3]))
    return chosen


def _summarise_documents_block(block: str, max_head: int = 600) -> str:
    lines = block.split("\n")
    head = "\n".join(lines[:30])
    tail = "\n".join(lines[-10:]) if len(lines) > 40 else ""
    n_docs   = block.count("=== Document [")
    n_excpts = block.count("<<< Section:")
    n_chars  = len(block)
    stat_line = (
        f"  [block stats] docs_in_block={n_docs}  excerpts={n_excpts}  "
        f"chars={n_chars}  lines={len(lines)}"
    )
    out = stat_line + "\n\n--- HEAD (first 30 lines) ---\n" + head
    if tail:
        out += "\n...\n--- TAIL (last 10 lines) ---\n" + tail
    return out


def phase2_filter_probe(
    cache_dir: Path,
    csv_path: Path,
    specific_row: Optional[int],
) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("PHASE 2 — Document filter + refinement handoff probe")
    print(f"  CSV       : {csv_path.resolve()}")
    print(f"  cache_dir : {cache_dir.resolve()}")
    print(sep)

    # --- imports (must be in repo root on sys.path) ---
    try:
        from core.pipelines.adaptive_pipeline import AdaptivePipeline
        from core.agents.refinement_agent_fulltext import RefinementAgent1PassFullText
        from core.utils.fulltext_indexer import FullTextIndexer
        from core.utils.helpers import get_llm_model
    except ImportError as e:
        log.error("Import failed — run from repo root with PYTHONPATH set: %s", e)
        sys.exit(1)

    rows = _load_csv_rows(csv_path, n=200)
    log.info("Loaded %d CSV rows", len(rows))

    probes = _pick_probe_rows(rows, specific_row)
    log.info("Selected %d probe row(s): indices %s",
             len(probes), [i for i, _ in probes])

    pipeline  = AdaptivePipeline(cache_dir=str(cache_dir))
    indexer   = FullTextIndexer(cache_dir=cache_dir)
    ref_llm   = get_llm_model("mistral-small-latest", temperature=0.0)

    for probe_idx, (row_idx, row) in enumerate(probes, start=1):
        question = row.get("question", "").strip()
        aql_raw  = row.get("aql_results", "")
        docs     = _parse_docs(aql_raw)

        print(f"\n{'─'*72}")
        print(f"Probe {probe_idx}/{len(probes)}  (CSV row {row_idx})")
        print(f"  Question : {textwrap.shorten(question, 120)}")
        print(f"  Docs in AQL results: {len(docs)}")

        if not docs:
            print("  ⚠  No docs parsed from aql_results — skipping probe")
            continue

        # ----------------------------------------------------------------
        # 1. Profile + route + filter (exercises the filter LLM pass)
        # ----------------------------------------------------------------
        print("\n[1] profile_and_route_with_filter ...")
        try:
            ontology, profile, cfg, filter_summary = (
                pipeline.profile_and_route_with_filter(
                    question=question,
                    docs=docs,
                    aql_results_str=aql_raw,
                )
            )
        except Exception as exc:
            log.error("profile_and_route_with_filter failed: %s", exc, exc_info=True)
            continue

        print(f"  rule_hit          : {cfg.rule_hit}")
        print(f"  model_name        : {cfg.model_name}")
        print(f"  evidence_mode     : {cfg.evidence_mode}")
        print(f"  doc_filter_min_keep: {cfg.doc_filter_min_keep}")
        print(f"\n  Profile:")
        print(f"    question_type      = {profile.question_type}")
        print(f"    complexity         = {getattr(profile,'complexity',None):.2f}"
              if profile.complexity is not None else "    complexity         = None")
        print(f"    quantitativity     = {getattr(profile,'quantitativity',None):.2f}"
              if profile.quantitativity is not None else "    quantitativity     = None")
        print(f"    spatial_specificity= {getattr(profile,'spatial_specificity',None):.2f}"
              if profile.spatial_specificity is not None else "    spatial_specificity= None")
        print(f"    confidence         = {profile.confidence}")
        print(f"    answer_shape       = {getattr(profile,'answer_shape','?')}")

        n_total    = filter_summary.get("n_total", len(docs))
        n_full_f   = filter_summary.get("n_full", 0)
        n_abstract = filter_summary.get("n_abstract", 0)
        n_drop     = filter_summary.get("n_drop", 0)
        filter_ran = n_total > cfg.doc_filter_min_keep

        print(f"\n  Filter summary (filter_ran={filter_ran}):")
        print(f"    total={n_total}  full={n_full_f}  abstract={n_abstract}  drop={n_drop}")

        if filter_ran and n_full_f == n_total and n_abstract == 0 and n_drop == 0:
            print("  ⚠  Filter returned all docs as 'full' — possible fallback "
                  "(LLM error or all docs truly relevant)")
        elif n_full_f == 0 and n_abstract == 0:
            print("  ⚠  Filter returned zero kept docs — this will produce empty context")
        else:
            print("  ✓  Filter split looks sane")

        # titles summary
        if filter_summary.get("full_titles"):
            print(f"\n  Full ({n_full_f}):")
            for t in filter_summary["full_titles"][:5]:
                print(f"    + {t}")
        if filter_summary.get("abstract_titles"):
            print(f"  Abstract ({n_abstract}):")
            for t in filter_summary["abstract_titles"][:3]:
                print(f"    ~ {t}")
        if filter_summary.get("drop_titles"):
            print(f"  Dropped ({n_drop}):")
            for t in filter_summary["drop_titles"][:3]:
                print(f"    - {t}")

        # ----------------------------------------------------------------
        # 2. Build documents_block (exercises FullTextIndexer cache lookups)
        # ----------------------------------------------------------------
        if cfg.evidence_mode == "abstracts":
            print("\n[2] evidence_mode=abstracts → skipping fulltext excerpt build")
            print("  ✓  Refinement will use plain aql_results_str (tier-1/fallback path)")
            continue

        print("\n[2] Building documents_block (FullTextIndexer) ...")
        full_docs     = [docs[i] for i, t in enumerate(docs)
                         if filter_summary.get("full_titles") is None
                         or docs[i].get("title", "")[:80] in (filter_summary.get("full_titles") or [])]
        abstract_docs = [docs[i] for i, t in enumerate(docs)
                         if docs[i].get("title", "")[:80] in (filter_summary.get("abstract_titles") or [])]

        # rebuild from filter_summary indices if title matching proves fragile
        if not full_docs and not abstract_docs:
            # fallback: use all docs as full_docs for the block test
            log.warning("Title matching for full/abstract split failed — using all docs as full")
            full_docs     = list(docs)
            abstract_docs = []

        try:
            excerpts, excerpt_stats = indexer.select_excerpts_for_question(
                question=question,
                ontology=ontology,
                documents=full_docs,
                per_doc_budget=cfg.per_doc_budget,
                global_budget=cfg.global_budget,
                top_k_per_doc=cfg.top_k_per_doc,
            )
        except Exception as exc:
            log.error("select_excerpts_for_question failed: %s", exc, exc_info=True)
            excerpts      = []
            excerpt_stats = {"error": str(exc)}

        n_docs_with_chunks = excerpt_stats.get("n_docs_with_chunks", 0)
        n_excerpts         = excerpt_stats.get("n_excerpts", 0)
        total_tokens       = excerpt_stats.get("total_tokens", 0)
        print(f"  Excerpt stats:")
        print(f"    docs_requested     = {len(full_docs)}")
        print(f"    docs_with_chunks   = {n_docs_with_chunks}")
        print(f"    n_excerpts         = {n_excerpts}")
        print(f"    total_tokens (est) = {total_tokens}")

        # per-doc status breakdown
        per_doc = excerpt_stats.get("per_doc", [])
        if per_doc:
            cache_hits   = sum(1 for d in per_doc if d.get("status") == "ok")
            paywall_hits = sum(1 for d in per_doc if d.get("status") in
                               ("already_failed", "pdf_download_failed", "no_oa_pdf"))
            other_fails  = len(per_doc) - cache_hits - paywall_hits
            print(f"    cache_hit          = {cache_hits}")
            print(f"    paywall/no_oa      = {paywall_hits}")
            print(f"    other_fail         = {other_fails}")
            # highlight docs that failed unexpectedly
            for d in per_doc:
                if d.get("status") not in ("ok", "already_failed",
                                            "pdf_download_failed", "no_oa_pdf",
                                            "not_openalex"):
                    print(f"    ⚠  [{d.get('doc_index','?')}] {d.get('title','')[:60]} → {d.get('status')}")

        if n_docs_with_chunks == 0:
            print("  ⚠  No chunks retrieved — all docs are abstract-only (paywalled or not yet cached)")
        elif n_docs_with_chunks < len(full_docs) * 0.5:
            print(f"  ⚠  <50% of full_docs returned chunks ({n_docs_with_chunks}/{len(full_docs)})")
        else:
            print(f"  ✓  Cache coverage: {n_docs_with_chunks}/{len(full_docs)} full docs have chunks")

        # build aql_lookup and documents_block
        aql_lookup = pipeline._build_aql_lookup(aql_raw, docs)
        try:
            documents_block = pipeline._render_documents_block(
                full_docs, abstract_docs, excerpts, aql_lookup
            )
        except Exception as exc:
            log.error("_render_documents_block failed: %s", exc, exc_info=True)
            continue

        print(f"\n  documents_block preview:")
        print(textwrap.indent(_summarise_documents_block(documents_block), "    "))

        # check that {documents_block} was actually populated (not empty)
        if not documents_block.strip():
            print("  ⚠  documents_block is EMPTY — refinement agent will receive no context")
        else:
            print("  ✓  documents_block is non-empty")

        # ----------------------------------------------------------------
        # 3. Refinement agent handoff (no generation)
        # ----------------------------------------------------------------
        print("\n[3] Refinement agent handoff ...")
        ref_agent = RefinementAgent1PassFullText(
            ref_llm,
            prompt_dir="prompts/refinement",
        )
        ref_agent.set_documents_block(documents_block)

        query_hint = pipeline._build_query_hint(question, profile)
        try:
            refined = ref_agent.process_context(
                question=query_hint,
                structured_context="",
                ontology=ontology,
                include_ontology=True,
                aql_results_str=None,
                context_filter="full",
            )
        except Exception as exc:
            log.error("refinement agent process_context failed: %s", exc, exc_info=True)
            continue

        enriched = refined.enriched_context or ""
        est_tokens = len(enriched) // 4
        print(f"  Refined context:")
        print(f"    chars       = {len(enriched)}")
        print(f"    tokens(est) = {est_tokens}")
        if not enriched.strip():
            print("  ⚠  enriched_context is EMPTY — generation will produce hallucinated answer")
        else:
            print("  ✓  enriched_context is non-empty")
            print(f"\n  --- enriched_context head (500 chars) ---")
            print(textwrap.indent(enriched[:500], "  "))

    print(f"\n{sep}")
    print("Diagnostic complete.")
    print(sep)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cache health + filter/refinement handoff diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase",  type=int, choices=[1, 2], default=None,
                    help="run only phase 1 (cache) or phase 2 (filter+refinement). "
                         "default: both")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                    help=f"fulltext cache directory (default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help=f"DLR questions CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--row", type=int, default=None,
                    help="probe a specific 0-based CSV row (phase 2 only)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    csv_path  = Path(args.csv)

    run_phase1 = args.phase in (None, 1)
    run_phase2 = args.phase in (None, 2)

    if run_phase1:
        phase1_cache_audit(cache_dir)

    if run_phase2:
        if not csv_path.exists():
            log.error("CSV not found: %s", csv_path)
            sys.exit(1)
        phase2_filter_probe(cache_dir, csv_path, args.row)


if __name__ == "__main__":
    main()
