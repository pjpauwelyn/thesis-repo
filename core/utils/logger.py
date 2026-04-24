"""Centralised logging setup for the thesis pipeline.

Two outputs:
  console  -- clean, INFO-level, human-readable for terminal use
  file     -- DEBUG-level, timestamped, full detail for post-run inspection

Usage
-----
    from core.utils.logger import get_logger, configure_pipeline_logging

    # called once at startup (main.py / pipeline.py entry points)
    configure_pipeline_logging(log_file="logs/run.log")

    # every module
    log = get_logger(__name__)
    log.info("something happened")

Generation-test helpers
-----------------------
    log_question_start(log, idx, question)
    log_ontology(log, ontology)
    log_profile_and_route(log, profile, cfg)
    log_doc_filter(log, full, abstract, drop)
    log_excerpt_stats(log, stats)
    log_refinement(log, enriched_context)
    log_generation(log, answer_obj, elapsed)
    log_question_end(log, idx, status, elapsed, error)
    log_run_summary(log, stats)

All helpers write a compact INFO line to the console and a verbose DEBUG
block to the file. They are no-ops when passed None, so call sites are
always safe.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# internal constants
# ---------------------------------------------------------------------------

_CONSOLE_FMT  = "%(message)s"
_FILE_FMT     = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_DATE_FMT     = "%Y-%m-%d %H:%M:%S"
_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "mistralai",
    "openai", "asyncio", "hpack",
)

# module-level flag so configure_pipeline_logging is idempotent
_configured = False


# ---------------------------------------------------------------------------
# public: setup
# ---------------------------------------------------------------------------

def configure_pipeline_logging(
    log_file: str = "logs/pipeline.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    suppress_noisy: bool = True,
) -> None:
    """Wire up the root logger once.  Safe to call multiple times."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # remove any handlers already attached (pytest / ipython noise)
    root.handlers.clear()

    # -- console --------------------------------------------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(_ConsoleFormatter())
    ch.addFilter(_DuplicateFilter())
    root.addHandler(ch)

    # -- file -----------------------------------------------------------------
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    root.addHandler(fh)

    # -- silence third-party noise --------------------------------------------
    if suppress_noisy:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  configure_pipeline_logging() must be called first."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# public: generation-test event helpers
# ---------------------------------------------------------------------------

def log_question_start(log: logging.Logger, idx: int, question: str) -> None:
    """Print the question banner before the pipeline runs."""
    short = question[:90] + ("\u2026" if len(question) > 90 else "")
    _console_section(log, f"Q{idx}  {short}")
    log.debug("[Q%d] question (full): %s", idx, question)


def log_ontology(
    log: logging.Logger,
    ontology: Any,          # DynamicOntology | None
    elapsed: float = 0.0,
) -> None:
    """Log ontology AV-pairs + relationships."""
    if ontology is None:
        log.info("  \u25cb ontology  skipped")
        return

    pairs = getattr(ontology, "attribute_value_pairs", [])
    rels  = getattr(ontology, "logical_relationships", [])
    log.info(
        "  \u2713 ontology  %d attrs, %d rels  (%.1fs)",
        len(pairs), len(rels), elapsed,
    )

    # verbose file detail
    lines = [f"[ontology] {len(pairs)} AV-pairs (elapsed={elapsed:.2f}s)"]
    for av in pairs:
        lines.append(
            f"  attr={av.attribute!r:30s}  val={av.value!r:25s}  "
            f"centrality={getattr(av,'centrality',0):.2f}"
        )
    if rels:
        lines.append(f"  {len(rels)} relationships:")
        for r in rels:
            lines.append(
                f"    {r.source_attribute} --[{r.relationship_type}]--> {r.target_attribute}"
            )
    log.debug("\n".join(lines))


def log_profile_and_route(
    log: logging.Logger,
    profile: Any,           # QuestionProfile | None
    cfg: Any,               # PipelineConfig | None
    elapsed: float = 0.0,
) -> None:
    """Log profiler output and routing decision."""
    if profile is None or cfg is None:
        log.info("  \u25cb routing   skipped (no profile/config)")
        return

    conf_str = (
        f"{profile.confidence:.2f}"
        if getattr(profile, "confidence", None) is not None
        else "None"
    )
    rule  = getattr(cfg, "rule_hit", "?")
    model = getattr(cfg, "model_name", "?")
    mode  = getattr(cfg, "evidence_mode", "?")

    log.info(
        "  \u2713 routing   rule=%-14s model=%-26s evidence=%-16s conf=%s  (%.1fs)",
        rule, model, mode, conf_str, elapsed,
    )

    # tuning flags
    _warn_routing_flags(log, profile, cfg)

    # verbose file detail
    log.debug(
        "[profile+route]\n"
        "  question_type=%s  complexity=%.2f  quantitativity=%.2f\n"
        "  spatial=%.2f  temporal=%.2f  methodological=%.2f\n"
        "  needs_numeric=%s  confidence=%s\n"
        "  \u2192 rule=%-12s  model=%s  evidence=%s\n"
        "  reason: %s",
        getattr(profile, "question_type", "?"),
        getattr(profile, "complexity", 0),
        getattr(profile, "quantitativity", 0),
        getattr(profile, "spatial_specificity", 0),
        getattr(profile, "temporal_specificity", 0),
        getattr(profile, "methodological_depth", 0),
        getattr(profile, "needs_numeric_emphasis", False),
        conf_str,
        rule, model, mode,
        getattr(cfg, "reason", ""),
    )


def log_doc_filter(
    log: logging.Logger,
    full_docs: List[Any],
    abstract_docs: List[Any],
    drop_docs: List[Any],
    elapsed: float = 0.0,
) -> None:
    """Log document filter results with per-document breakdown in the file."""
    n_total = len(full_docs) + len(abstract_docs) + len(drop_docs)
    log.info(
        "  \u2713 doc-filter full=%-3d abstract=%-3d drop=%-3d  total=%d  (%.1fs)",
        len(full_docs), len(abstract_docs), len(drop_docs), n_total, elapsed,
    )

    # tuning flag: high drop rate
    if n_total and len(drop_docs) / n_total > 0.5:
        log.warning(
            "  \u26a0 doc-filter drop-rate %.0f%% -- check ontology quality or min_keep",
            100 * len(drop_docs) / n_total,
        )

    lines = [f"[doc-filter] {n_total} total  full={len(full_docs)}  "
             f"abstract={len(abstract_docs)}  drop={len(drop_docs)}"]
    for d in full_docs:
        lines.append(f"  [FULL]     {_doc_title(d)[:90]}")
    for d in abstract_docs:
        lines.append(f"  [ABSTRACT] {_doc_title(d)[:90]}")
    for d in drop_docs:
        lines.append(f"  [DROP]     {_doc_title(d)[:90]}")
    log.debug("\n".join(lines))


def log_excerpt_stats(
    log: logging.Logger,
    stats: Dict[str, Any],
    elapsed: float = 0.0,
) -> None:
    """Log excerpt selection stats from FullTextIndexer."""
    if not stats:
        log.info("  \u25cb excerpts  none (abstracts mode)")
        return

    n_exc   = stats.get("n_excerpts", 0)
    n_docs  = stats.get("n_docs", 0)
    tot_ch  = stats.get("total_chars", 0)
    budget  = stats.get("global_budget", 0)
    pct     = (100 * tot_ch / budget) if budget else 0

    log.info(
        "  \u2713 excerpts  %d excerpts from %d docs  %d chars (%.0f%% of budget)  (%.1fs)",
        n_exc, n_docs, tot_ch, pct, elapsed,
    )

    # tuning flag: budget nearly full
    if pct > 90:
        log.warning(
            "  \u26a0 excerpts  %.0f%% of budget used -- consider raising global_budget", pct
        )
    # tuning flag: very few excerpts
    if n_exc < 3 and n_docs > 0:
        log.warning(
            "  \u26a0 excerpts  only %d excerpt(s) -- top_k_per_doc or per_doc_budget may be too low",
            n_exc,
        )

    log.debug("[excerpt-stats] %s", stats)


def log_refinement(
    log: logging.Logger,
    enriched_context: str,
    elapsed: float = 0.0,
) -> None:
    """Log refinement output size with tuning hints."""
    n_chars  = len(enriched_context or "")
    n_tokens = n_chars // 4

    log.info(
        "  \u2713 refinement %d chars (~%d tokens)  (%.1fs)",
        n_chars, n_tokens, elapsed,
    )

    # tuning flags
    if n_chars == 0:
        log.warning("  \u26a0 refinement produced EMPTY context -- generation will fail")
    elif n_chars < 500:
        log.warning(
            "  \u26a0 refinement context very short (%d chars) -- answer may be shallow", n_chars
        )
    elif n_tokens > 70_000:
        log.warning(
            "  \u26a0 refinement context large (%d tokens) -- approaching 60%% window cap",
            n_tokens,
        )

    # write first 600 chars of enriched context to file for inspection
    preview = (enriched_context or "")[:600]
    log.debug(
        "[refinement] %d chars (~%d tokens) elapsed=%.2fs\npreview:\n%s%s",
        n_chars, n_tokens, elapsed,
        textwrap.indent(preview, "  "),
        "\n  \u2026" if n_chars > 600 else "",
    )


def log_generation(
    log: logging.Logger,
    answer_obj: Any,        # generation_agent.Answer | None
    elapsed: float = 0.0,
) -> None:
    """Log generation output with reference count and tuning hints."""
    if answer_obj is None:
        log.warning("  \u2717 generation  returned None")
        return

    answer   = getattr(answer_obj, "answer", "") or ""
    fmt_refs = getattr(answer_obj, "formatted_references", []) or []
    refs     = getattr(answer_obj, "references", []) or []
    n_chars  = len(answer)

    log.info(
        "  \u2713 generation %d chars  %d refs  (%.1fs)",
        n_chars, len(fmt_refs) or len(refs), elapsed,
    )

    # tuning flags
    if n_chars < 100:
        log.warning(
            "  \u26a0 generation answer short (%d chars) -- prompt or context may need tuning",
            n_chars,
        )
    if not fmt_refs and not refs:
        log.warning("  \u26a0 generation no references extracted -- check reference parsing")

    # full answer + refs to file
    ref_block = "\n".join(fmt_refs or refs) if (fmt_refs or refs) else "(none)"
    log.debug(
        "[generation] %d chars  elapsed=%.2fs\nanswer:\n%s\nreferences:\n%s",
        n_chars, elapsed,
        textwrap.indent(answer, "  "),
        textwrap.indent(ref_block, "  "),
    )


def log_question_end(
    log: logging.Logger,
    idx: int,
    status: str,                    # "success" | "error" | "skipped"
    elapsed: float,
    error: Optional[str] = None,
) -> None:
    """Print the question-level result line."""
    if status == "success":
        log.info("  \u2192 Q%d  %-8s  %.2fs", idx, status.upper(), elapsed)
    elif status == "error":
        short_err = (error or "unknown")[:120]
        log.error("  \u2192 Q%d  %-8s  %.2fs  %s", idx, status.upper(), elapsed, short_err)
        log.debug("[Q%d error detail] %s", idx, error)
    else:
        log.info("  \u2192 Q%d  %-8s  %.2fs", idx, status.upper(), elapsed)


def log_run_summary(
    log: logging.Logger,
    stats: Any,             # PipelineStats | dict | object with .total/.successful/.failed/.avg_time
) -> None:
    """Print the end-of-run summary banner."""
    total   = getattr(stats, "total",      stats.get("total",      0) if isinstance(stats, dict) else 0)
    ok      = getattr(stats, "successful", stats.get("successful", 0) if isinstance(stats, dict) else 0)
    failed  = getattr(stats, "failed",     stats.get("failed",     0) if isinstance(stats, dict) else 0)
    avg     = getattr(stats, "avg_time",   stats.get("avg_time",   0.0) if isinstance(stats, dict) else 0.0)

    _console_section(log, "RUN SUMMARY")
    log.info("  total=%-4d  ok=%-4d  failed=%-4d  avg=%.2fs", total, ok, failed, avg)
    if failed:
        log.warning("  \u26a0 %d question(s) failed -- check logs/pipeline.log for details", failed)


def log_llm_retry(
    log: logging.Logger,
    attempt: int,
    max_attempts: int,
    error: str,
    delay: float,
) -> None:
    """Log LLM retry events (called from helpers.py wrappers)."""
    log.warning(
        "  \u26a0 LLM retry %d/%d  sleeping=%.1fs  reason: %s",
        attempt, max_attempts, delay, error[:120],
    )
    log.debug("[llm-retry] attempt=%d/%d delay=%.2f error=%s", attempt, max_attempts, delay, error)


def log_llm_failure(
    log: logging.Logger,
    attempt: int,
    max_attempts: int,
    error: str,
) -> None:
    """Log a final LLM failure after all retries exhausted."""
    log.error(
        "  \u2717 LLM failed after %d/%d attempts: %s",
        attempt, max_attempts, error[:200],
    )
    log.debug("[llm-failure] attempt=%d/%d error=%s", attempt, max_attempts, error)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _warn_routing_flags(
    log: logging.Logger,
    profile: Any,
    cfg: Any,
) -> None:
    """Emit WARNING log lines for actionable routing situations.

    Called immediately after the INFO routing line in log_profile_and_route().
    Two situations are flagged:
      1. safety-tier3 fired (profiler confidence was below the floor or None)
         -- indicates a question the profiler could not parse reliably.
      2. Confidence is valid but below 0.70 (borderline routing territory)
         -- worth noting in test logs as a tuning signal.
    Neither flag changes any pipeline behaviour; they are observability only.
    """
    rule       = getattr(cfg, "rule_hit", "") or ""
    confidence = getattr(profile, "confidence", None)

    if rule == "safety-tier3":
        log.warning(
            "  \u26a0 routing   safety-tier3 fired (low/null profiler confidence=%.2f) "
            "-- question will use large model + full excerpts",
            confidence if confidence is not None else 0.0,
        )
    elif confidence is not None and confidence < 0.70:
        log.warning(
            "  \u26a0 routing   borderline confidence=%.2f for rule=%s "
            "-- profile may be ambiguous",
            confidence, rule,
        )


class _ConsoleFormatter(logging.Formatter):
    """Coloured, compact console formatter.  Falls back gracefully if no TTY."""

    _GREY   = "\033[38;5;245m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _use_color: Optional[bool] = None

    @classmethod
    def _color(cls) -> bool:
        if cls._use_color is None:
            cls._use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        return cls._use_color

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        msg = record.getMessage()
        if not self._color():
            return msg
        if record.levelno >= logging.ERROR:
            return f"{self._RED}{msg}{self._RESET}"
        if record.levelno >= logging.WARNING:
            return f"{self._YELLOW}{msg}{self._RESET}"
        if record.levelno == logging.DEBUG:
            return f"{self._GREY}{msg}{self._RESET}"
        return msg


class _DuplicateFilter(logging.Filter):
    """Suppress back-to-back identical messages (prevents log spam on retries)."""
    _last: Optional[tuple] = None

    def filter(self, record: logging.LogRecord) -> bool:
        current = (record.name, record.levelno, record.getMessage())
        if current == self._last:
            return False
        self._last = current
        return True


def _console_section(log: logging.Logger, title: str) -> None:
    bar = "\u2500" * max(0, 76 - len(title))
    log.info("\n\u2500\u2500 %s %s", title, bar)


def _doc_title(doc: Any) -> str:
    if isinstance(doc, dict):
        return doc.get("title") or doc.get("title_or_name") or "(untitled)"
    return getattr(doc, "title", None) or getattr(doc, "title_or_name", None) or "(untitled)"
