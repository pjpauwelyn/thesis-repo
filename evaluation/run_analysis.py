#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_analysis.py — aggregate + visualise baseline + experimental judge runs
#
# NOTES ON AMBIGUITIES / INCONSISTENCIES IN THE EXISTING FRAMEWORK
# ---------------------------------------------------------------------------
# 1. `full_evaluation_framework.py` guards its main block under
#    `if __name__ == '__main__':`, so importing it via importlib does NOT
#    trigger the __main__ body. We therefore import `compute_summary`,
#    `compute_pairwise_tests`, and `CRITERIA` directly. No re-implementation
#    is necessary; a fallback is provided just in case the upstream module
#    layout changes.
#
# 2. The baseline `pipeline` field in all_eval_results_4pipelines.json
#    contains strings like "DLR 2-Step RAG (rag_two_steps)".
#    `PIPELINE_CONFIG[r['set']]['label']` maps `set` -> clean label, so we
#    rebuild the dataframe via `r['set']` rather than relying on the raw
#    pipeline string. For new conditions we use `r['set']` verbatim (the
#    value passed to run_judge via --pipeline).
#
# 3. `compute_pairwise_tests` in the upstream module adds `mean_a`, `mean_b`,
#    and `diff` columns beyond what the task spec asks for. We prune these
#    down to the required columns (pipeline_a, pipeline_b, criterion, W,
#    p_value, effect_r, sig) before writing the CSV, but do NOT modify the
#    original function.
#
# 4. The task spec's Wilcoxon table is per-criterion + overall per new
#    condition vs. "Ontology-Enhanced RAG", so we iterate criteria × new
#    conditions ourselves (the upstream function is designed for overall-only
#    pairs; its signature only returns one row per pair). The statistical
#    core (scipy.stats.wilcoxon) and effect-size formula are kept identical.
# ---------------------------------------------------------------------------

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FRAMEWORK_PATH = (
    _REPO_ROOT
    / "evaluation"
    / "results"
    / "FINAL_sonnet4.6_eval_results"
    / "full_evaluation_framework.py"
)
_SUMMARY_BASELINE_CSV = _FRAMEWORK_PATH.parent / "summary_4pipelines.csv"

PALETTE = {
    "Zero-Shot":             "#9E9E9E",
    "DLR 2-Step RAG":        "#2196F3",
    "No-Ontology RAG":       "#4CAF50",
    "Ontology-Enhanced RAG": "#FF9800",
}
NEW_COLORS = ["#E91E63", "#9C27B0", "#00BCD4"]

NICE = {
    "scientific_accuracy":    "Sci. Accuracy",
    "relevance":              "Relevance",
    "completeness":           "Completeness",
    "directness":             "Directness",
    "quantitative_precision": "Quant. Precision",
    "response_structure":     "Response Struct.",
    "scope_awareness":        "Scope Awareness",
    "helpfulness":            "Helpfulness",
    "overall":                "Overall",
}

LABEL_MAP = {
    "exp1_small_latest":  "Exp 1 – Small Latest",
    "exp2_large_latest":  "Exp 2 – Large Latest",
}

BASELINE_ORDER = [
    "Zero-Shot",
    "DLR 2-Step RAG",
    "No-Ontology RAG",
    "Ontology-Enhanced RAG",
]
_ONTOLOGY = "Ontology-Enhanced RAG"

# `set` id → clean pipeline label (for baseline JSON)
_SET_TO_LABEL = {
    "set1": "DLR 2-Step RAG",
    "set2": "Ontology-Enhanced RAG",
    "set3": "No-Ontology RAG",
    "set4": "Zero-Shot",
}

# ---------------------------------------------------------------------------
# import upstream framework without triggering __main__
# ---------------------------------------------------------------------------

def _import_framework():
    try:
        spec = importlib.util.spec_from_file_location("_eval_framework_ra", _FRAMEWORK_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[warn] could not import upstream framework ({e}); using local fallbacks")
        return None


_framework = _import_framework()

if _framework is not None and hasattr(_framework, "CRITERIA"):
    CRITERIA = _framework.CRITERIA
    compute_summary = _framework.compute_summary
    compute_pairwise_tests = _framework.compute_pairwise_tests
else:
    # identical re-implementation (kept bit-compatible with upstream)
    CRITERIA = [
        "scientific_accuracy", "relevance", "completeness", "directness",
        "quantitative_precision", "response_structure", "scope_awareness", "helpfulness",
    ]

    def compute_summary(df: pd.DataFrame, pipe_order: List[str]) -> pd.DataFrame:
        rows = []
        for pipe in pipe_order:
            sub = df[df["pipeline"] == pipe]
            row = {"Pipeline": pipe, "N": len(sub)}
            for c in CRITERIA:
                row[f"{c}_mean"] = round(sub[c].mean(), 2)
                row[f"{c}_std"] = round(sub[c].std(), 2)
            row["overall_mean"] = round(sub["overall"].mean(), 2)
            row["overall_std"] = round(sub["overall"].std(), 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def compute_pairwise_tests(df: pd.DataFrame, pipe_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        rows = []
        for pipe_a, pipe_b in pipe_pairs:
            da = df[df["pipeline"] == pipe_a].sort_values("question_id")["overall"].values
            db = df[df["pipeline"] == pipe_b].sort_values("question_id")["overall"].values
            try:
                w, p = scipy_stats.wilcoxon(da, db)
            except Exception:
                w, p = 0, 1
            n = len(da)
            r_eff = 1 - (2 * w) / (n * (n + 1) / 2) if n > 0 else 0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            rows.append({
                "pipeline_a": pipe_a, "pipeline_b": pipe_b,
                "mean_a": round(np.mean(da), 2), "mean_b": round(np.mean(db), 2),
                "diff": round(np.mean(db) - np.mean(da), 2),
                "W": round(w), "p_value": f"{p:.6f}", "effect_r": round(r_eff, 3), "sig": sig,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _load_baseline(path: Path) -> pd.DataFrame:
    """Load 4-pipeline baseline JSON and return a tidy dataframe."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        set_id = r.get("set")
        pipeline = _SET_TO_LABEL.get(set_id, r.get("pipeline", set_id))
        scores = r.get("scores", {}) or {}
        row = {
            "question_id": int(r["question_id"]),
            "pipeline": pipeline,
            "set": set_id,
        }
        skip = False
        for c in CRITERIA:
            v = scores.get(c)
            if v is None:
                skip = True
                break
            row[c] = v
        if skip:
            continue
        row["overall"] = round(np.mean([row[c] for c in CRITERIA]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_new(path: Path) -> Tuple[pd.DataFrame, List[int]]:
    """Load a new-condition JSON; returns (df, null_qids)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    null_qids: List[int] = []
    for r in data:
        qid = int(r["question_id"])
        pipeline = r.get("pipeline") or r.get("set")
        scores = r.get("scores", {}) or {}
        if any(scores.get(c) is None for c in CRITERIA):
            null_qids.append(qid)
            continue
        row = {"question_id": qid, "pipeline": pipeline, "set": r.get("set", pipeline)}
        for c in CRITERIA:
            row[c] = scores[c]
        row["overall"] = round(np.mean([row[c] for c in CRITERIA]), 2)
        rows.append(row)
    return pd.DataFrame(rows), null_qids


def _pipeline_name_from_path(p: Path) -> str:
    """Derive pipeline key (e.g. 'exp1_small_fulltext') from filename stem."""
    stem = p.stem
    for suffix in ("_eval", "_results"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# extended Wilcoxon table: per-criterion × (new vs Ontology-Enhanced RAG)
# ---------------------------------------------------------------------------

def _per_criterion_wilcoxon(df: pd.DataFrame, new_pipelines: List[str]) -> pd.DataFrame:
    """Two-tailed Wilcoxon signed-rank tests: each new vs Ontology-Enhanced RAG.
    One row per (new_pipeline, criterion) plus one row for overall.
    """
    rows = []
    baseline = _ONTOLOGY
    for new in new_pipelines:
        for c in CRITERIA + ["overall"]:
            da = df[df["pipeline"] == baseline].sort_values("question_id")[c].values
            db = df[df["pipeline"] == new].sort_values("question_id")[c].values
            n = min(len(da), len(db))
            if n == 0 or len(da) != len(db):
                rows.append({
                    "pipeline_a": baseline, "pipeline_b": new, "criterion": c,
                    "W": "", "p_value": "", "effect_r": "", "sig": "ns",
                })
                continue
            try:
                w, p = scipy_stats.wilcoxon(da, db, alternative="two-sided")
            except Exception:
                # e.g. all-zero differences
                w, p = 0, 1.0
            r_eff = 1 - (2 * w) / (n * (n + 1) / 2) if n > 0 else 0
            sig = (
                "***" if p < 0.001
                else "**" if p < 0.01
                else "*" if p < 0.05
                else "ns"
            )
            rows.append({
                "pipeline_a": baseline,
                "pipeline_b": new,
                "criterion": c,
                "W": round(w) if isinstance(w, (int, float)) else w,
                "p_value": f"{p:.6f}",
                "effect_r": round(r_eff, 3),
                "sig": sig,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# chart helpers
# ---------------------------------------------------------------------------

def _auto_text_color(bg_hex_or_rgb) -> str:
    """Return black/white for contrast against a matplotlib colour."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(bg_hex_or_rgb)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def _chart_bars(df, pipe_order, colors, new_set, outdir, n_per_pipeline):
    n_pipes = len(pipe_order)
    fig, ax = plt.subplots(figsize=(22, 8))
    x = np.arange(len(CRITERIA))
    total_w = 0.85
    w = total_w / max(n_pipes, 1)
    offsets = (np.arange(n_pipes) - (n_pipes - 1) / 2) * w
    for i, pipe in enumerate(pipe_order):
        sub = df[df["pipeline"] == pipe]
        means = [sub[c].mean() for c in CRITERIA]
        stds = [sub[c].std() for c in CRITERIA]
        hatch = "//" if pipe in new_set else None
        ax.bar(
            x + offsets[i], means, w, yerr=stds, label=pipe,
            color=colors.get(pipe, "#777"), capsize=2, alpha=0.9,
            hatch=hatch, edgecolor="black", linewidth=0.6,
        )
    ax.set_ylabel("Score (1–10)", fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in CRITERIA], fontsize=11)
    ax.set_title(
        f"Pipeline Comparison – All Criteria (N={n_per_pipeline} questions each)",
        fontweight="bold", fontsize=14,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10, frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"chart_{n_pipes}pipe_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _chart_heatmap(df, pipe_order, new_set, baselines_in_order, outdir):
    """Heatmap: pipelines × (criteria + overall). Diverging cmap centred on grand mean."""
    cols = CRITERIA + ["overall"]
    matrix = np.array([
        [df[df["pipeline"] == p][c].mean() for c in cols]
        for p in pipe_order
    ])
    grand_mean = np.nanmean(matrix)
    extent = float(np.nanmax(np.abs(matrix - grand_mean)))
    vmin = grand_mean - extent
    vmax = grand_mean + extent

    n_pipes = len(pipe_order)
    fig_h = max(6, n_pipes * 0.9)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([NICE[c] for c in cols], fontsize=11)
    ax.set_yticks(np.arange(n_pipes))
    y_labels = []
    for p in pipe_order:
        y_labels.append(p)
    ax.set_yticklabels(y_labels, fontsize=11)

    # italic labels for new conditions
    for i, p in enumerate(pipe_order):
        if p in new_set:
            ax.get_yticklabels()[i].set_fontstyle("italic")

    # annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            rgba = cmap((val - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=_auto_text_color(rgba),
                fontsize=11, fontweight="bold",
            )

    # thick separator line between last baseline row and first new row
    n_base = len(baselines_in_order)
    if n_base > 0 and n_base < n_pipes:
        ax.axhline(y=n_base - 0.5, color="black", linewidth=2.5)

    ax.set_title("Score Heatmap — All Pipelines × All Criteria",
                 fontweight="bold", fontsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean Score", fontsize=11)
    plt.tight_layout()
    plt.savefig(outdir / f"chart_{n_pipes}pipe_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _chart_radar(df, pipe_order, colors, new_set, outdir):
    n_pipes = len(pipe_order)
    categories = [NICE[c] for c in CRITERIA]
    n = len(categories)
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    for pipe in pipe_order:
        sub = df[df["pipeline"] == pipe]
        vals = [sub[c].mean() for c in CRITERIA]
        vals += vals[:1]
        color = colors.get(pipe, "#777")
        linestyle = "--" if pipe in new_set else "-"
        ax.plot(angles, vals, linewidth=2, linestyle=linestyle, color=color, label=pipe)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(4, 10)
    ax.set_yticks([5, 6, 7, 8, 9, 10])
    ax.set_yticklabels(["5", "6", "7", "8", "9", "10"], fontsize=9)
    ax.set_title("Radar — Per-Criterion Profile by Pipeline",
                 fontweight="bold", fontsize=14, pad=30)
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(outdir / f"chart_{n_pipes}pipe_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _chart_box(df, pipe_order, colors, outdir):
    n_pipes = len(pipe_order)
    fig, ax = plt.subplots(figsize=(14, 7))
    data = [df[df["pipeline"] == p]["overall"].values for p in pipe_order]

    bp = ax.boxplot(
        data, labels=pipe_order, patch_artist=True,
        widths=0.55, showfliers=False,
        medianprops=dict(color="black", linewidth=2.5),
    )
    for patch, pipe in zip(bp["boxes"], pipe_order):
        patch.set_facecolor(colors.get(pipe, "#777"))
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")

    # jittered points
    rng = np.random.default_rng(42)
    for i, pipe in enumerate(pipe_order, 1):
        vals = df[df["pipeline"] == pipe]["overall"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            alpha=0.3, s=16, color=colors.get(pipe, "#777"),
            edgecolor="black", linewidth=0.3,
        )

    ax.set_ylabel("Overall Score (mean of 8 criteria)", fontsize=12)
    ax.set_title("Overall Score Distribution by Pipeline",
                 fontweight="bold", fontsize=14)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(outdir / f"chart_{n_pipes}pipe_box.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# validation table
# ---------------------------------------------------------------------------

def _print_validation_table(summary_df: pd.DataFrame, baseline_labels: List[str]):
    expected = {}
    if _SUMMARY_BASELINE_CSV.exists():
        base = pd.read_csv(_SUMMARY_BASELINE_CSV)
        for _, r in base.iterrows():
            expected[r["Pipeline"]] = float(r["overall_mean"])

    print("\n" + "=" * 72)
    print("VALIDATION TABLE — computed vs. expected overall_mean")
    print("=" * 72)
    print(f"{'Pipeline':<30}{'Computed':>12}{'Expected':>12}{'Status':>16}")
    print("-" * 72)
    warning_count = 0
    for _, r in summary_df.iterrows():
        pipe = r["Pipeline"]
        comp = float(r["overall_mean"])
        if pipe in baseline_labels:
            exp = expected.get(pipe)
            if exp is None:
                status = "N/A (not in csv)"
            else:
                diff = abs(comp - exp)
                if diff > 0.01:
                    status = f"WARNING Δ={diff:.3f}"
                    warning_count += 1
                else:
                    status = f"ok (Δ={diff:.3f})"
            exp_str = "N/A" if exp is None else f"{exp:.2f}"
        else:
            exp_str = "N/A"
            status = "new condition"
        print(f"{pipe:<30}{comp:>12.2f}{exp_str:>12}{status:>16}")
    print("=" * 72)
    if warning_count:
        print(f"\n⚠  {warning_count} baseline discrepanc{'y' if warning_count==1 else 'ies'} > 0.01")
    else:
        print("\nAll baseline means match expected values within 0.01.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate + visualise baseline + experimental evals.")
    p.add_argument("--baseline", required=True,
                   help="Path to all_eval_results_4pipelines.json")
    p.add_argument("--new", nargs="+", required=True,
                   help="One or more JSON paths from run_judge.py")
    p.add_argument("--outdir", default="evaluation/results/",
                   help="Output directory (default: evaluation/results/)")
    args = p.parse_args()

    baseline_path = Path(args.baseline)
    new_paths = [Path(x) for x in args.new]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- load -----
    base_df = _load_baseline(baseline_path)

    new_frames: List[pd.DataFrame] = []
    new_labels_in_order: List[str] = []
    null_report: Dict[str, List[int]] = {}

    for idx, path in enumerate(new_paths):
        key = _pipeline_name_from_path(path)
        nice_label = LABEL_MAP.get(key, key)
        ndf, null_qids = _load_new(path)
        # rename pipeline to nice label
        if not ndf.empty:
            ndf["pipeline"] = nice_label
            ndf["set"] = nice_label
        new_frames.append(ndf)
        new_labels_in_order.append(nice_label)
        null_report[nice_label] = null_qids

    df = pd.concat([base_df] + new_frames, ignore_index=True)

    # ----- pipeline ordering + colors -----
    pipe_order = list(BASELINE_ORDER) + new_labels_in_order
    colors = dict(PALETTE)
    for i, label in enumerate(new_labels_in_order):
        colors[label] = NEW_COLORS[i % len(NEW_COLORS)]
    new_set = set(new_labels_in_order)

    # ----- summary -----
    summary_df = compute_summary(df, pipe_order)
    n_pipes = len(pipe_order)
    summary_path = outdir / f"exp_summary_{n_pipes}pipelines.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ summary → {summary_path}")
    print(summary_df.to_string(index=False))

    # ----- wilcoxon (per criterion + overall, new vs Ontology-Enhanced RAG) -----
    wilcoxon_df = _per_criterion_wilcoxon(df, new_labels_in_order)
    # keep only required columns (pipeline_a, pipeline_b, criterion, W, p_value, effect_r, sig)
    wanted = ["pipeline_a", "pipeline_b", "criterion", "W", "p_value", "effect_r", "sig"]
    wilcoxon_df = wilcoxon_df[[c for c in wanted if c in wilcoxon_df.columns]]
    wilcoxon_path = outdir / "exp_wilcoxon_tests.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"\n✓ wilcoxon → {wilcoxon_path}")
    print(wilcoxon_df.to_string(index=False))

    # ----- charts -----
    # determine N (questions per pipeline) for the bar chart title
    ns = {p: int((df["pipeline"] == p).sum()) for p in pipe_order}
    n_per_pipeline = min(ns.values()) if ns else 0

    _chart_bars(df, pipe_order, colors, new_set, outdir, n_per_pipeline)
    _chart_heatmap(df, pipe_order, new_set, BASELINE_ORDER, outdir)
    _chart_radar(df, pipe_order, colors, new_set, outdir)
    _chart_box(df, pipe_order, colors, outdir)
    print(f"\n✓ charts → {outdir}/chart_{n_pipes}pipe_*.png")

    # ----- validation -----
    _print_validation_table(summary_df, BASELINE_ORDER)

    # ----- null score report -----
    print("\nNull-score report (question_ids with any missing criterion score):")
    any_null = False
    for label, qids in null_report.items():
        if qids:
            any_null = True
            print(f"  • {label}: {len(qids)} missing → {sorted(qids)}")
    if not any_null:
        print("  (none)")


if __name__ == "__main__":
    main()
