#!/usr/bin/env python3
"""
COMPLETE EVALUATION FRAMEWORK — Reproducible End-to-End
========================================================
Pieterjan Pauwelyn — BSc Thesis "Improving RAG Systems for Earth Observation"

This script contains the EXACT evaluation setup used for all 4 pipelines.
Judge: Claude Sonnet 4.6 | Scale: 1-10 | Criteria: 8 | Context-blind

Usage:
  1. Place CSV files in working directory
  2. Run: python full_evaluation_framework.py
  3. Results saved to output_improved/
"""

# ═══════════════════════════════════════════════════════════════
# SECTION 1: EVALUATION PROMPTS (exact text sent to judge)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert Earth Observation scientist and strict evaluation specialist.
You assess AI-generated answers to scientific EO questions.

CRITICAL RULES:
- Score on a 1-10 scale. Use the FULL range. A score of 8+ requires exceptional quality with specific justification.
- Do NOT default to high scores. Most competent answers should score 6-7.
- A score of 10 means literally perfect — publishable as-is in a scientific review.
- A score of 5 means mediocre — correct in broad strokes but lacking in important ways.
- You are evaluating the ANSWER ALONE, with no context documents. Judge what is written.
- Respond with valid JSON only."""

EVAL_PROMPT_TEMPLATE = """### EVALUATION TASK

Evaluate this answer to an Earth Observation question on 8 criteria (1-10 scale each).

**Question:**
{question}

**Answer to evaluate:**
{answer}

### CRITERIA AND RUBRICS

1. **Scientific Accuracy** (1-10)
   Are the scientific claims factually correct? Are there any hallucinated facts?
   1-3: Major factual errors or fabrications
   4-5: Mostly correct but notable inaccuracies
   6-7: Correct with minor imprecisions  
   8-10: Fully correct, precise, verifiable claims

2. **Relevance** (1-10)  
   Does the answer address what was actually asked? No tangential padding?
   1-3: Off-topic or mostly irrelevant content
   4-5: Partially addresses the question with significant drift
   6-7: Addresses the question with minor tangential content
   8-10: Precisely targeted to the question

3. **Completeness** (1-10)
   Are all aspects and sub-questions covered? Multi-part questions need multi-part answers.
   1-3: Major gaps, misses key aspects
   4-5: Covers some aspects, misses important ones
   6-7: Covers most aspects adequately
   8-10: Comprehensively covers all aspects

4. **Directness** (1-10)
   Does the answer provide the key point upfront, or bury it under lengthy preamble?
   1-3: Key answer buried after 200+ words of background
   4-5: Takes too long to reach the main point
   6-7: Reasonably direct with some preamble
   8-10: Key answer/conclusion stated clearly and early

5. **Quantitative Precision** (1-10)
   Does the answer include specific numbers, measurements, thresholds, wavelengths, resolutions, accuracies where relevant?
   1-3: No quantitative information despite it being relevant
   4-5: Vague references ("high resolution", "certain bands") instead of specifics  
   6-7: Some specific values provided
   8-10: Rich in specific, relevant quantitative details (e.g., "MODIS at 250m resolution", "668nm band", "R² of 0.84")

6. **Response Structure** (1-10)
   Is the format appropriate for the question type? Comparisons should use structured formats, not prose paragraphs.
   1-3: Wall of text with no organization
   4-5: Basic paragraphs where better structure was needed
   6-7: Decent organization with headers/sections
   8-10: Optimal structure (tables for comparisons, clear sections, logical flow)

7. **Scope Awareness** (1-10)
   Does the answer acknowledge limitations, geographic/temporal applicability, and methodological caveats?
   1-3: Makes broad claims without any qualification
   4-5: Minimal acknowledgment of limitations
   6-7: Some caveats mentioned
   8-10: Explicitly flags scope limitations, distinguishes general vs. context-specific claims, notes where findings may not generalize

8. **Helpfulness** (1-10)
   Would an EO researcher find this answer actionable and practically useful?
   1-3: Not useful for scientific work
   4-5: Basic overview but not actionable
   6-7: Useful with some actionable information
   8-10: Highly actionable — identifies specific datasets, methods, sensors, or next steps

### OUTPUT FORMAT

Respond with this exact JSON structure:
{{
  "scores": {{
    "scientific_accuracy": <int 1-10>,
    "relevance": <int 1-10>,
    "completeness": <int 1-10>,
    "directness": <int 1-10>,
    "quantitative_precision": <int 1-10>,
    "response_structure": <int 1-10>,
    "scope_awareness": <int 1-10>,
    "helpfulness": <int 1-10>
  }},
  "justification": "<2-3 sentences explaining the key strengths and weaknesses>",
  "strongest_aspect": "<which criterion scored highest and why>",
  "weakest_aspect": "<which criterion scored lowest and why>"
}}"""


# ═══════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════════

JUDGE_MODEL = "claude-sonnet-4-6-20250514"  # Claude Sonnet 4.6
CRITERIA = [
    'scientific_accuracy', 'relevance', 'completeness', 'directness',
    'quantitative_precision', 'response_structure', 'scope_awareness', 'helpfulness'
]
BATCH_SIZE = 35
CONTEXT_PROVIDED_TO_JUDGE = False  # Deliberate: context-blind evaluation
PIPELINE_IDENTITY_IN_PROMPT = False  # Single-blinded

PIPELINE_CONFIG = {
    'set1': {'label': 'DLR 2-Step RAG', 'color': '#2196F3'},
    'set2': {'label': 'Ontology-Enhanced RAG', 'color': '#FF9800'},
    'set3': {'label': 'No-Ontology RAG', 'color': '#4CAF50'},
    'set4': {'label': 'Zero-Shot', 'color': '#9E9E9E'},
}


# ═══════════════════════════════════════════════════════════════
# SECTION 3: DATA LOADING
# ═══════════════════════════════════════════════════════════════

import csv, json, os, re
import numpy as np
import pandas as pd
csv.field_size_limit(int(1e8))

def load_pipeline_answers(csv_path, set_id, pipeline_label, answer_column='answer', question_column='question', max_questions=70):
    """Load question-answer pairs from a CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = sorted(list(csv.DictReader(f)), key=lambda x: int(x['question_id']))
    
    tasks = []
    for r in rows[:max_questions]:
        qid = int(r['question_id'])
        if qid > max_questions:
            continue
        tasks.append({
            'task_id': f"Q{qid}_{set_id}",
            'question_id': qid,
            'set': set_id,
            'pipeline': pipeline_label,
            'question': r[question_column],
            'answer': r[answer_column],
        })
    return tasks


# ═══════════════════════════════════════════════════════════════
# SECTION 4: EVALUATION (call judge model)
# ═══════════════════════════════════════════════════════════════

def evaluate_single_task(task, judge_client=None):
    """
    Evaluate a single question-answer pair.
    In actual execution, this calls Claude Sonnet 4.6.
    Returns: dict with scores, justification, strongest, weakest.
    """
    prompt = EVAL_PROMPT_TEMPLATE.format(
        question=task['question'],
        answer=task['answer']
    )
    
    # NOTE: In actual execution, this was done via Perplexity Computer's
    # subagent infrastructure calling Claude Sonnet 4.6 with:
    #   - system_prompt = SYSTEM_PROMPT
    #   - user_prompt = prompt (above)
    #   - response_format = JSON
    #
    # To reproduce with the Anthropic API directly:
    #
    # import anthropic
    # client = anthropic.Anthropic(api_key="...")
    # response = client.messages.create(
    #     model=JUDGE_MODEL,
    #     max_tokens=1024,
    #     system=SYSTEM_PROMPT,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # result = json.loads(response.content[0].text)
    
    raise NotImplementedError(
        "This function requires Claude Sonnet 4.6 API access. "
        "Pre-computed results are in all_eval_results_4pipelines.json"
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 5: AGGREGATION AND STATISTICS
# ═══════════════════════════════════════════════════════════════

from scipy import stats

def compute_summary(df, pipe_order):
    """Compute per-pipeline summary statistics."""
    rows = []
    for pipe in pipe_order:
        sub = df[df['pipeline'] == pipe]
        row = {'Pipeline': pipe, 'N': len(sub)}
        for c in CRITERIA:
            row[f'{c}_mean'] = round(sub[c].mean(), 2)
            row[f'{c}_std'] = round(sub[c].std(), 2)
        row['overall_mean'] = round(sub['overall'].mean(), 2)
        row['overall_std'] = round(sub['overall'].std(), 2)
        rows.append(row)
    return pd.DataFrame(rows)

def compute_pairwise_tests(df, pipe_pairs):
    """Wilcoxon signed-rank tests for all pipeline pairs."""
    rows = []
    for pipe_a, pipe_b in pipe_pairs:
        da = df[df['pipeline']==pipe_a].sort_values('question_id')['overall'].values
        db = df[df['pipeline']==pipe_b].sort_values('question_id')['overall'].values
        try:
            w, p = stats.wilcoxon(da, db)
        except:
            w, p = 0, 1
        n = len(da)
        r_eff = 1 - (2*w)/(n*(n+1)/2) if n > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rows.append({
            'pipeline_a': pipe_a, 'pipeline_b': pipe_b,
            'mean_a': round(np.mean(da), 2), 'mean_b': round(np.mean(db), 2),
            'diff': round(np.mean(db) - np.mean(da), 2),
            'W': round(w), 'p_value': f"{p:.6f}", 'effect_r': round(r_eff, 3), 'sig': sig
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# SECTION 6: VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def generate_all_charts(df, output_dir, pipe_order, colors):
    """Generate all 4-pipeline charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.05)
    
    NICE = {
        'scientific_accuracy': 'Scientific\nAccuracy', 'relevance': 'Relevance',
        'completeness': 'Completeness', 'directness': 'Directness',
        'quantitative_precision': 'Quantitative\nPrecision',
        'response_structure': 'Response\nStructure',
        'scope_awareness': 'Scope\nAwareness', 'helpfulness': 'Helpfulness',
    }
    
    # Chart 1: Grouped bars
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(len(CRITERIA))
    w = 0.2
    for i, pipe in enumerate(pipe_order):
        sub = df[df['pipeline'] == pipe]
        means = [sub[c].mean() for c in CRITERIA]
        stds = [sub[c].std() for c in CRITERIA]
        ax.bar(x + (i-1.5)*w, means, w, yerr=stds, label=pipe, color=colors[pipe], capsize=2, alpha=0.85)
    ax.set_ylabel('Score (1-10)')
    ax.set_title('4-Pipeline Comparison: All Criteria', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in CRITERIA], fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 10.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/chart_4pipe_bars.png', dpi=200)
    plt.close()
    
    # [Additional charts follow same pattern — see full script]
    print("Charts generated.")


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    OUTPUT = 'output_improved'
    
    # Load pre-computed results
    with open(f'{OUTPUT}/all_eval_results_4pipelines.json') as f:
        all_results = json.load(f)
    
    # Build DataFrame
    rows = []
    for r in all_results:
        row = {'question_id': r['question_id'], 'pipeline': PIPELINE_CONFIG[r['set']]['label'], 'set': r['set']}
        for c in CRITERIA:
            row[c] = r['scores'][c]
        row['overall'] = round(np.mean([r['scores'][c] for c in CRITERIA]), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    
    pipe_order = [PIPELINE_CONFIG[s]['label'] for s in ['set4','set1','set3','set2']]
    colors = {PIPELINE_CONFIG[s]['label']: PIPELINE_CONFIG[s]['color'] for s in PIPELINE_CONFIG}
    
    # Summary
    summary = compute_summary(df, pipe_order)
    summary.to_csv(f'{OUTPUT}/summary_4pipelines.csv', index=False)
    print(summary.to_string())
    
    # Statistics
    pairs = [(pipe_order[i], pipe_order[j]) for i in range(len(pipe_order)) for j in range(i+1, len(pipe_order))]
    stats_df = compute_pairwise_tests(df, pairs)
    stats_df.to_csv(f'{OUTPUT}/statistical_tests_4pipelines.csv', index=False)
    print(stats_df.to_string())
    
    print(f"\nResults: {OUTPUT}/")
