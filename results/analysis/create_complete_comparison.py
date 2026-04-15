#!/usr/bin/env python3
"""
📊 CREATE COMPLETE COMPARISON WITH ALL EXPERIMENTS

Creates a comprehensive comparison that includes:
- zero-shot (dlr_1step)
- dlr_2step
- scores-only experiment
- slim-ontology experiment

Generates comparison graphs showing all 4 pipelines with proper data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Input files
DEEPEVAL_RESULTS_FILE = '../deepeval_evaluations/deepeval_results_with_relevancy.csv'
RELEVANCY_SCORES_FILE = '../deepeval_relevancy_scores_experiments.csv'

# Output directory
OUTPUT_DIR = 'graphs/complete_comparison'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("📊 Creating complete comparison with all experiments...")
print(f"Deepeval results: {DEEPEVAL_RESULTS_FILE}")
print(f"Relevancy scores: {RELEVANCY_SCORES_FILE}")
print(f"Output: {OUTPUT_DIR}/")
print()

try:
    # Load data
    deepeval_df = pd.read_csv(DEEPEVAL_RESULTS_FILE)
    relevancy_df = pd.read_csv(RELEVANCY_SCORES_FILE)
    
    print(f"✅ Loaded {len(deepeval_df)} deepeval results")
    print(f"✅ Loaded {len(relevancy_df)} relevancy scores")
    
    # Create a combined dataframe
    # Start with the original deepeval results
    combined_df = deepeval_df.copy()
    
    # Add a column to track which experiment each row belongs to
    combined_df['experiment_type'] = 'original'
    
    # For 1_pass_with_ontology rows, add experiment distinction
    for idx, row in combined_df[combined_df['pipeline'] == '1_pass_with_ontology'].iterrows():
        if row['relevancy_experiment'] == 'scores-only':
            combined_df.at[idx, 'experiment_type'] = 'scores-only'
        elif row['relevancy_experiment'] == 'slim-ontology':
            combined_df.at[idx, 'experiment_type'] = 'slim-ontology'
    
    # Create display names
    combined_df['pipeline_display'] = combined_df['pipeline'].map({
        'zero_shot': 'Zero-Shot (DLR 1-Step)',
        'dlr_2step': 'DLR 2-Step',
        '1_pass_without_ontology': '1-Pass Without Ontology',
        '1_pass_with_ontology': '1-Pass With Ontology'
    })
    
    # Add experiment suffix for 1_pass_with_ontology
    combined_df.loc[combined_df['pipeline'] == '1_pass_with_ontology', 'pipeline_display'] = \
        combined_df.loc[combined_df['pipeline'] == '1_pass_with_ontology', 'experiment_type'] + ' (1-Pass With Ontology)'
    
    print(f"\nPipelines in combined comparison:")
    print(combined_df['pipeline_display'].value_counts())
    
    # Create comparison graphs
    plt.figure(figsize=(16, 10))
    
    # Plot each pipeline
    pipelines = sorted(combined_df['pipeline_display'].unique())
    
    for i, pipeline in enumerate(pipelines):
        pipeline_data = combined_df[combined_df['pipeline_display'] == pipeline]
        
        # Determine which score column to use
        if 'relevancy_score' in pipeline_data.columns and pipeline_data['relevancy_score'].notna().any():
            scores = pipeline_data['relevancy_score'].dropna()
            source = 'New Evaluation'
        elif 'relevancy' in pipeline_data.columns and pipeline_data['relevancy'].notna().any():
            scores = pipeline_data['relevancy'].dropna()
            source = 'Original Data'
        else:
            continue
        
        # Create subplot
        plt.subplot(2, 2, i+1)
        sns.histplot(scores, bins=20, kde=True, color='skyblue', alpha=0.8)
        
        # Add mean line
        mean_score = scores.mean()
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_score:.3f}')
        
        plt.title(pipeline, fontsize=12, pad=10)
        plt.xlabel('Relevancy Score', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Add stats
        stats_text = f'N={len(scores)}\n'
        stats_text += f'Mean={mean_score:.3f}\n'
        stats_text += f'Median={scores.median():.3f}\n'
        stats_text += f'Source: {source}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Complete Relevancy Score Comparison - All 4 Pipelines', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the complete comparison
    complete_filename = os.path.join(OUTPUT_DIR, 'complete_relevancy_comparison.png')
    plt.savefig(complete_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Complete comparison saved to: {complete_filename}")
    
    # Print summary statistics
    print(f"\n📊 Complete Comparison Statistics:")
    for pipeline in pipelines:
        pipeline_data = combined_df[combined_df['pipeline_display'] == pipeline]
        
        # Get scores
        if 'relevancy_score' in pipeline_data.columns and pipeline_data['relevancy_score'].notna().any():
            scores = pipeline_data['relevancy_score'].dropna()
        elif 'relevancy' in pipeline_data.columns and pipeline_data['relevancy'].notna().any():
            scores = pipeline_data['relevancy'].dropna()
        else:
            continue
        
        print(f"  {pipeline}:")
        print(f"    N = {len(scores)}")
        print(f"    Mean = {scores.mean():.3f}")
        print(f"    Median = {scores.median():.3f}")
        print(f"    Std = {scores.std():.3f}")
        print(f"    Min = {scores.min():.3f}")
        print(f"    Max = {scores.max():.3f}")
    
    print(f"\n✅ Complete comparison created successfully!")
    print(f"📁 Saved in: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()