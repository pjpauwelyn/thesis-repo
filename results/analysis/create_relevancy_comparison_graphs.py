#!/usr/bin/env python3
"""
📊 CREATE RELEVANCY COMPARISON GRAPHS

Creates comparison graphs showing relevancy scores for the 4 pipelines:
- zero-shot (dlr_1step)
- dlr_2step
- experiment 1 (scores-only)
- experiment 2 (slim-ontology)

Generates 4 separate PNG graphs and one combined comparison graph.
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

# Input file
INPUT_FILE = '../deepeval_evaluations/deepeval_results_with_relevancy.csv'

# Output directory
OUTPUT_DIR = 'graphs/relevancy_comparison'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("📊 Creating relevancy comparison graphs...")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}/")
print()

try:
    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} results")
    
    # Map pipeline names to display names
    pipeline_names = {
        'zero_shot': 'Zero-Shot (DLR 1-Step)',
        'dlr_2step': 'DLR 2-Step',
        '1_pass_without_ontology': '1-Pass Without Ontology',
        '1_pass_with_ontology': '1-Pass With Ontology (Experiments)'
    }
    
    # Add display names
    df['pipeline_display'] = df['pipeline'].map(pipeline_names)
    
    # For the 1_pass_with_ontology pipeline, use the experiment type for distinction
    # This creates separate display names for scores-only and slim-ontology experiments
    experiment_display = df.loc[df['pipeline'] == '1_pass_with_ontology', 'relevancy_experiment'].map({
        'scores-only': '1-Pass With Ontology (Scores-Only)',
        'slim-ontology': '1-Pass With Ontology (Slim-Ontology)'
    })
    df.loc[df['pipeline'] == '1_pass_with_ontology', 'pipeline_display'] = experiment_display
    
    print(f"Pipelines found: {df['pipeline_display'].unique()}")
    print()
    
    # Create individual pipeline graphs
    for pipeline in df['pipeline_display'].unique():
        pipeline_df = df[df['pipeline_display'] == pipeline]
        
        if len(pipeline_df) > 0:
            print(f"Creating graph for {pipeline}...")
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot score distribution
            if 'relevancy_score' in pipeline_df.columns:
                scores = pipeline_df['relevancy_score'].dropna()
                if len(scores) > 0:
                    sns.histplot(scores, bins=20, kde=True, color='skyblue', alpha=0.8)
                    
                    # Add mean line
                    mean_score = scores.mean()
                    plt.axvline(mean_score, color='red', linestyle='--', 
                               linewidth=2, label=f'Mean: {mean_score:.3f}')
                    
                    plt.title(f'{pipeline} - Relevancy Score Distribution', fontsize=16, pad=20)
                    plt.xlabel('Relevancy Score', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    
                    # Add statistics
                    stats_text = f'Count: {len(scores)}\n'
                    stats_text += f'Mean: {mean_score:.3f}\n'
                    stats_text += f'Median: {scores.median():.3f}\n'
                    stats_text += f'Std: {scores.std():.3f}\n'
                    stats_text += f'Min: {scores.min():.3f}\n'
                    stats_text += f'Max: {scores.max():.3f}'
                    
                    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    plt.text(0.5, 0.5, 'No relevancy scores available',
                            ha='center', va='center', fontsize=14)
                    plt.title(f'{pipeline} - No Relevancy Data', fontsize=16)
            else:
                plt.text(0.5, 0.5, 'No relevancy scores available',
                        ha='center', va='center', fontsize=14)
                plt.title(f'{pipeline} - No Relevancy Data', fontsize=16)
            
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(OUTPUT_DIR, f'{pipeline.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")}_relevancy.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved to {filename}")
        else:
            print(f"  ⚠️  No data for {pipeline}")
    
    # Create combined comparison graph with histograms
    print(f"\nCreating combined comparison graph with histograms...")
    
    # Create a figure with subplots for each pipeline
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    pipelines = sorted(df['pipeline_display'].unique())
    
    for i, pipeline in enumerate(pipelines):
        ax = axes[i]
        pipeline_df = df[df['pipeline_display'] == pipeline]
        
        # Plot histogram for this pipeline
        # Try new relevancy_score first, then fall back to original relevancy column
        scores = None
        score_source = None
        
        if 'relevancy_score' in pipeline_df.columns:
            scores = pipeline_df['relevancy_score'].dropna()
            score_source = 'new'
        
        if (scores is None or len(scores) == 0) and 'relevancy' in pipeline_df.columns:
            scores = pipeline_df['relevancy'].dropna()
            score_source = 'original'
        
        if scores is not None and len(scores) > 0:
            sns.histplot(scores, bins=20, kde=True, ax=ax, color='skyblue', alpha=0.8)
            
            # Add mean line
            mean_score = scores.mean()
            source_text = " (New)" if score_source == 'new' else " (Original)"
            ax.axvline(mean_score, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean_score:.3f}{source_text}')
            
            ax.set_title(f'{pipeline}', fontsize=14, pad=10)
            ax.set_xlabel('Relevancy Score', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add stats
            stats_text = f'N={len(scores)}\n'
            stats_text += f'Mean={mean_score:.3f}\n'
            stats_text += f'Median={scores.median():.3f}'
            if score_source == 'new':
                stats_text += f'\nSource: New Evaluation'
            else:
                stats_text += f'\nSource: Original Data'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No relevancy scores',
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{pipeline} - No Data', fontsize=14)
        
        ax.set_xlim(0, 1)  # Consistent x-axis for all subplots
    
    plt.suptitle('Relevancy Score Distribution Comparison Across All Pipelines', 
                fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Save combined figure
    combined_filename = os.path.join(OUTPUT_DIR, 'all_pipelines_relevancy_comparison.png')
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved combined comparison to {combined_filename}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    for pipeline in sorted(df['pipeline_display'].unique()):
        pipeline_df = df[df['pipeline_display'] == pipeline]
        scores = pipeline_df['relevancy_score'].dropna()
        
        if len(scores) > 0:
            print(f"  {pipeline}:")
            print(f"    N = {len(scores)}")
            print(f"    Mean = {scores.mean():.3f}")
            print(f"    Median = {scores.median():.3f}")
            print(f"    Std = {scores.std():.3f}")
            print(f"    Min = {scores.min():.3f}")
            print(f"    Max = {scores.max():.3f}")
        else:
            print(f"  {pipeline}: No relevancy scores")
    
    print(f"\n✅ All graphs created successfully!")
    print(f"📁 Saved in: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()