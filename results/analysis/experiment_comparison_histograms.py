#!/usr/bin/env python3
"""
📊 EXPERIMENT COMPARISON HISTOGRAMS

Creates comparison histograms showing score distributions between dlr_2step and experiment_2 pipelines.
Saves 5 images (one per criteria) with two histograms per image.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib backend for saving
import matplotlib
matplotlib.use('Agg')

# Define criteria to plot
CRITERIA = ['factuality', 'groundedness', 'helpfulness', 'depth', 'overall_score']

# Define pipelines to compare
PIPELINE_1 = 'dlr_2step'
PIPELINE_2 = 'experiment_2'

# Input files
DLR_FILE = 'dlr_results_gpt4.1m.csv'
EXPERIMENT_FILE = 'compiled_results_w_exprmts_gpt4.1m.csv'

# Output directory
OUTPUT_DIR = 'graphs/experiment_graphs'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📊 Creating experiment comparison histograms...")
print(f"Comparing: {PIPELINE_1} vs {PIPELINE_2}")
print(f"Criteria: {', '.join(CRITERIA)}")
print()

# Load data
print("Loading data...")
dlr_data = pd.read_csv(DLR_FILE)
experiment_data = pd.read_csv(EXPERIMENT_FILE)

print(f"DLR data: {len(dlr_data)} rows")
print(f"Experiment data: {len(experiment_data)} rows")
print()

# Filter data for the specific pipelines
pipeline1_data = dlr_data[dlr_data['pipeline'] == PIPELINE_1]
pipeline2_data = experiment_data[experiment_data['pipeline'] == PIPELINE_2]

print(f"{PIPELINE_1} data: {len(pipeline1_data)} rows")
print(f"{PIPELINE_2} data: {len(pipeline2_data)} rows")
print()

def create_comparison_histogram(criteria, pipeline1_data, pipeline2_data, pipeline1_name, pipeline2_name):
    """Create a comparison histogram with two histograms side by side"""
    print(f"Creating comparison histogram for {criteria}...")
    
    # Check if criteria exists in both datasets
    if criteria not in pipeline1_data.columns or criteria not in pipeline2_data.columns:
        print(f"  ⚠️  Criteria '{criteria}' not found in one or both datasets")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Plot pipeline 1 histogram
    ax1 = axes[0]
    pipeline1_scores = pipeline1_data[criteria].dropna()
    ax1.hist(pipeline1_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
    ax1.set_title(f'{pipeline1_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel(criteria.replace("_", " ").title(), fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add mean line for pipeline 1
    mean_val1 = pipeline1_scores.mean()
    ax1.axvline(mean_val1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val1:.3f}')
    ax1.legend(fontsize=10)
    
    # Plot pipeline 2 histogram
    ax2 = axes[1]
    pipeline2_scores = pipeline2_data[criteria].dropna()
    ax2.hist(pipeline2_scores, bins=20, color='lightcoral', edgecolor='black', alpha=0.8)
    ax2.set_title(f'{pipeline2_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel(criteria.replace("_", " ").title(), fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add mean line for pipeline 2
    mean_val2 = pipeline2_scores.mean()
    ax2.axvline(mean_val2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val2:.3f}')
    ax2.legend(fontsize=10)
    
    plt.suptitle(f'Comparison: {criteria.replace("_", " ").title()} Scores - {pipeline1_name} vs {pipeline2_name}',
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(OUTPUT_DIR, f'{criteria}_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved to {filename}")
    
    # Print statistics
    print(f"  📊 Statistics:")
    print(f"    {pipeline1_name}:")
    print(f"      Count: {len(pipeline1_scores)}")
    print(f"      Mean: {mean_val1:.3f}")
    print(f"      Median: {pipeline1_scores.median():.3f}")
    print(f"      Std: {pipeline1_scores.std():.3f}")
    print(f"      Min: {pipeline1_scores.min():.3f}")
    print(f"      Max: {pipeline1_scores.max():.3f}")
    print(f"    {pipeline2_name}:")
    print(f"      Count: {len(pipeline2_scores)}")
    print(f"      Mean: {mean_val2:.3f}")
    print(f"      Median: {pipeline2_scores.median():.3f}")
    print(f"      Std: {pipeline2_scores.std():.3f}")
    print(f"      Min: {pipeline2_scores.min():.3f}")
    print(f"      Max: {pipeline2_scores.max():.3f}")
    print()

# Create comparison histograms for each criteria
for criteria in CRITERIA:
    create_comparison_histogram(criteria, pipeline1_data, pipeline2_data, PIPELINE_1, PIPELINE_2)

print("✅ All comparison histograms created successfully!")
print(f"📁 Saved in: {OUTPUT_DIR}")
