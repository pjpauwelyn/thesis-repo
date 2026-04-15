#!/usr/bin/env python3
"""
📊 EXPERIMENT COMPARISON GRAPHS

Creates comparison histograms showing score distributions between scores_only and slim_ontology experiments.
Saves 5 images (one per criteria) with two histograms per image side by side.
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

# Define experiments to compare
EXPERIMENT_1 = 'scores_only'
EXPERIMENT_2 = 'slim_ontology'

# Input files
SCORES_ONLY_FILE = '../scores_only_evaluation_results.csv'
SLIM_ONTOLOGY_FILE = '../slim_ontology_evaluation_results.csv'

# Output directory
OUTPUT_DIR = 'graphs/experiment_graphs'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📊 Creating experiment comparison histograms...")
print(f"Comparing: {EXPERIMENT_1} vs {EXPERIMENT_2}")
print(f"Criteria: {', '.join(CRITERIA)}")
print()

# Load data
print("Loading data...")
scores_only_data = pd.read_csv(SCORES_ONLY_FILE)
slim_ontology_data = pd.read_csv(SLIM_ONTOLOGY_FILE)

print(f"Scores-only data: {len(scores_only_data)} rows")
print(f"Slim-ontology data: {len(slim_ontology_data)} rows")
print()

def create_comparison_histogram(criteria, experiment1_data, experiment2_data, experiment1_name, experiment2_name):
    """Create a comparison histogram with two histograms side by side"""
    print(f"Creating comparison histogram for {criteria}...")
    
    # Check if criteria exists in both datasets
    if criteria not in experiment1_data.columns or criteria not in experiment2_data.columns:
        print(f"  ⚠️  Criteria '{criteria}' not found in one or both datasets")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Plot experiment 1 histogram
    ax1 = axes[0]
    experiment1_scores = experiment1_data[criteria].dropna()
    ax1.hist(experiment1_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
    ax1.set_title(f'{experiment1_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel(criteria.replace("_", " ").title(), fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add mean line for experiment 1
    mean_val1 = experiment1_scores.mean()
    ax1.axvline(mean_val1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val1:.3f}')
    ax1.legend(fontsize=10)
    
    # Plot experiment 2 histogram
    ax2 = axes[1]
    experiment2_scores = experiment2_data[criteria].dropna()
    ax2.hist(experiment2_scores, bins=20, color='lightcoral', edgecolor='black', alpha=0.8)
    ax2.set_title(f'{experiment2_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel(criteria.replace("_", " ").title(), fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add mean line for experiment 2
    mean_val2 = experiment2_scores.mean()
    ax2.axvline(mean_val2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val2:.3f}')
    ax2.legend(fontsize=10)
    
    plt.suptitle(f'Comparison: {criteria.replace("_", " ").title()} Scores - {experiment1_name} vs {experiment2_name}',
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(OUTPUT_DIR, f'{criteria}_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved to {filename}")
    
    # Print statistics
    print(f"  📊 Statistics:")
    print(f"    {experiment1_name}:")
    print(f"      Count: {len(experiment1_scores)}")
    print(f"      Mean: {mean_val1:.3f}")
    print(f"      Median: {experiment1_scores.median():.3f}")
    print(f"      Std: {experiment1_scores.std():.3f}")
    print(f"      Min: {experiment1_scores.min():.3f}")
    print(f"      Max: {experiment1_scores.max():.3f}")
    print(f"    {experiment2_name}:")
    print(f"      Count: {len(experiment2_scores)}")
    print(f"      Mean: {mean_val2:.3f}")
    print(f"      Median: {experiment2_scores.median():.3f}")
    print(f"      Std: {experiment2_scores.std():.3f}")
    print(f"      Min: {experiment2_scores.min():.3f}")
    print(f"      Max: {experiment2_scores.max():.3f}")
    print()

# Create comparison histograms for each criteria
for criteria in CRITERIA:
    create_comparison_histogram(criteria, scores_only_data, slim_ontology_data, EXPERIMENT_1, EXPERIMENT_2)

print("✅ All comparison histograms created successfully!")
print(f"📁 Saved in: {OUTPUT_DIR}")