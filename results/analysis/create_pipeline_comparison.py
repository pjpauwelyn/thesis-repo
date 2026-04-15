#!/usr/bin/env python3
"""
📊 CREATE PIPELINE COMPARISON WITH EXPERIMENT RESULTS

Creates a new pipeline comparison CSV file that includes the latest experiment results
from scores_only_evaluation_results.csv and slim_ontology_evaluation_results.csv.
Replaces the experiment stats in the existing comparison file.
"""

import pandas as pd
import os

# Input files
SCORES_ONLY_FILE = '../scores_only_evaluation_results.csv'
SLIM_ONTOLOGY_FILE = '../slim_ontology_evaluation_results.csv'
ORIGINAL_COMPARISON_FILE = 'pipeline_comparison_(w_failed_experiments)_GPT4.1m.csv'

# Output file
OUTPUT_FILE = 'pipeline_comparison_(w_experiments)_GPT4.1m.csv'

print("📊 Creating pipeline comparison with latest experiment results...")
print()

# Load the original comparison data
original_comparison = pd.read_csv(ORIGINAL_COMPARISON_FILE, comment='#')
print(f"Loaded original comparison: {len(original_comparison)} pipelines")
print(original_comparison)
print()

# Load experiment data
print("Loading experiment data...")
scores_only_data = pd.read_csv(SCORES_ONLY_FILE)
slim_ontology_data = pd.read_csv(SLIM_ONTOLOGY_FILE)

print(f"Scores-only data: {len(scores_only_data)} rows")
print(f"Slim-ontology data: {len(slim_ontology_data)} rows")
print()

# Calculate statistics for each experiment
def calculate_stats(data, pipeline_name):
    """Calculate mean statistics for a pipeline"""
    if len(data) == 0:
        return None
    
    stats = {
        'Pipeline': pipeline_name,
        'Overall Score': round(data['overall_score'].mean(), 3),
        'Factuality': round(data['factuality'].mean(), 3),
        'Relevance': round(data['relevance'].mean(), 3),
        'Groundedness': round(data['groundedness'].mean(), 3),
        'Helpfulness': round(data['helpfulness'].mean(), 3),
        'Depth': round(data['depth'].mean(), 3),
        'Question Count': len(data)
    }
    return stats

# Calculate stats for experiments
scores_only_stats = calculate_stats(scores_only_data, 'experiment_1')
slim_ontology_stats = calculate_stats(slim_ontology_data, 'experiment_2')

print("Experiment statistics:")
print(f"Scores-only (experiment_1): {scores_only_stats}")
print(f"Slim-ontology (experiment_2): {slim_ontology_stats}")
print()

# Replace experiment rows in the original comparison
updated_comparison = original_comparison.copy()

# Find and replace experiment_1 row
if scores_only_stats:
    experiment_1_index = updated_comparison[updated_comparison['Pipeline'] == 'experiment_1'].index
    if len(experiment_1_index) > 0:
        updated_comparison.loc[experiment_1_index[0]] = scores_only_stats
        print(f"✅ Updated experiment_1 row with scores-only results")
    else:
        updated_comparison = pd.concat([updated_comparison, pd.DataFrame([scores_only_stats])], ignore_index=True)
        print(f"✅ Added experiment_1 row with scores-only results")

# Find and replace experiment_2 row  
if slim_ontology_stats:
    experiment_2_index = updated_comparison[updated_comparison['Pipeline'] == 'experiment_2'].index
    if len(experiment_2_index) > 0:
        updated_comparison.loc[experiment_2_index[0]] = slim_ontology_stats
        print(f"✅ Updated experiment_2 row with slim-ontology results")
    else:
        updated_comparison = pd.concat([updated_comparison, pd.DataFrame([slim_ontology_stats])], ignore_index=True)
        print(f"✅ Added experiment_2 row with slim-ontology results")

print()
print("Updated comparison:")
print(updated_comparison)
print()

# Add header comments
header_comments = [
    "# Pipeline Performance Comparison",
    "# Comprehensive analysis of all 6 pipelines evaluated using GPT-4.1-mini",
    "# Generated: 2025-03-15 (updated with latest experiment results)",
    "# Updated experiment results from scores_only_evaluation_results.csv and slim_ontology_evaluation_results.csv",
    ""
]

# Save the updated comparison
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(header_comments))
    updated_comparison.to_csv(f, index=False)

print(f"✅ Pipeline comparison with updated experiment results saved to: {OUTPUT_FILE}")
print()
print("Summary of changes:")
print(f"  - Updated experiment_1 with scores-only results")
print(f"  - Updated experiment_2 with slim-ontology results")
print(f"  - Maintained all other pipeline statistics")
print(f"  - Added generation timestamp and update notes")