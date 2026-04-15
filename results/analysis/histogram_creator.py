#!/usr/bin/env python3
"""
📊 ENHANCED VISUALIZATION CREATOR

Creates various types of visualizations showing score distributions across pipelines.

USAGE EXAMPLES:

# Default: Faceted histograms for all criteria and pipelines
python histogram_creator.py

# Specific criteria with faceted histograms
python histogram_creator.py --criteria hallucination
python histogram_creator.py --criteria overall_quality

# Specific pipelines with faceted histograms
python histogram_creator.py --pipelines 1_pass_with_ontology 1_pass_without_ontology

# Use alternative visualization types
python histogram_creator.py --plot-type boxplot
python histogram_creator.py --plot-type violin
python histogram_creator.py --plot-type ecdf

# Combined options
python histogram_creator.py --criteria hallucination --pipelines 1_pass_with_ontology 1_pass_without_ontology --plot-type violin

AVAILABLE CRITERIA:
- hallucination (from deepeval_results.csv)
- relevancy (from deepeval_results.csv)
- faithfulness (from deepeval_results.csv)
- contextual_relevancy (from deepeval_results.csv)
- overall_quality (from deepeval_results.csv)

AVAILABLE PIPELINES:
- dlr_1step (from deepeval_results.csv)
- dlr_2step (from deepeval_results.csv)
- 1_pass_with_ontology (from deepeval_results.csv)
- 1_pass_without_ontology (from deepeval_results.csv)

AVAILABLE PLOT TYPES:
- faceted (default): Faceted histograms - separate panels for each pipeline
- boxplot: Box plots with jittered points showing individual data
- violin: Violin plots with embedded box plots for density + quartiles
- ecdf: Empirical Cumulative Distribution Functions for statistical comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Create visualizations of pipeline score distributions')
parser.add_argument('--criteria', nargs='?', default='all', 
                   help='Criteria to plot (default: all)')
parser.add_argument('--pipelines', nargs='+', default='all',
                   help='Pipelines to include (default: all)')
parser.add_argument('--plot-type', choices=['faceted', 'boxplot', 'violin', 'ecdf'], 
                   default='faceted', help='Type of plot to create (default: faceted)')
parser.add_argument('--save', action='store_true', 
                   help='Save plots to files instead of displaying them')
args = parser.parse_args()

# Set matplotlib backend if saving
if args.save:
    import matplotlib
    matplotlib.use('Agg')

# Define available criteria and their source files
CRITERIA_SOURCES = {
    'hallucination': 'deepeval_results.csv',
    'relevancy': 'deepeval_results.csv',
    'faithfulness': 'deepeval_results.csv',
    'contextual_relevancy': 'deepeval_results.csv',
    'overall_quality': 'deepeval_results.csv',
    'overall_quality_recalculated': 'deepeval_results.csv',
    # Add DLR criteria from compiled_results_w_exprmts_gpt4.1m.csv
    'factuality': 'compiled_results_w_exprmts_gpt4.1m.csv',
    'groundedness': 'compiled_results_w_exprmts_gpt4.1m.csv',
    'helpfulness': 'compiled_results_w_exprmts_gpt4.1m.csv',
    'depth': 'compiled_results_w_exprmts_gpt4.1m.csv'
}

# Define all available pipelines
ALL_PIPELINES = ['dlr_1step', 'dlr_2step', '1_pass_with_ontology', '1_pass_without_ontology', 'experiment_1', 'experiment_2']

# Determine which criteria to plot
if args.criteria == 'all':
    criteria_list = list(CRITERIA_SOURCES.keys())
else:
    criteria_list = [args.criteria] if args.criteria in CRITERIA_SOURCES else [args.criteria]

# Determine which pipelines to include
if args.pipelines == ['all'] or 'all' in args.pipelines:
    pipelines_to_plot = ALL_PIPELINES
else:
    pipelines_to_plot = [p for p in args.pipelines if p in ALL_PIPELINES]

print(f"📊 Creating histograms for criteria: {', '.join(criteria_list)}")
print(f"📊 Including pipelines: {', '.join(pipelines_to_plot)}")
print()

# Load data from source files
print("Loading data...")
deepeval_data = pd.read_csv('deepeval_results.csv')
deepeval_n1_data = pd.read_csv('deepeval_resultsN1.csv')
dlr_data = pd.read_csv('compiled_results_w_exprmts_gpt4.1m.csv')

# Combine all data into one DataFrame
# Use the main deepeval_results.csv as it contains all the data we need
combined_df = deepeval_data[deepeval_data['pipeline'].isin(pipelines_to_plot)].copy()

# Add DLR criteria from the compiled_results file
dlr_criteria = ['factuality', 'groundedness', 'helpfulness', 'depth']
for criteria in dlr_criteria:
    if criteria in dlr_data.columns:
        # Get data for pipelines that exist in both files
        for pipeline in pipelines_to_plot:
            if pipeline in dlr_data['pipeline'].unique():
                pipeline_dlr_data = dlr_data[dlr_data['pipeline'] == pipeline]
                if not pipeline_dlr_data.empty:
                    # Merge the DLR criteria into our combined dataframe
                    for _, row in pipeline_dlr_data.iterrows():
                        mask = (combined_df['pipeline'] == pipeline) & (combined_df['question_id'] == row['question_id'])
                        if mask.any():
                            combined_df.loc[mask, criteria] = row[criteria]

# Recalculate overall_quality using inverse hallucination
# Overall quality is a weighted combination of other metrics, with hallucination inverted
combined_df['overall_quality_recalculated'] = (
    0.4 * (1 - combined_df['hallucination']) +  # Inverse hallucination
    0.2 * combined_df['relevancy'] +
    0.2 * combined_df['faithfulness'] +
    0.2 * combined_df['contextual_relevancy']
)

print(f"Loaded {len(combined_df)} total evaluations")
print()

def create_faceted_histograms(combined_df, criteria, pipelines_to_plot):
    """Create faceted histograms with separate panels for each pipeline"""
    print(f"Creating faceted histograms for {criteria}...")
    
    # Create a grid of histograms
    n_pipelines = len(pipelines_to_plot)
    n_cols = 2
    n_rows = (n_pipelines + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    if n_pipelines == 1:
        axes = np.array([[axes]])
    
    axes = axes.flatten()
    
    for i, pipeline in enumerate(pipelines_to_plot):
        pipeline_data = combined_df[combined_df['pipeline'] == pipeline]
        if not pipeline_data.empty and criteria in pipeline_data.columns:
            ax = axes[i]
            ax.hist(pipeline_data[criteria], bins=20, color='skyblue', 
                   edgecolor='black', alpha=0.8)
            ax.set_title(f'{pipeline}', fontsize=12, fontweight='bold')
            ax.set_xlabel(criteria.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = pipeline_data[criteria].mean()
            ax.axvline(mean_val, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.legend(fontsize=9)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'Faceted Histograms: {criteria.replace("_", " ").title()} Scores by Pipeline', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save or show
    if args.save:
        filename = f'{criteria}_faceted.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved to {filename}")
    else:
        plt.show()

def create_boxplots(combined_df, criteria, pipelines_to_plot):
    """Create box plots with jittered points"""
    print(f"Creating box plots for {criteria}...")
    
    plt.figure(figsize=(12, 6))
    
    # Create boxplot
    sns.boxplot(data=combined_df, x='pipeline', y=criteria, 
               hue='pipeline', palette='viridis', order=pipelines_to_plot, legend=False)
    
    # Add jittered points
    sns.stripplot(data=combined_df, x='pipeline', y=criteria, 
                 color='black', alpha=0.3, jitter=True, size=3, 
                 order=pipelines_to_plot)
    
    plt.title(f'Box Plot with Jitter: {criteria.replace("_", " ").title()} Scores by Pipeline', 
             fontsize=14)
    plt.xlabel('Pipeline', fontsize=12)
    plt.ylabel(criteria.replace("_", " ").title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or show
    if args.save:
        filename = f'{criteria}_boxplot.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved to {filename}")
    else:
        plt.show()

def create_violin_plots(combined_df, criteria, pipelines_to_plot):
    """Create violin plots with embedded box plots"""
    print(f"Creating violin plots for {criteria}...")
    
    plt.figure(figsize=(12, 6))
    
    # Create violin plot
    sns.violinplot(data=combined_df, x='pipeline', y=criteria, 
                  hue='pipeline', palette='viridis', inner=None, order=pipelines_to_plot, legend=False)
    
    # Add box plot overlay
    sns.boxplot(data=combined_df, x='pipeline', y=criteria, 
               width=0.15, color='white', showcaps=False,
               boxprops={'facecolor': 'white', 'edgecolor': 'black'},
               whiskerprops={'color': 'black'}, order=pipelines_to_plot)
    
    plt.title(f'Violin Plot: {criteria.replace("_", " ").title()} Score Distribution by Pipeline', 
             fontsize=14)
    plt.xlabel('Pipeline', fontsize=12)
    plt.ylabel(criteria.replace("_", " ").title(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or show
    if args.save:
        filename = f'{criteria}_violin.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved to {filename}")
    else:
        plt.show()

def create_ecdf_plots(combined_df, criteria, pipelines_to_plot):
    """Create Empirical Cumulative Distribution Function plots"""
    print(f"Creating ECDF plots for {criteria}...")
    
    plt.figure(figsize=(12, 6))
    
    # Color palette
    colors = sns.color_palette('viridis', len(pipelines_to_plot))
    
    for i, pipeline in enumerate(pipelines_to_plot):
        pipeline_data = combined_df[combined_df['pipeline'] == pipeline]
        if not pipeline_data.empty and criteria in pipeline_data.columns:
            sorted_data = np.sort(pipeline_data[criteria])
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            plt.plot(sorted_data, y, marker='o', markersize=4,
                    label=pipeline, color=colors[i], alpha=0.8)
    
    plt.title(f'ECDF: {criteria.replace("_", " ").title()} Scores by Pipeline', 
             fontsize=14)
    plt.xlabel(criteria.replace("_", " ").title(), fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.legend(title='Pipeline', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or show
    if args.save:
        filename = f'{criteria}_ecdf.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved to {filename}")
    else:
        plt.show()

# Check if we have data to plot
if len(combined_df) == 0:
    print("❌ No pipelines selected or no data available")
else:
    # Create visualizations for each criteria
    for criteria in criteria_list:
        if criteria in CRITERIA_SOURCES:
            print(f"\n📊 Processing {criteria}...")
            
            # Choose visualization type based on argument
            if args.plot_type == 'faceted':
                # Use recalculated overall quality if available
                if criteria == 'overall_quality' and 'overall_quality_recalculated' in combined_df.columns:
                    create_faceted_histograms(combined_df, 'overall_quality_recalculated', pipelines_to_plot)
                else:
                    create_faceted_histograms(combined_df, criteria, pipelines_to_plot)
            elif args.plot_type == 'boxplot':
                # Use recalculated overall quality if available
                if criteria == 'overall_quality' and 'overall_quality_recalculated' in combined_df.columns:
                    create_boxplots(combined_df, 'overall_quality_recalculated', pipelines_to_plot)
                else:
                    create_boxplots(combined_df, criteria, pipelines_to_plot)
            elif args.plot_type == 'violin':
                # Use recalculated overall quality if available
                if criteria == 'overall_quality' and 'overall_quality_recalculated' in combined_df.columns:
                    create_violin_plots(combined_df, 'overall_quality_recalculated', pipelines_to_plot)
                else:
                    create_violin_plots(combined_df, criteria, pipelines_to_plot)
            elif args.plot_type == 'ecdf':
                # Use recalculated overall quality if available
                if criteria == 'overall_quality' and 'overall_quality_recalculated' in combined_df.columns:
                    create_ecdf_plots(combined_df, 'overall_quality_recalculated', pipelines_to_plot)
                else:
                    create_ecdf_plots(combined_df, criteria, pipelines_to_plot)
            
            # Print statistics
            print(f"\n📊 Statistics for {criteria}:")
            for pipeline in pipelines_to_plot:
                pipeline_data = combined_df[combined_df['pipeline'] == pipeline]
                if not pipeline_data.empty:
                    # Use recalculated overall quality if available
                    if criteria == 'overall_quality' and 'overall_quality_recalculated' in pipeline_data.columns:
                        criteria_to_use = 'overall_quality_recalculated'
                    else:
                        criteria_to_use = criteria
                    
                    if criteria_to_use in pipeline_data.columns:
                        mean_val = pipeline_data[criteria_to_use].mean()
                        std_val = pipeline_data[criteria_to_use].std()
                        min_val = pipeline_data[criteria_to_use].min()
                        max_val = pipeline_data[criteria_to_use].max()
                        median_val = pipeline_data[criteria_to_use].median()
                        print(f"  {pipeline}:")
                        print(f"    Mean: {mean_val:.3f}")
                        print(f"    Median: {median_val:.3f}")
                        print(f"    Std: {std_val:.3f}")
                        print(f"    Range: [{min_val:.3f}, {max_val:.3f}]")
            print()
        else:
            print(f"⚠️  Criteria '{criteria}' not found in available data")

print("✅ Histogram creation complete!")
