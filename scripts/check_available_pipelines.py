import pandas as pd

# Load the CSV file
df = pd.read_csv('dlr_data/DARES25_EarthObsertvation_QA_RAG_results_v1.csv')

# Get unique pipelines
unique_pipelines = df['pipeline'].unique()

print(f"Available pipelines: {unique_pipelines}")

# Count rows per pipeline
for pipeline in unique_pipelines:
    pipeline_df = df[df['pipeline'] == pipeline]
    print(f"{pipeline}: {len(pipeline_df)} rows")
