import pandas as pd

# Load the experiments data
df = pd.read_csv('../deepeval_relevancy_scores_experiments.csv')

# Check the unique values in the 'experiment' column
print("Unique experiments:", df['experiment'].unique())

# Filter for slim-ontology
df_slim = df[df['experiment'] == 'slim-ontology']
print(f"\nSlim-ontology results: {len(df_slim)}")

# Check relevancy scores
print("Relevancy scores:")
print(df_slim['relevancy_score'].describe())

# Check for NaN values
print(f"\nNaN values: {df_slim['relevancy_score'].isna().sum()}")

# Check a sample of the data
print("\nSample data:")
print(df_slim[['question_id', 'experiment', 'relevancy_score']].head(10))
