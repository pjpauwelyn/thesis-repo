import pandas as pd

# Load the fixed CSV
df = pd.read_csv('../../results_to_be_processed/refined-context_results_fixed.csv')

# Check the context for the regenerated questions
print("Context for regenerated questions:")
for idx, row in df.iterrows():
    question_id = row.get('question_id', idx + 1)
    context = row.get('context', '')
    
    # Check for placeholders
    has_placeholders = '{' in str(context) and '}' in str(context)
    
    print(f"\nQuestion {question_id}:")
    print(f"  Has placeholders: {has_placeholders}")
    if has_placeholders:
        print(f"  Context preview: {context[:200]}...")
