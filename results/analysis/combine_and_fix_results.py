#!/usr/bin/env python3
"""
📊 COMBINE AND FIX RESULTS

Combines results from refined-context_results.csv and refined-context_results_copy.csv,
replacing only the questions with incorrect context from refined-context_results_fixed.csv.
Ensures the final output maintains the same structure as the original files.
"""

import pandas as pd
import os
from pathlib import Path

# Input files
PRIMARY_CSV = '../../results_to_be_processed/refined-context_results.csv'
COPY_CSV = '../../results_to_be_processed/refined-context_results_copy.csv'
FIXED_CSV = '../../results_to_be_processed/refined-context_results_fixed.csv'

# Output file
FINAL_CSV = '../../results_to_be_processed/refined-context_results_final.csv'

print("📊 Combining and fixing results...")
print(f"Primary CSV: {PRIMARY_CSV}")
print(f"Copy CSV: {COPY_CSV}")
print(f"Fixed CSV: {FIXED_CSV}")
print(f"Final CSV: {FINAL_CSV}")
print()

try:
    # Load all data
    df_primary = pd.read_csv(PRIMARY_CSV)
    df_copy = pd.read_csv(COPY_CSV)
    df_fixed = pd.read_csv(FIXED_CSV)
    
    print(f"✅ Loaded {len(df_primary)} results from primary CSV")
    print(f"✅ Loaded {len(df_copy)} results from copy CSV")
    print(f"✅ Loaded {len(df_fixed)} results from fixed CSV")
    print()
    
    # Combine primary and copy results
    df_combined = pd.concat([df_copy, df_primary], ignore_index=True)
    
    # Ensure question_id is unique and correctly assigned
    df_combined['question_id'] = df_combined.index + 1
    
    print(f"✅ Combined {len(df_combined)} results")
    print()
    
    # Replace fixed questions in the combined dataframe
    fixed_count = 0
    for _, fixed_row in df_fixed.iterrows():
        question_id = fixed_row['question_id']
        
        # Find the corresponding question in the combined dataframe
        if question_id in df_combined['question_id'].values:
            # Update the context, answer, and references
            df_combined.loc[df_combined['question_id'] == question_id, ['context', 'answer', 'references']] = fixed_row[['context', 'answer', 'references']]
            fixed_count += 1
    
    print(f"✅ Replaced {fixed_count} questions with fixed versions")
    print()
    
    # Save the final combined results
    df_combined.to_csv(FINAL_CSV, index=False)
    print(f"✅ Saved final results to {FINAL_CSV}")
    print()
    
    # Print summary
    print(f"📊 Summary:")
    print(f"  Total questions: {len(df_combined)}")
    print(f"  Questions from copy CSV: {len(df_copy)}")
    print(f"  Questions from primary CSV: {len(df_primary)}")
    print(f"  Questions fixed: {fixed_count}")
    
    # Verify no placeholders remain
    def has_placeholders(context):
        if pd.isna(context):
            return True
        placeholders = ['{query}', '{ontology}', '{aql_results}', '{attribute_name}', '{value}', '{units}', '{region}', '{time}', '{method}']
        for placeholder in placeholders:
            if placeholder in str(context):
                return True
        return False
    
    placeholder_count = sum(1 for context in df_combined['context'] if has_placeholders(context))
    print(f"  Questions with placeholder context: {placeholder_count}")
    
    print(f"\n✅ Combine and fix completed successfully!")
    print(f"📝 Final results saved to: {FINAL_CSV}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
