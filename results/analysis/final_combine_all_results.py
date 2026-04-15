#!/usr/bin/env python3
"""
📊 FINAL COMBINE ALL RESULTS

Combines all results to ensure all 70 questions are included with correct contexts.
"""

import pandas as pd
import os
from pathlib import Path

# Input files
FINAL_FIXED_CSV = '../../results_to_be_processed/refined-context_results_final_fixed.csv'
COMBINED_CSV = '../../results_to_be_processed/refined-context_results_final.csv'

# Output file
COMPLETE_CSV = '../../results_to_be_processed/refined-context_results_complete.csv'

print("📊 Combining all results...")
print(f"Final fixed CSV: {FINAL_FIXED_CSV}")
print(f"Combined CSV: {COMBINED_CSV}")
print(f"Complete CSV: {COMPLETE_CSV}")
print()

try:
    # Load all data
    df_final_fixed = pd.read_csv(FINAL_FIXED_CSV)
    df_combined = pd.read_csv(COMBINED_CSV)
    
    print(f"✅ Loaded {len(df_final_fixed)} results from final fixed CSV")
    print(f"✅ Loaded {len(df_combined)} results from combined CSV")
    print()
    
    # Replace the fixed questions in the combined dataframe
    fixed_count = 0
    for _, fixed_row in df_final_fixed.iterrows():
        question_id = fixed_row['question_id']
        
        # Find the corresponding question in the combined dataframe
        if question_id in df_combined['question_id'].values:
            # Update the context, answer, and references
            df_combined.loc[df_combined['question_id'] == question_id, ['context', 'answer', 'references']] = fixed_row[['context', 'answer', 'references']]
            fixed_count += 1
    
    print(f"✅ Replaced {fixed_count} questions with fixed versions")
    print()
    
    # Save the complete results
    df_combined.to_csv(COMPLETE_CSV, index=False)
    print(f"✅ Saved complete results to {COMPLETE_CSV}")
    print()
    
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
    
    # Print summary
    print(f"📊 Summary:")
    print(f"  Total questions: {len(df_combined)}")
    print(f"  Questions with placeholder context: {placeholder_count}")
    
    if placeholder_count == 0:
        print(f"\n✅ All questions have correct contexts!")
    else:
        print(f"\n⚠️ {placeholder_count} questions still have placeholder context")
    
    print(f"\n📝 Complete results saved to: {COMPLETE_CSV}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
