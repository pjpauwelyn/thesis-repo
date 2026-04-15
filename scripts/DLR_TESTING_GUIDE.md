# DLR Testing Guide

## ✅ Successful DLR Pipeline Testing

**Issue Identified:** The pipeline requires CSV files with specific columns including `aql_results` for proper context refinement.

## DLR File Types

### 1. Generated Questions File (❌ Missing AQL Results)
**File:** `DARES25_EarthObservation_generated_questions_v1.csv`
**Columns:** `idx, intent, topic, terms, context, instructions, question`
**Issue:** Missing `aql_results` column → "No documents available"

### 2. Results File (✅ Proper Structure)
**File:** `dlr-results.csv`
**Columns:** `idx, model, temperature, question, aql_params, query, aql_results, context, rag_answer, zero_shot_answer, timestamp, structured_context, rag_two_steps_answer`
**Status:** Contains `aql_results` → Full functionality

## Correct Usage

### ❌ Incorrect (Missing AQL Results)
```bash
# This will show "No documents available"
python core/main.py run --type 1_pass_with_ontology_refined \
  --csv data/dlr/DARES25_EarthObservation_generated_questions_v1.csv \
  --num 2
```

### ✅ Correct (With AQL Results)
```bash
# This works perfectly with full context refinement
python core/main.py run --type 1_pass_with_ontology_refined \
  --csv data/dlr/dlr-results.csv \
  --num 2 --overwrite
```

## Test Results with Correct File

### Question 1
**Input:** "How do variations in cryospheric indicators, such as glacier retreat and sea ice extent, influence carbon flux dynamics and land surface heat absorption patterns?"

**Results:**
- ✅ Ontology: 5 attributes extracted
- ✅ Context: 9361 chars (~2340 tokens)
- ✅ AQL Results: Properly parsed and used
- ✅ References: 6 academic references generated
- ✅ Processing Time: 29.48 seconds

### Question 2  
**Input:** "How do cryospheric indicators compare to land surface/agriculture indicators in their sensitivity to climate-induced changes in albedo?"

**Results:**
- ✅ Ontology: 5 attributes extracted
- ✅ Context: 9954 chars (~2488 tokens)
- ✅ AQL Results: Properly parsed and used
- ✅ References: 6 academic references generated
- ✅ Processing Time: Included in total

## Key Differences

### Without AQL Results (Generated Questions File)
```
Context: "No documents available"
Clean AQL Results: (empty)
References: (empty)
```

### With AQL Results (Results File)
```
Context: Full refined context with:
- Ontology summary (critical + supporting attributes)
- Topics and information sections
- Findings and evidence
- References and citations

Clean AQL Results: Parsed JSON with document metadata
References: 6 formatted academic references
```

## File Structure Requirements

For the pipeline to work correctly, input CSV must contain:

### Required Columns
1. **question** - The question text
2. **aql_results** - JSON string with AQL query results
3. **structured_context** - (Optional but recommended) Pre-structured context
4. **context** - (Optional) Additional context information

### Example Structure
```csv
idx,question,aql_results,structured_context,context
1,"What are cryospheric indicators?","[{\"title\": \"Cryosphere Study\", \"abstract\": \"...\", \"uri\": \"...\"}]","Structured info...","Additional context..."
```

## Available DLR Files in clean_repo

### `data/dlr/` Directory
1. **DARES25_EarthObsertvation_QA_RAG_results_v1.csv** - Generated questions (no AQL)
2. **DARES25_EarthObservation_generated_questions_v1.csv** - Generated questions (no AQL)  
3. **dlr-results.csv** - Full results with AQL data ✅

## Recommendations

### For Testing
```bash
# Use the results file for full functionality
python core/main.py run --type 1_pass_with_ontology_refined \
  --csv data/dlr/dlr-results.csv \
  --num 5 --overwrite
```

### For Production
```bash
# Ensure your input CSV has aql_results column
python core/main.py run --type 1_pass_with_ontology_refined \
  --csv your_data_with_aql.csv \
  --num 10
```

### Data Preparation
If you have questions without AQL results:
1. Run AQL queries to get results
2. Add `aql_results` column to CSV
3. Include JSON-formatted document lists

## Troubleshooting

### Issue: "No documents available"
**Cause:** Missing `aql_results` column in input CSV
**Solution:** Use `dlr-results.csv` or add AQL results to your data

### Issue: Empty context
**Cause:** Input CSV lacks required columns
**Solution:** Ensure CSV has `question` and `aql_results` columns

### Issue: API warnings
**Cause:** LLM API constraints (normal)
**Solution:** Monitor but not critical for functionality

## Conclusion

The pipeline works perfectly when provided with the correct input format:
- ✅ **Use `dlr-results.csv`** for testing
- ✅ **Ensure `aql_results` column** in production data
- ✅ **Full functionality** with proper AQL data
- ✅ **Complete context refinement** and answer generation

**Status: FULLY FUNCTIONAL WITH PROPER INPUT DATA** 🚀