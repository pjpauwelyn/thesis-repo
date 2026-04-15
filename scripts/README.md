# Scripts

**IMPORTANT: This is where ALL utility scripts should be placed!**

Utility and analysis scripts for the pipeline. Any new scripts you create should go in this folder to maintain proper organization.

## Structure

```
scripts/
├── check_unique_questions.py  # Check for unique questions
└── validation_evaluation.py    # Validation and evaluation scripts
```

## Available Scripts

### check_unique_questions.py
- Checks input CSV for unique questions
- Identifies duplicates and potential issues
- Helps ensure data quality

### validation_evaluation.py
- Validates pipeline outputs
- Performs evaluation of generated answers
- Includes scoring and quality assessment

## Usage

```bash
# Check for unique questions
python check_unique_questions.py input.csv

# Run validation and evaluation
python validation_evaluation.py results.csv
```

## Adding New Scripts

**When creating new scripts, ALWAYS place them in this folder!**

This maintains the clean organization and prevents clutter in the main directory. Examples of scripts that belong here:
- Data preprocessing scripts
- Analysis and visualization tools
- Utility functions for CSV manipulation
- Evaluation and validation scripts
- Any helper scripts for the pipeline

## Key Features

- **Data quality**: Ensures input data integrity
- **Validation**: Checks pipeline outputs for correctness
- **Evaluation**: Assesses answer quality and relevance
- **Reporting**: Provides detailed analysis and metrics
- **Organization**: Keeps all utility scripts in one place