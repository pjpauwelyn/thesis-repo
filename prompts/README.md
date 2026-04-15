# Prompts

Prompt templates for all agents in the pipeline.

## Structure

```
prompts/
├── generation/        # Generation prompt templates
│   └── generation_prompt_exp4.txt  # Main generation prompt
├── ontology/          # Ontology extraction prompts
│   └── extract_attributes.txt  # Attribute extraction prompt
└── refinement/        # Context refinement prompts
    └── refinement_1pass_refined_exp4.txt  # 1-pass refined refinement prompt
```

## Prompt Types

### Generation Prompts
- **generation_prompt_exp4.txt**: Main prompt for answer generation
- Uses refined context to produce structured answers
- Includes reference formatting instructions

### Ontology Prompts
- **extract_attributes.txt**: Extracts ontology attributes from questions
- Identifies critical and contextual attributes
- Sorts by relevance/centrality

### Refinement Prompts
- **refinement_1pass_refined_exp4.txt**: 1-pass refined context refinement
- Uses AQL-only approach for efficient processing
- Extracts both findings and references from AQL results
- Creates compact context for generation agent

## Usage

Prompts are automatically loaded by agents based on the pipeline configuration.

## Key Features

- **Modular design**: Each agent has its own prompt template
- **Versioned**: Prompts include version numbers for tracking
- **Optimized**: Designed for efficient token usage
- **Structured**: Clear instructions and formatting guidelines