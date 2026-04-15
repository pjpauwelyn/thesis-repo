# Core System

The core system contains the main pipeline components and agents.

## Structure

```
core/
├── main.py                # Main pipeline orchestrator
├── agents/                # Agent implementations
│   ├── base_agent.py      # Base agent class
│   ├── base_refinement_agent.py  # Base refinement agent
│   ├── ontology_agent.py  # Ontology construction agent
│   ├── generation_agent.py # Answer generation agent
│   └── refinement_agent_1pass_refined.py # 1-pass refined refinement agent
└── utils/                 # Utility functions
    ├── data_models.py      # Data structures
    ├── openalex_client.py  # OpenAlex API client
    └── aql_parser.py       # AQL results parser
```

## Key Components

### main.py
- Orchestrates the entire pipeline
- Handles different pipeline types (1_pass_with_ontology_refined, etc.)
- Manages performance monitoring and logging
- Provides command-line interface

### Agents
- **ontology_agent.py**: Extracts ontology attributes from questions
- **refinement_agent_1pass_refined.py**: Refines context using AQL-only approach
- **generation_agent.py**: Generates answers from refined context
- **base_agent.py**: Base class for all agents
- **base_refinement_agent.py**: Base class for refinement agents

### Utils
- **data_models.py**: Contains data structures (DynamicOntology, RefinedContext, etc.)
- **openalex_client.py**: Handles OpenAlex API interactions
- **aql_parser.py**: Parses AQL query results

## Usage

```bash
# Run the main pipeline
python main.py run --type 1_pass_with_ontology_refined --csv data.csv --num 5

# Available pipeline types:
# - 1_pass_with_ontology_refined (main)
# - 1_pass_with_ontology
# - 1_pass_without_ontology
```

## Key Features

- **AQL-only refinement**: Efficient context processing
- **Performance monitoring**: Real-time tracking and alerts
- **Comprehensive logging**: Debug and performance logs
- **Error handling**: Robust exception management
- **Configurable**: Multiple pipeline types and options