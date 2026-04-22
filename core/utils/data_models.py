"""pydantic models shared across all pipeline stages."""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AttributeValuePair(BaseModel):
    attribute: str = Field(..., description="attribute concept")
    value: str = Field(..., description="specific value")
    description: str = Field(..., description="short explanation")
    value_type: Literal["numeric", "categorical", "temporal", "spatial"] = "categorical"
    constraints: List[str] = Field(default_factory=list)
    centrality: float = Field(..., ge=0.0, le=1.0, description="importance score 0-1")


class LogicalRelationship(BaseModel):
    source_attribute: str
    source_value: str
    target_attribute: str
    target_value: str
    relationship_type: str = Field(..., description="influences, causes, requires, etc.")
    logical_constraint: str


class DynamicOntology(BaseModel):
    attribute_value_pairs: List[AttributeValuePair]
    logical_relationships: List[LogicalRelationship]
    source_query: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    should_use_ontology: bool = Field(default=True)
    include_relationships: bool = Field(default=True)

    def get_critical_attributes(self, threshold: float = 0.8) -> List[AttributeValuePair]:
        return [av for av in self.attribute_value_pairs if av.centrality >= threshold]

    def get_contextual_attributes(self, threshold: float = 0.4) -> List[AttributeValuePair]:
        return [av for av in self.attribute_value_pairs if av.centrality >= threshold]


class DocumentAssessment(BaseModel):
    """per-document assessment produced by the refinement agent."""
    id: Optional[str] = None
    title_or_name: str
    position: int
    type: Literal["publication", "keyword"]
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    scope: str
    caveats: str
    instruction: str
    include: bool = Field(default=True)
    geographic_scope: Optional[str] = None
    geographic_match: Optional[str] = None
    temporal_scope: Optional[str] = None
    temporal_match: Optional[str] = None
    what_is_relevant: Optional[str] = None
    context_limitations: Optional[str] = None
    raw_metrics_text: Optional[str] = None
    spatial_context_text: Optional[str] = None
    temporal_context_text: Optional[str] = None
    supporting_details_text: Optional[str] = None
    generation_instruction: Optional[str] = None
    citation: Optional[str] = None
    full_reference: Optional[str] = None
    reference_id: Optional[str] = None


class RefinedContext(BaseModel):
    original_context: str
    question: str
    ontology_summary: str
    assessments: List[DocumentAssessment]
    summary: str
    enriched_context: Optional[str] = None
    enriched_documents: Optional[List[Dict[str, Any]]] = None

    def get_included_assessments(self) -> List[DocumentAssessment]:
        return [a for a in self.assessments if a.include]

    def get_sorted_by_relevance(self) -> List[DocumentAssessment]:
        return sorted(
            self.get_included_assessments(),
            key=lambda a: a.relevance_score,
            reverse=True,
        )


# ---------------------------------------------------------------------------
# adaptive pipeline (set5)  --  question profile + routing decision
# ---------------------------------------------------------------------------

class QuestionProfile(BaseModel):
    """compact question characterisation emitted by the profiler.

    Fields used by the router (rules.yaml):
      question_type, complexity, quantitativity, spatial_specificity,
      temporal_specificity, confidence

    Fields used downstream (refinement query hint):
      needs_numeric_emphasis, methodological_depth

    answer_shape and expected_length have been intentionally removed:
      - answer_shape was always overwritten by router._resolve_answer_shape()
        and never read from the LLM response.
      - expected_length was never consumed anywhere in the pipeline.
      Both fields added LLM noise with no routing or generation benefit.
      Generation structure is controlled by the prompt templates selected
      per tier, not by per-question shape directives.
    """
    identity: str
    one_line_summary: str = ""
    question_type: Literal[
        "definition", "mechanism", "comparison",
        "quantitative", "method_eval", "application", "continuous",
    ] = "continuous"
    complexity: float = Field(0.5, ge=0.0, le=1.0)
    quantitativity: float = Field(0.3, ge=0.0, le=1.0)
    spatial_specificity: float = Field(0.1, ge=0.0, le=1.0)
    temporal_specificity: float = Field(0.1, ge=0.0, le=1.0)
    methodological_depth: float = Field(0.1, ge=0.0, le=1.0)
    needs_numeric_emphasis: bool = False
    # None = parse failure -> safety net triggers in router
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """resolved routing decision for one question.

    Per-agent model split
    ---------------------
    model_name         : generation model (and refinement fallback if
                         refinement_model_name is not set)
    refinement_model_name: explicit refinement model; when None the pipeline
                         falls back to model_name so existing rules that only
                         set model_name keep working unchanged.

    Per-call timeouts
    -----------------
    timeout_refine_s   : hard ceiling for the refinement LLM call (seconds).
    timeout_generate_s : hard ceiling for the generation LLM call (seconds).
    Both are passed to get_llm_model() so the Mistral client socket is closed
    after this deadline rather than hanging indefinitely.
    Values are model-specific (large needs 360s, small needs 60s) and are set
    in rules.yaml so the ceiling always matches the assigned model.
    """
    # --- generation / fallback model ---
    model_name: str = "mistral-small-latest"
    # --- refinement model (None -> use model_name) ---
    refinement_model_name: Optional[str] = None

    # --- evidence ---
    evidence_mode: Literal["abstracts", "excerpts_narrow", "excerpts_full"] = "abstracts"
    top_k_per_doc: int = 6
    per_doc_budget: int = 6000
    global_budget: int = 30000

    # --- prompts ---
    refinement_prompt: str = "refinement_1pass_refined_exp4.txt"
    generation_prompt: str = "generation_prompt_exp4.txt"

    # --- filters / synthesis ---
    scope_filter: bool = False
    synthesis_mode: Literal["homogeneous", "focused"] = "homogeneous"
    doc_filter_min_keep: int = 6

    # --- temperatures ---
    temperature_refine: float = 0.1
    temperature_generate: float = 0.2

    # --- per-call timeouts (seconds, model-aware) ---
    timeout_refine_s: int = 120
    timeout_generate_s: int = 120

    # --- generation caps ---
    gen_context_cap: int = 60_000
    max_output_tokens: int = 700
    system_prompt_modifier: str = ""

    # --- routing metadata ---
    rule_hit: str = "fallback"
    reason: str = ""
