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

# not used in latest version
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

# not used in latest version
class DocumentAssessment(BaseModel):
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
