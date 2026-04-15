"""constructs a dynamic ontology from a user question via llm calls."""

import os
from typing import List

from core.agents.base_agent import BaseAgent
from core.utils.data_models import (
    AttributeValuePair,
    DynamicOntology,
    LogicalRelationship,
)


class OntologyConstructionAgent(BaseAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/ontology"):
        super().__init__("OntologyConstructionAgent", llm)
        self.prompt_dir = prompt_dir

    def load_prompt(self, filename: str) -> str:
        filepath = os.path.join(self.prompt_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def process(self, question: str, include_relationships: bool = True) -> DynamicOntology:
        av_pairs = self._extract_attribute_value_pairs(question)
        self.logger.info(f"extracted {len(av_pairs)} attribute-value pairs")

        relationships: List[LogicalRelationship] = []
        if include_relationships:
            relationships = self._define_logical_relationships(av_pairs, question)
            self.logger.info(f"defined {len(relationships)} logical relationships")

        return DynamicOntology(
            attribute_value_pairs=av_pairs,
            logical_relationships=relationships,
            source_query=question,
            should_use_ontology=True,
            include_relationships=include_relationships,
        )

    # ------------------------------------------------------------------
    # private llm helpers
    # ------------------------------------------------------------------

    def _extract_attribute_value_pairs(self, question: str) -> List[AttributeValuePair]:
        template = self.load_prompt("extract_attributes_slim.txt")
        prompt = template.replace("{question}", question)

        try:
            response = self.llm.invoke(prompt, force_json=True)
            if isinstance(response, dict) and "pairs" in response:
                return [
                    AttributeValuePair(
                        attribute=p.get("attribute", ""),
                        value=p.get("value", ""),
                        description=p.get("description", ""),
                        centrality=p.get("centrality", 0.5),
                    )
                    for p in response["pairs"]
                ]
            self.logger.error(f"unexpected av-pair response type: {type(response)}")
            return []
        except Exception as e:
            self.logger.error(f"attribute extraction failed: {e}")
            return []

    def _define_logical_relationships(
        self, av_pairs: List[AttributeValuePair], question: str
    ) -> List[LogicalRelationship]:
        if not av_pairs:
            return []

        attrs_text = "\n".join(f"- {av.attribute}: {av.value}" for av in av_pairs)
        template = self.load_prompt("extract_relationships.txt")
        prompt = (
            template.replace("{question}", question).replace("{attributes}", attrs_text)
        )

        try:
            response = self.llm.invoke(prompt, force_json=True)
            if isinstance(response, dict) and "relationships" in response:
                return [
                    LogicalRelationship(
                        source_attribute=r.get("source_attribute", ""),
                        source_value=r.get("source_value", ""),
                        target_attribute=r.get("target_attribute", ""),
                        target_value=r.get("target_value", ""),
                        relationship_type=r.get("relationship_type", "relates_to"),
                        logical_constraint=r.get("logical_constraint", ""),
                    )
                    for r in response["relationships"]
                ]
            self.logger.error(f"unexpected relationship response type: {type(response)}")
            return []
        except Exception as e:
            self.logger.error(f"relationship extraction failed: {e}")
            return []
