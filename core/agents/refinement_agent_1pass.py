# agents/refinement_agent_1pass.py

from core.agents.base_refinement_agent import BaseRefinementAgent
from core.utils.data_models import DynamicOntology, DocumentAssessment
from typing import List, Dict, Any, Optional
import re

class RefinementAgent(BaseRefinementAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/refinement"):
        super().__init__("RefinementAgent1Pass", llm, prompt_dir)

    def _assess_documents(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        ontology: Optional[DynamicOntology],
        include_ontology: bool,
        aql_results_str: Optional[str] = None
    ) -> List[DocumentAssessment]:
        if not documents:
            return []
        
        docs_text = self._format_documents_for_prompt(documents)
        prompt = self._build_assessment_prompt(question, documents, ontology, include_ontology, docs_text)
        
        self.logger.debug(f"Calling LLM for assessment of {len(documents)} documents")
        try:
            response = self.llm.invoke(prompt, force_json=True)
            if not response:
                return []
            
            assessments_data = response.get("assessments", response if isinstance(response, list) else [])
            assessments = []
            
            for item in assessments_data:
                try:
                    if isinstance(item, dict):
                        assessments.append(DocumentAssessment(**item))
                    elif isinstance(item, DocumentAssessment):
                        assessments.append(item)
                except Exception as e:
                    self.logger.debug(f"Skip assessment item: {e}")
            
            return assessments
        except Exception as e:
            self.logger.error(f"LLM assessment failed: {e}")
            return []

    def _build_assessment_prompt(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        ontology: Optional[DynamicOntology],
        include_ontology: bool,
        docs_text: str
    ) -> str:
        template = self.load_prompt("refinement_prompt_1pass_slim.txt")
        
        if include_ontology and ontology:
            critical_attrs = "\n".join([
                f"- {av.attribute}: {av.value} (centrality: {av.centrality:.2f})"
                for av in ontology.get_critical_attributes(threshold=0.7)
            ])
            supporting_attrs = "\n".join([
                f"- {av.attribute}: {av.value} (centrality: {av.centrality:.2f})"
                for av in ontology.get_contextual_attributes(threshold=0.3)
                if av.centrality < 0.7
            ])
        else:
            critical_attrs = "Ignore any mention of ontology. Assess documents purely on relevance to the question."
            supporting_attrs = "No ontology constraints."
        
        prompt = template.replace("{question}", question)
        prompt = prompt.replace("{critical_attributes}", critical_attrs)
        prompt = prompt.replace("{supporting_attributes}", supporting_attrs)
        prompt = prompt.replace("{documents}", docs_text)
        
        return prompt

    def _format_documents_for_prompt(self, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "No documents provided"
        
        formatted = []
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            title = doc.get("title_or_name", "Unknown")
            content = doc.get("content", "")
            
            # Extract citation and reference_id from AQL results if available
            reference_id = doc.get("reference_id", f"[{doc_id}]")
            citation = doc.get("citation", "(Unknown, n.d.)")
            
            # Include AQL data if available
            doc_section = f"[{doc_id}] {title}\n{content[:500]}..."
            
            if doc.get("aql_data"):
                doc_section += f"\n\nAQL DATA:\n{doc['aql_data'][:1000]}..."
            
            if doc.get("aql_note"):
                doc_section += f"\n\nNOTE: {doc['aql_note']}"
            
            # Add citation and reference_id only if the document is a valid reference
            if self._is_valid_reference(doc):
                doc_section = f"[DOC] {doc_id}] {title}\n{content[:500]}..."
                doc_section += f"\n\nCITATION: {citation}"
                doc_section += f"\nREFERENCE_ID: {reference_id}"
            else:
                doc_section = f"[EVIDENCE] {doc_id}] {title}\n{content[:500]}..."
                doc_section += f"\n\nNOTE: Supporting information, not a valid reference. Do not cite as a document."
            
            formatted.append(doc_section)
        
        return "\n---\n".join(formatted)
    
    def _is_valid_reference(self, doc: Dict[str, Any]) -> bool:
        """Check if a document is a valid reference with title, authors, and URL."""
        # Check for required fields to be considered a valid reference
        has_title = bool(doc.get("title_or_name"))
        has_authors = bool(doc.get("authors") or doc.get("author"))
        has_url = bool(doc.get("url") or doc.get("source"))
        
        return has_title and (has_authors or has_url)
