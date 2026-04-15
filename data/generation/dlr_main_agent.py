# dlr_generation/dlr_agent.py

from core.agents.base_agent import BaseAgent
from core.agents.generation_agent import Answer
from typing import Optional
import logging
import os

class DLRAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__("DLRAgent", llm)
        self.zero_shot_template = self._load_template("prompts/generation/zero_shot.txt")
        self.dlr_refinement_template = self._load_template("prompts/generation_dlr/dlr_grounded_prompt.txt")
        
    def _load_template(self, path: str) -> str:
        """Load template file with error handling"""
        try:
            # Try current directory first
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Fallback to parent directory
            full_path = os.path.join('..', path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            self.logger.error(f"Template not found: {path} or {full_path}")
            raise FileNotFoundError(f"Template not found: {path}")
        except Exception as e:
            self.logger.error(f"Error loading template {path}: {e}")
            raise
    
    def generate(self, question: str, context: str = "", pipeline_type: str = "dlr_1step") -> Answer:
        """Generate answer using specified DLR pipeline"""
        if pipeline_type == "dlr_1step":
            # Pure zero-shot (no context)
            self.logger.info(f"Generating DLR 1-step answer for: {question[:60]}...")
            prompt = self.zero_shot_template.replace("{question}", question)
            answer = self._generate_with_llm(prompt)

        elif pipeline_type == "dlr_2step":
            # Step 1: Zero-shot generation
            self.logger.info(f"Generating DLR 2-step answer for: {question[:60]}...")
            draft_answer = self.generate(question, pipeline_type="dlr_1step").answer

            # Step 2: DLR refinement with context
            refinement_prompt = self.dlr_refinement_template
            refinement_prompt = refinement_prompt.replace("{question}", question)
            refinement_prompt = refinement_prompt.replace("{draft_answer}", draft_answer)
            refinement_prompt = refinement_prompt.replace("{context}", context)

            answer = self._generate_with_llm(refinement_prompt)

        else:
            raise ValueError(f"Unknown DLR pipeline type: {pipeline_type}")

        return Answer(answer=answer)
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate answer using the LLM with error handling"""
        try:
            response = self.llm.invoke(prompt, force_json=False)
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return "Error: Could not generate answer due to LLM failure."

    def generate_1step(self, question: str) -> Answer:
        """Convenience method for 1-step generation"""
        return self.generate(question, pipeline_type="dlr_1step")
    
    def generate_2step(self, question: str, context: str) -> Answer:
        """Convenience method for 2-step generation"""
        return self.generate(question, context, pipeline_type="dlr_2step")
    
    def process(self, input_data):
        """Required abstract method implementation"""
        # This method is required by the base class but not used in DLR pipelines
        raise NotImplementedError("DLR agent uses generate() method instead of process()")