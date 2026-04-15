"""llm wrapper and shared helpers for the multi-agent system."""

import os
import json
import re
import logging
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

# default model used across the pipeline; experiments can override via env or config
DEFAULT_MODEL = os.getenv("PIPELINE_MODEL", "mistral-small-24b-instruct-2503")


class MistralLLMWrapper:
    """thin wrapper around the mistral chat api with json mode support."""

    def __init__(
        self,
        client: Mistral,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt_text: str, force_json: bool = False) -> Optional[Union[dict, str]]:
        """send a prompt and return parsed json (force_json=True) or raw text."""
        if force_json:
            full_prompt = (
                "You MUST respond with ONLY valid JSON when instructed to do so.\n"
                + prompt_text
            )
        else:
            full_prompt = prompt_text

        messages = [{"role": "user", "content": full_prompt}]

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=8000,
            )
            content = response.choices[0].message.content
            return self._parse_output(content, force_json=force_json)
        except Exception as e:
            logging.error(f"llm call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _parse_output(self, content: str, force_json: bool = False) -> Optional[Union[dict, str]]:
        """strip markdown fences, then either parse json or return raw text."""
        content = re.sub(r"```json\n?", "", content)
        content = re.sub(r"```\n?", "", content).strip()

        if not content:
            logging.error("empty content from llm")
            return None

        if not force_json:
            return content

        try:
            data = json.loads(content)
            # normalise bare lists into a dict so callers always get a dict
            if isinstance(data, list):
                data = {"assessments": data}
            return data
        except json.JSONDecodeError:
            logging.error("json parsing failed with force_json=True")
            return None


def get_llm_model(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> MistralLLMWrapper:
    """create a mistral llm wrapper from env credentials."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env")
    client = Mistral(api_key=api_key)
    return MistralLLMWrapper(client, model=model, temperature=temperature)


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """return a logger with a stream handler; avoids duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
