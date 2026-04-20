"""llm wrapper and shared helpers for the multi-agent system."""

import os
import json
import re
import time
import random
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
        max_tokens: int = 1400,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

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

        # Retry with exponential backoff on rate-limit / transient errors.
        # Max ~5 min of total backoff: 2, 4, 8, 16, 32, 60, 60, 60 seconds.
        max_attempts = 8
        # Streaming avoids edge-level idle-connection kills that manifest as
        # "Server disconnected without sending a response" on slow generations
        # (large model, heavy prompts). Same tokens, same deterministic output
        # at temperature=0 / low temperatures; equivalent to non-streaming.
        use_stream = os.getenv("MISTRAL_STREAM", "1") not in ("0", "false", "False", "")
        for attempt in range(max_attempts):
            try:
                if use_stream:
                    parts: list = []
                    stream = self.client.chat.stream(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    for ev in stream:
                        # mistralai v1: each event has .data.choices[0].delta.content
                        data = getattr(ev, "data", ev)
                        choices = getattr(data, "choices", None) or []
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        chunk = getattr(delta, "content", None) if delta is not None else None
                        if chunk:
                            parts.append(chunk)
                    content = "".join(parts)
                    if not content:
                        raise RuntimeError("empty stream response")
                    return self._parse_output(content, force_json=force_json)
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                return self._parse_output(content, force_json=force_json)
            except Exception as e:
                msg = str(e)
                status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
                # Treat 429, 5xx, and connection/timeout errors as retryable.
                msg_l = msg.lower()
                retryable = (
                    status == 429
                    or (isinstance(status, int) and 500 <= status < 600)
                    or "429" in msg
                    or "rate limit" in msg_l
                    or "too many requests" in msg_l
                    or "timeout" in msg_l
                    or "connection" in msg_l
                    or "service unavailable" in msg_l
                    or "disconnected" in msg_l
                    or "server disconnected" in msg_l
                    or "remote protocol" in msg_l
                    or "read error" in msg_l
                    or "broken pipe" in msg_l
                    or "reset by peer" in msg_l
                    or "eof" in msg_l
                    or "incomplete read" in msg_l
                    or "bad gateway" in msg_l
                    or "gateway timeout" in msg_l
                    or "overloaded" in msg_l
                    or "capacity" in msg_l
                )
                if not retryable or attempt == max_attempts - 1:
                    logging.error(f"llm call failed (attempt {attempt+1}/{max_attempts}): {e}")
                    return None
                # Exponential backoff with jitter; cap at 60s.
                delay = min(60.0, (2 ** attempt)) + random.uniform(0, 1.5)
                logging.warning(
                    f"llm call retryable error (attempt {attempt+1}/{max_attempts}): {e} - sleeping {delay:.1f}s"
                )
                time.sleep(delay)
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
    max_tokens: int = 1400,
) -> MistralLLMWrapper:
    """create a mistral llm wrapper from env credentials."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env")
    # Large model on 40k+ token prompts can take > 60s (default) to start streaming.
    # Use 5-minute timeout so we don't see spurious "Server disconnected" errors.
    # Allow override via env var for local debugging.
    timeout_ms = int(os.getenv("MISTRAL_TIMEOUT_MS", "300000"))
    try:
        client = Mistral(api_key=api_key, timeout_ms=timeout_ms)
    except TypeError:
        # older SDK versions don't support timeout_ms kwarg
        client = Mistral(api_key=api_key)
    return MistralLLMWrapper(client, model=model, temperature=temperature, max_tokens=max_tokens)


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
