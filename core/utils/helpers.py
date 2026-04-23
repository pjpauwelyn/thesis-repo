"""llm wrapper and shared helpers for the multi-agent system."""

import os
import json
import re
import time
import random
import logging
from typing import Optional, Union
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

DEFAULT_MODEL = os.getenv("PIPELINE_MODEL", "mistral-small-latest")

# Per-model Mistral client timeout defaults (milliseconds).
_MODEL_TIMEOUT_MS: dict = {
    "mistral-small-latest":  90_000,
    "mistral-medium-latest": 180_000,
    "mistral-large-latest":  420_000,
}
_DEFAULT_TIMEOUT_MS = int(os.getenv("MISTRAL_TIMEOUT_MS", "300000"))

_log = logging.getLogger(__name__)

# Transient API / network errors — safe to retry with exponential back-off.
_API_RETRYABLE_TOKENS = (
    "429", "rate limit", "too many requests", "timeout",
    "connection", "service unavailable", "disconnected",
    "server disconnected", "remote protocol", "read error",
    "broken pipe", "reset by peer", "eof", "incomplete read",
    "bad gateway", "gateway timeout", "overloaded", "capacity",
    "server error", "internal error", "empty stream",
)

# Local Python bugs — return None immediately, no retry.
# Retrying these wastes up to 8× exponential back-off (~4 min) on real bugs.
_CODE_ERROR_TOKENS = (
    "nonetype",
    "subscriptable",
    "object has no attribute",
    "attributeerror",
    "typeerror",
)


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
        from core.utils.logger import log_llm_retry, log_llm_failure

        full_prompt = (
            "You MUST respond with ONLY valid JSON when instructed to do so.\n" + prompt_text
            if force_json else prompt_text
        )
        messages = [{"role": "user", "content": full_prompt}]
        use_stream = os.getenv("MISTRAL_STREAM", "1") not in ("0", "false", "False", "")

        for attempt in range(8):
            try:
                if use_stream:
                    parts: list = []
                    for ev in self.client.chat.stream(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    ):
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
                else:
                    resp = self.client.chat.complete(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = resp.choices[0].message.content
                return self._parse_output(content, force_json=force_json)

            except Exception as e:
                msg = str(e).lower()
                status = getattr(e, "status_code", None) or getattr(e, "http_status", None)

                # Local code bug — bail immediately, no retry.
                if any(tok in msg for tok in _CODE_ERROR_TOKENS):
                    _log.error(
                        "llm call aborted — local code error (will not retry): %s", e
                    )
                    return None

                retryable = (
                    status == 429
                    or (isinstance(status, int) and 500 <= status < 600)
                    or any(tok in msg for tok in _API_RETRYABLE_TOKENS)
                )
                if not retryable or attempt == 7:
                    log_llm_failure(_log, attempt + 1, 8, str(e))
                    return None
                delay = min(60.0, 2 ** attempt) + random.uniform(0, 1.5)
                log_llm_retry(_log, attempt + 1, 8, str(e), delay)
                time.sleep(delay)
        return None

    def _parse_output(self, content: str, force_json: bool = False) -> Optional[Union[dict, str]]:
        content = re.sub(r"```json\n?", "", content)
        content = re.sub(r"```\n?", "", content).strip()
        if not content:
            _log.error("empty content from llm")
            return None
        if not force_json:
            return content
        try:
            data = json.loads(content)
            if isinstance(data, list):
                data = {"assessments": data}
            return data
        except json.JSONDecodeError:
            _log.error("json parsing failed with force_json=True")
            return None


class OpenRouterLLMWrapper:
    """wrapper for openrouter-hosted models via openai-compatible api."""

    def __init__(self, client, model: str, temperature: float = 0.3, max_tokens: int = 1400):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt_text: str, force_json: bool = False) -> Optional[Union[dict, str]]:
        from core.utils.logger import log_llm_retry, log_llm_failure

        full_prompt = (
            "You MUST respond with ONLY valid JSON when instructed to do so.\n" + prompt_text
            if force_json else prompt_text
        )
        messages = [{"role": "user", "content": full_prompt}]

        for attempt in range(8):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content or ""
                return self._parse_output(content, force_json=force_json)
            except Exception as e:
                msg = str(e).lower()
                status = getattr(e, "status_code", None)

                # Local code bug — bail immediately, no retry.
                if any(tok in msg for tok in _CODE_ERROR_TOKENS):
                    _log.error(
                        "llm call aborted — local code error (will not retry): %s", e
                    )
                    return None

                retryable = (
                    status == 429
                    or (isinstance(status, int) and 500 <= status < 600)
                    or any(tok in msg for tok in _API_RETRYABLE_TOKENS)
                )
                if not retryable or attempt == 7:
                    log_llm_failure(_log, attempt + 1, 8, str(e))
                    return None
                delay = min(60.0, 2 ** attempt) + random.uniform(0, 1.5)
                log_llm_retry(_log, attempt + 1, 8, str(e), delay)
                time.sleep(delay)
        return None

    def _parse_output(self, content: str, force_json: bool = False) -> Optional[Union[dict, str]]:
        content = re.sub(r"```json\n?", "", content)
        content = re.sub(r"```\n?", "", content).strip()
        if not content:
            _log.error("empty content from openrouter")
            return None
        if not force_json:
            return content
        try:
            data = json.loads(content)
            if isinstance(data, list):
                data = {"assessments": data}
            return data
        except json.JSONDecodeError:
            _log.error("json parsing failed with force_json=True (openrouter)")
            return None


def get_llm_model(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1400,
    timeout_s: Optional[int] = None,
) -> Union[MistralLLMWrapper, OpenRouterLLMWrapper]:
    """create an llm wrapper.

    timeout_s  — per-call socket timeout in seconds.  When None the model-
                 specific default from _MODEL_TIMEOUT_MS is used.
    """
    if model.startswith(("qwen/", "openrouter/", "mistralai/", "google/")):
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        return OpenRouterLLMWrapper(client, model=model, temperature=temperature, max_tokens=max_tokens)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env")

    if timeout_s is not None:
        timeout_ms = timeout_s * 1000
    else:
        timeout_ms = _MODEL_TIMEOUT_MS.get(model, _DEFAULT_TIMEOUT_MS)

    try:
        client = Mistral(api_key=api_key, timeout_ms=timeout_ms)
    except TypeError:
        client = Mistral(api_key=api_key)
    return MistralLLMWrapper(client, model=model, temperature=temperature, max_tokens=max_tokens)


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger.  Delegates to core.utils.logger when available."""
    try:
        from core.utils.logger import get_logger
        return get_logger(name)
    except ImportError:
        pass
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
    return logger
