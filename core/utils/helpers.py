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
# Also covers mistralai/-prefixed OpenRouter aliases for the same models.
_MODEL_TIMEOUT_MS: dict = {
    "mistral-small-latest":             90_000,
    "mistralai/mistral-small-latest":   90_000,
    "mistral-medium-latest":            180_000,
    "mistralai/mistral-medium-latest":  180_000,
    "mistral-large-latest":             480_000,
    "mistralai/mistral-large-latest":   480_000,
}
_DEFAULT_TIMEOUT_MS = int(os.getenv("MISTRAL_TIMEOUT_MS", "300000"))

# OpenRouter default timeout (seconds).  480s = TTFT ~30s + 1400 tok @47 tok/s
# + 180s headroom for heavy parallel load.  Rules.yaml timeout_generate_s
# overrides this per-tier via get_llm_model(timeout_s=...).
_OPENROUTER_DEFAULT_TIMEOUT_S = int(os.getenv("OPENROUTER_TIMEOUT_S", "480"))

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error classification tokens
# ---------------------------------------------------------------------------

# Transient API / network errors — safe to retry with exponential back-off.
_API_RETRYABLE_TOKENS = (
    # HTTP status signals
    "429", "rate limit", "too many requests",
    # Network / transport
    "timeout", "connection", "service unavailable", "disconnected",
    "server disconnected", "remote protocol", "read error",
    "broken pipe", "reset by peer", "eof", "incomplete read",
    # HTTP gateway errors
    "bad gateway", "gateway timeout",
    # Provider capacity
    "overloaded", "capacity", "server error", "internal error",
    # Streaming
    "empty stream",
    # OpenRouter-specific transients — provider temporarily unavailable
    "no available model",
    "provider returned error",
    "upstream error",
    "model is currently",       # "model is currently unavailable"
)

# Local Python bugs or unrecoverable API errors — return None immediately,
# no retry.  Retrying wastes up to 8× exponential back-off (~4 min) on real
# bugs or permanent auth/billing failures.
_CODE_ERROR_TOKENS = (
    # Python exceptions
    "nonetype",
    "subscriptable",
    "object has no attribute",
    "attributeerror",
    "typeerror",
    # Permanent API failures — retrying will never help
    "invalid api key",
    "authentication",
    "credits",                  # out of OpenRouter credits
    "context length exceeded",  # prompt is too long — must fix at call site
)


class MistralLLMWrapper:
    """Thin wrapper around the Mistral chat API with system-prompt and json-mode support."""

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

    def invoke(
        self,
        prompt_text: str,
        force_json: bool = False,
        system: str = "",
        max_tokens: Optional[int] = None,
    ) -> Optional[Union[dict, str]]:
        """Send a prompt and return parsed JSON (force_json=True) or raw text.

        Args:
            prompt_text: the user-turn content.
            force_json:  when True, prepends a JSON-mode instruction to the
                         user message and parses the response as JSON.
            system:      optional system prompt sent as a proper
                         {role: system} message before the user turn.
                         Pass GenerationAgent quality contracts here so they
                         reach the model at full authority.
            max_tokens:  per-call token ceiling override.  Falls back to
                         self.max_tokens (set at construction time) when None.
        """
        from core.utils.logger import log_llm_retry, log_llm_failure

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        user_content = (
            "You MUST respond with ONLY valid JSON when instructed to do so.\n" + prompt_text
            if force_json else prompt_text
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        use_stream = os.getenv("MISTRAL_STREAM", "1") not in ("0", "false", "False", "")

        for attempt in range(8):
            try:
                if use_stream:
                    parts: list = []
                    for ev in self.client.chat.stream(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=effective_max_tokens,
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
                        max_tokens=effective_max_tokens,
                    )
                    content = resp.choices[0].message.content
                return self._parse_output(content, force_json=force_json)

            except Exception as e:
                msg = str(e).lower()
                status = getattr(e, "status_code", None) or getattr(e, "http_status", None)

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
    """Wrapper for OpenRouter-hosted models via the OpenAI-compatible API.

    Uses streaming by default to avoid gateway timeouts on long generations
    (critical for mistral-large with 1400 output tokens and ~30s TTFT).
    timeout_s is applied as the OpenAI client socket deadline and defaults to
    _OPENROUTER_DEFAULT_TIMEOUT_S (480s), but is overridden per-tier via
    rules.yaml → timeout_generate_s / timeout_refine_s.
    """

    def __init__(
        self,
        client,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 1400,
        timeout_s: int = _OPENROUTER_DEFAULT_TIMEOUT_S,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def invoke(
        self,
        prompt_text: str,
        force_json: bool = False,
        system: str = "",
        max_tokens: Optional[int] = None,
    ) -> Optional[Union[dict, str]]:
        """Send a prompt and return parsed JSON (force_json=True) or raw text.

        Args:
            prompt_text: the user-turn content.
            force_json:  when True, prepends a JSON-mode instruction.
            system:      optional system prompt sent as {role: system} before the
                         user turn so quality contracts reach the model at full
                         authority rather than being buried in user content.
            max_tokens:  per-call override; falls back to self.max_tokens when None.
        """
        from core.utils.logger import log_llm_retry, log_llm_failure

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        user_content = (
            "You MUST respond with ONLY valid JSON when instructed to do so.\n" + prompt_text
            if force_json else prompt_text
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        for attempt in range(8):
            try:
                parts: list = []
                for chunk in self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=effective_max_tokens,
                    stream=True,
                    timeout=self.timeout_s,
                ):
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        parts.append(delta)
                content = "".join(parts)
                if not content:
                    raise RuntimeError("empty stream response")
                return self._parse_output(content, force_json=force_json)

            except Exception as e:
                msg = str(e).lower()
                status = getattr(e, "status_code", None)

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
    """Create and return an LLM wrapper for the given model.

    Routing:
      - Models with a provider prefix (mistralai/, qwen/, google/, openrouter/)
        → OpenRouterLLMWrapper using OPENROUTER_API_KEY
      - All others → MistralLLMWrapper using MISTRAL_API_KEY

    timeout_s — per-call socket timeout in seconds.
      When None the model-specific default from _MODEL_TIMEOUT_MS (Mistral) or
      _OPENROUTER_DEFAULT_TIMEOUT_S (OpenRouter) is used.  Pass the value from
      cfg.timeout_generate_s / cfg.timeout_refine_s to honour rules.yaml limits.
    """
    if model.startswith(("qwen/", "openrouter/", "mistralai/", "google/")):
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        effective_timeout = timeout_s or _OPENROUTER_DEFAULT_TIMEOUT_S
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=effective_timeout,
        )
        return OpenRouterLLMWrapper(
            client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=effective_timeout,
        )

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
