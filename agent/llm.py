"""LLM backends.

* OpenAIChat - any OpenAI-compatible endpoint: Groq or Gemini (free cloud tiers, nothing
  to download), OpenAI, or Ollama (local) - configured through .env. Rate limits (HTTP 429)
  are retried with backoff.
* MockLLM   - deterministic offline stand-in used only by the test-suite and the
  `--backend mock` smoke test. It replays each task's reference tool plan, so it
  proves the harness works end-to-end; its scores are NOT model results.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class FatalLLMError(RuntimeError):
    """Configuration problem (bad key, unknown model ...): retrying won't help, so stop the run."""


class OpenAIChat:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0):
        from openai import OpenAI  # imported lazily so tests run without the package

        self.model = model or os.getenv("LLM_MODEL")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        api_key = api_key or os.getenv("LLM_API_KEY")
        if not (self.model and base_url and api_key):
            raise FatalLLMError("LLM settings missing. Run `cp .env.example .env` and fill in "
                                "LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (see README).")
        if "your_key_here" in api_key or api_key.startswith("AIza_your"):
            raise FatalLLMError("LLM_API_KEY in .env is still the placeholder. Paste your real key.")
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.temperature = temperature
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "8"))
        # free tiers cap requests per minute; space calls out by at least this many seconds
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL_S", "0"))
        self._last_call = 0.0

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        import openai

        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature)
        if tools:
            kwargs["tools"] = tools
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                break
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise FatalLLMError(f"API key rejected: {exc}. Check LLM_API_KEY in .env.") from exc
            except openai.NotFoundError as exc:
                raise FatalLLMError(f"Model or URL not found: {exc}. Check LLM_MODEL and "
                                    "LLM_BASE_URL in .env.") from exc
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"LLM call failed after {attempt + 1} attempts: {exc}") from exc
                wait = min(60.0, 2.0 * 2 ** attempt)
                headers = getattr(getattr(exc, "response", None), "headers", None) or {}
                try:  # honour the server's own Retry-After hint when it sends one
                    wait = max(wait, min(90.0, float(headers.get("retry-after", 0))))
                except ValueError:
                    pass
                kind = "rate limit hit" if isinstance(exc, openai.RateLimitError) else "connection problem"
                print(f"    ({kind}; waiting {wait:.0f}s, retry {attempt + 1}/{self.max_retries})", flush=True)
                time.sleep(wait)
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id or f"call_{i}", tc.function.name, tc.function.arguments or "{}")
                 for i, tc in enumerate(msg.tool_calls or [])]
        usage = resp.usage
        return LLMResponse(msg.content, calls,
                           getattr(usage, "prompt_tokens", 0) or 0,
                           getattr(usage, "completion_tokens", 0) or 0, latency)


class MockLLM:
    """Replays task['plan'] (a list of tool calls) then answers task['answer']."""

    model = "mock"

    def __init__(self):
        self.task: dict | None = None
        self.step = 0

    def reset(self, task: dict):
        self.task, self.step = task, 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        plan = self.task.get("plan", []) if tools else []
        if self.step < len(plan):
            call = plan[self.step]
            self.step += 1
            return LLMResponse(None, [ToolCall(f"call_{self.step}", call["tool"], json.dumps(call["args"]))],
                               50, 20, 0.0)
        if not tools:  # baseline mode: the mock has no knowledge without tools
            return LLMResponse("FINAL ANSWER: unknown", [], 40, 5, 0.0)
        return LLMResponse(f"FINAL ANSWER: {self.task['answer']}", [], 50, 10, 0.0)
