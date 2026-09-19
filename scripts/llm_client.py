"""Thin OpenAI-compatible chat client shared by discovery and curation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .config import Config
from .util import StageError, get_logger

log = get_logger("llm_client")


def chat_completion(
    messages: list[dict[str, str]],
    cfg: Config,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any] | None:
    """Send a chat completion request to the configured OpenAI-compatible endpoint."""
    c = cfg.curator
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("`requests` is required for LLM features (pip install requests)") from exc

    api_key = c.api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": c.model,
        "messages": messages,
        "temperature": temperature if temperature is not None else c.temperature,
        "max_tokens": max_tokens if max_tokens is not None else c.max_tokens,
    }

    try:
        resp = requests.post(
            f"{c.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=c.timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("LLM request failed: %s", exc)
        return None


def extract_message_text(completion: dict[str, Any] | None) -> str:
    """Return the assistant message text, falling back to reasoning fields."""
    if not completion or "choices" not in completion:
        return ""
    msg = completion["choices"][0].get("message", {})
    raw = msg.get("content") or ""
    if not raw.strip():
        raw = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return raw.strip()


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from a fenced or plain text response.

    Tries the first ```json fence, then the first top-level object. If that
    object is malformed (often because a reasoning model put explanatory text
    before its final JSON), it backtracks and tries the last object in the text.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        first = re.search(r"(\{.*\})", raw, re.DOTALL)
        if first:
            raw = first.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.debug("First JSON object parse failed; trying last object")

    # Reasoning models may emit the final JSON at the end of a long explanation.
    matches = list(re.finditer(r"(\{.*?\})", raw, re.DOTALL))
    for match in reversed(matches[-10:]):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    log.debug("Could not parse any JSON object from LLM response")
    return None
