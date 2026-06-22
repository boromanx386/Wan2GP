from __future__ import annotations

import json
import re
from typing import Any

from models.ideogram4.prompt_enhancer import IDEOGRAM4_PROMPT_ENHANCER

from .deepseek_client import extract_assistant_text, resolve_deepseek_chat_request, uses_thinking_mode
from .prompt_sanitize import sanitize_ideogram_prompt_json

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_json_text(raw: str) -> str:
    text = str(raw or "").strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("Magic Prompt response did not contain a JSON object.")
    end = text.rfind("}")
    if end <= start:
        raise ValueError("Magic Prompt response JSON object is incomplete.")
    return text[start : end + 1]


def validate_json_prompt(json_text: str) -> str:
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError("Magic Prompt JSON root must be an object.")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_user_message(idea: str, aspect_ratio: str) -> str:
    idea = str(idea or "").strip()
    aspect_ratio = str(aspect_ratio or "1:1").strip() or "1:1"
    if not idea:
        raise ValueError("Enter a short idea before running Magic Prompt.")
    return f"{idea}\n\nTarget aspect ratio: {aspect_ratio}"


def run_magic_prompt(
    *,
    idea: str,
    aspect_ratio: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-v4-flash",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    thinking_enabled: bool = False,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("DeepSeek API key is missing. Set DEEPSEEK_API_KEY or edit ideogram_studio/config.json.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK: pip install openai") from exc

    api_model, request_kwargs = resolve_deepseek_chat_request(model, thinking_enabled=thinking_enabled)
    create_kwargs: dict[str, Any] = {
        "model": api_model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": IDEOGRAM4_PROMPT_ENHANCER},
            {"role": "user", "content": build_user_message(idea, aspect_ratio)},
        ],
        **request_kwargs,
    }
    if not uses_thinking_mode(model, thinking_enabled=thinking_enabled):
        create_kwargs["temperature"] = temperature

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(**create_kwargs)
    raw = extract_assistant_text(response.choices[0].message)
    json_text = validate_json_prompt(extract_json_text(raw))
    json_text, pretty, sanitize_notes = sanitize_ideogram_prompt_json(json_text)
    return {
        "json": json_text,
        "pretty": pretty,
        "raw": raw,
        "model": api_model,
        "thinking": uses_thinking_mode(model, thinking_enabled=thinking_enabled),
        "sanitize_notes": sanitize_notes,
    }
