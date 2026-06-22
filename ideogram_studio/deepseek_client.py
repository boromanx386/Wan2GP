from __future__ import annotations

from typing import Any

# https://api-docs.deepseek.com/quick_start/pricing
DEEPSEEK_MODELS = [
    ("V4 Flash — fast, cheap (recommended)", "deepseek-v4-flash"),
    ("V4 Pro — higher quality", "deepseek-v4-pro"),
    ("Legacy chat (→ V4 Flash, deprecated Jul 2026)", "deepseek-chat"),
    ("Legacy reasoner (→ V4 Flash thinking, deprecated Jul 2026)", "deepseek-reasoner"),
]

_V4_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}
_LEGACY_ALIASES = {"deepseek-chat", "deepseek-reasoner"}


def normalize_deepseek_model(model: str) -> str:
    return str(model or "deepseek-v4-flash").strip() or "deepseek-v4-flash"


def resolve_deepseek_chat_request(
    model: str,
    *,
    thinking_enabled: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Map UI/config model ids to DeepSeek API params.

    V4 models support thinking via extra_body.thinking.type = enabled|disabled.
    Legacy deepseek-chat / deepseek-reasoner route to V4 Flash until Jul 2026.
    """
    model = normalize_deepseek_model(model)
    extra_body: dict[str, Any] = {}

    if model == "deepseek-chat":
        api_model = "deepseek-v4-flash"
        extra_body["thinking"] = {"type": "disabled"}
    elif model == "deepseek-reasoner":
        api_model = "deepseek-v4-flash"
        extra_body["thinking"] = {"type": "enabled"}
    elif model in _V4_MODELS:
        api_model = model
        use_thinking = bool(thinking_enabled) if thinking_enabled is not None else False
        extra_body["thinking"] = {"type": "enabled" if use_thinking else "disabled"}
    else:
        api_model = model

    kwargs: dict[str, Any] = {}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return api_model, kwargs


def uses_thinking_mode(model: str, *, thinking_enabled: bool | None = None) -> bool:
    model = normalize_deepseek_model(model)
    if model == "deepseek-reasoner":
        return True
    if model == "deepseek-chat":
        return False
    if model in _V4_MODELS:
        return bool(thinking_enabled)
    return False


def extract_assistant_text(message) -> str:
    content = str(getattr(message, "content", None) or "").strip()
    if content:
        return content
    reasoning = str(getattr(message, "reasoning_content", None) or "").strip()
    if reasoning:
        start = reasoning.find("{")
        end = reasoning.rfind("}")
        if start >= 0 and end > start:
            return reasoning[start : end + 1]
    return content
