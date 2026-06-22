from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from .backend import IdeogramBackend
from .config import (
    ASPECT_RATIOS,
    CONFIG_PATH,
    DEEPSEEK_MODELS,
    IDEOGRAM_MODELS,
    PRESETS,
    REPO_ROOT,
    RESOLUTIONS,
    StudioConfig,
)
from .magic_prompt import run_magic_prompt
from .prompt_sanitize import sanitize_ideogram_prompt_json

UI_DIR = Path(__file__).resolve().parent / "ui"


class StudioApi:
    def __init__(self, config: StudioConfig, *, root: Path | None = None) -> None:
        self._config = config
        self._backend = IdeogramBackend(config, root=root)
        self._warmup_started = False

    def _maybe_warmup(self) -> None:
        if self._warmup_started:
            return
        self._warmup_started = True

        def worker() -> None:
            try:
                self._backend.warm_up()
            except Exception as exc:
                print(f"[Ideogram Studio] WanGP warmup failed: {exc}", file=sys.stderr)

        threading.Thread(target=worker, daemon=True, name="ideogram-warmup").start()

    def get_bootstrap(self) -> dict:
        self._maybe_warmup()
        return {
            "models": [{"label": label, "value": value} for label, value in IDEOGRAM_MODELS],
            "presets": [{"label": label, "value": value} for label, value in PRESETS],
            "resolutions": RESOLUTIONS,
            "aspect_ratios": ASPECT_RATIOS,
            "deepseek_models": [{"label": label, "value": value} for label, value in DEEPSEEK_MODELS],
            "config": self._config.public_dict(),
        }

    def save_config(self, payload: dict) -> dict:
        data = payload if isinstance(payload, dict) else {}
        for key, value in data.items():
            if not hasattr(self._config, key):
                continue
            if key == "deepseek_api_key" and (not value or "…" in str(value)):
                continue
            setattr(self._config, key, value)
        self._config.save()
        self._backend.update_config(self._config)
        return {"ok": True, "config": self._config.public_dict()}

    def magic_prompt(self, idea: str, aspect_ratio: str | None = None) -> dict:
        try:
            result = run_magic_prompt(
                idea=idea,
                aspect_ratio=aspect_ratio or self._config.aspect_ratio,
                api_key=self._config.deepseek_api_key,
                base_url=self._config.deepseek_base_url,
                model=self._config.deepseek_model,
                temperature=self._config.deepseek_temperature,
                max_tokens=self._config.deepseek_max_tokens,
                thinking_enabled=self._config.deepseek_thinking,
            )
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def start_generate(
        self,
        prompt: str,
        resolution: str | None = None,
        model_type: str | None = None,
        model_mode: str | None = None,
        seed: int = -1,
    ) -> dict:
        try:
            job_id = self._backend.start_generate(
                prompt=prompt,
                resolution=resolution,
                model_type=model_type,
                model_mode=model_mode,
                seed=int(seed),
            )
            return {"ok": True, "job_id": job_id}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def poll_job(self, job_id: str) -> dict:
        return self._backend.poll_job(job_id)

    def cancel_job(self, job_id: str) -> dict:
        return self._backend.cancel_job(job_id)

    def sanitize_prompt(self, prompt: str) -> dict:
        try:
            compact, pretty, notes = sanitize_ideogram_prompt_json(str(prompt or "").strip())
            return {"ok": True, "json": compact, "pretty": pretty, "sanitize_notes": notes}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def build_html() -> str:
    from models.ideogram4.prompt_helper import get_prompt_helper_css, get_prompt_helper_javascript

    template = (UI_DIR / "index.html").read_text(encoding="utf-8")
    template = template.replace("{{PROMPT_HELPER_CSS}}", get_prompt_helper_css())
    template = template.replace("{{PROMPT_HELPER_JS}}", get_prompt_helper_javascript())
    template = template.replace(
        '<link rel="stylesheet" href="style.css" />',
        f"<style>{(UI_DIR / 'style.css').read_text(encoding='utf-8')}</style>",
    )
    app_js = (UI_DIR / "app.js").read_text(encoding="utf-8")
    helper_patch_js = (UI_DIR / "helper_patch.js").read_text(encoding="utf-8")
    template = template.replace('<script src="app.js"></script>', f"<script>{helper_patch_js}</script>\n<script>{app_js}</script>")
    return template


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ideogram Studio — desktop UI for Ideogram 4 via WanGP.")
    parser.add_argument("--root", type=str, default=str(REPO_ROOT), help="WanGP repository root.")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import webview
    except ImportError as exc:
        raise SystemExit("Install pywebview: pip install pywebview") from exc

    config = StudioConfig.load()
    if not config.deepseek_api_key:
        print(
            "Warning: DEEPSEEK_API_KEY is not set. Magic Prompt will fail until you add it to "
            f"{CONFIG_PATH} or the environment.",
            file=sys.stderr,
        )

    api = StudioApi(config, root=root)
    window = webview.create_window(
        "Ideogram Studio",
        html=build_html(),
        js_api=api,
        width=args.width,
        height=args.height,
        min_size=(1100, 700),
        background_color="#07070c",
    )
    webview.start(debug=False)
