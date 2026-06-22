from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .deepseek_client import DEEPSEEK_MODELS, normalize_deepseek_model

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_PATH = PACKAGE_DIR / "config.json"

IDEOGRAM_MODELS = [
    ("Ideogram v4 FP8", "ideogram4"),
    ("Ideogram v4 NF4", "ideogram4_nf4"),
    ("Ideogram v4 TurboTime", "ideogram4_turbotime"),
]

PRESETS = [
    ("Default 20", "V4_DEFAULT_20"),
    ("Quality 48", "V4_QUALITY_48"),
    ("Turbo 12", "V4_TURBO_12"),
]

RESOLUTIONS = [
    "1024x1024",
    "1280x720",
    "720x1280",
    "1536x1024",
    "1024x1536",
    "1920x1080",
]

ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "3:2", "2:3"]


@dataclass
class StudioConfig:
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_thinking: bool = False
    deepseek_temperature: float = 0.5
    deepseek_max_tokens: int = 2048
    model_type: str = "ideogram4"
    model_mode: str = "V4_DEFAULT_20"
    resolution: str = "1024x1024"
    aspect_ratio: str = "1:1"
    wgp_profile: str = ""
    wgp_cli_args: list[str] = field(default_factory=list)
    output_dir: str = ""

    @classmethod
    def load(cls) -> StudioConfig:
        data: dict = {}
        if CONFIG_PATH.is_file():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        cfg = cls(
            deepseek_api_key=str(data.get("deepseek_api_key") or os.environ.get("DEEPSEEK_API_KEY") or ""),
            deepseek_base_url=str(data.get("deepseek_base_url") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"),
            deepseek_model=normalize_deepseek_model(
                str(data.get("deepseek_model") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash")
            ),
            deepseek_thinking=bool(data.get("deepseek_thinking", False)),
            deepseek_temperature=float(data.get("deepseek_temperature", 0.5)),
            deepseek_max_tokens=int(data.get("deepseek_max_tokens", 2048)),
            model_type=str(data.get("model_type") or "ideogram4"),
            model_mode=str(data.get("model_mode") or "V4_DEFAULT_20"),
            resolution=str(data.get("resolution") or "1024x1024"),
            aspect_ratio=str(data.get("aspect_ratio") or "1:1"),
            wgp_profile=str(data.get("wgp_profile") or ""),
            wgp_cli_args=list(data.get("wgp_cli_args") or []),
            output_dir=str(data.get("output_dir") or ""),
        )
        return cfg

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def cli_args(self) -> list[str]:
        args = list(self.wgp_cli_args)
        if self.wgp_profile and not any(arg == "--profile" for arg in args):
            args.extend(["--profile", self.wgp_profile])
        return args

    def public_dict(self) -> dict:
        payload = asdict(self)
        key = payload.get("deepseek_api_key") or ""
        payload["deepseek_api_key_set"] = bool(key)
        payload["deepseek_api_key"] = f"{key[:4]}…{key[-4:]}" if len(key) > 10 else ("set" if key else "")
        return payload
