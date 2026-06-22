from __future__ import annotations

import base64
import io
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, StudioConfig


@dataclass
class JobSnapshot:
    job_id: str
    phase: str = "queued"
    status: str = "Waiting…"
    progress: int = 0
    preview_b64: str = ""
    done: bool = False
    success: bool = False
    output_paths: list[str] = field(default_factory=list)
    error: str = ""


class IdeogramBackend:
    def __init__(self, config: StudioConfig, *, root: Path | None = None) -> None:
        self._config = config
        self._root = Path(root or REPO_ROOT)
        self._session = None
        self._session_lock = threading.Lock()
        self._jobs: dict[str, JobSnapshot] = {}
        self._jobs_lock = threading.Lock()
        self._active_jobs: dict[str, Any] = {}

    def update_config(self, config: StudioConfig) -> None:
        self._config = config

    def _ensure_session(self):
        with self._session_lock:
            if self._session is not None:
                return self._session
            from shared.api import init

            kwargs: dict[str, Any] = {
                "root": self._root,
                "cli_args": self._config.cli_args(),
                "console_output": True,
            }
            if self._config.output_dir:
                kwargs["output_dir"] = self._config.output_dir
            self._session = init(**kwargs)
            return self._session

    def warm_up(self) -> str:
        self._ensure_session()
        return "WanGP session ready."

    def _set_job(self, snapshot: JobSnapshot) -> None:
        with self._jobs_lock:
            self._jobs[snapshot.job_id] = snapshot

    def poll_job(self, job_id: str) -> dict[str, Any]:
        with self._jobs_lock:
            snapshot = self._jobs.get(job_id)
        if snapshot is None:
            return {"error": f"Unknown job: {job_id}"}
        return {
            "job_id": snapshot.job_id,
            "phase": snapshot.phase,
            "status": snapshot.status,
            "progress": snapshot.progress,
            "preview_b64": snapshot.preview_b64,
            "done": snapshot.done,
            "success": snapshot.success,
            "output_paths": snapshot.output_paths,
            "error": snapshot.error,
        }

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._jobs_lock:
            snapshot = self._jobs.get(job_id)
        if snapshot is None:
            return {"ok": False, "error": f"Unknown job: {job_id}"}
        session = None
        with self._jobs_lock:
            session_job = self._active_jobs.get(job_id)
        if session_job is not None:
            try:
                session_job.cancel()
            except Exception:
                pass
        return {"ok": True}

    @staticmethod
    def _preview_to_b64(image) -> str:
        if image is None:
            return ""
        buffer = io.BytesIO()
        if getattr(image, "mode", "") in ("RGBA", "LA", "P"):
            image.save(buffer, format="PNG")
        else:
            image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _resolve_output_path(path: str | Path, root: Path) -> Path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        rooted = root / candidate
        if rooted.is_file():
            return rooted
        return candidate

    @staticmethod
    def _final_output_b64(result, root: Path) -> str:
        from PIL import Image

        for raw_path in list(getattr(result, "generated_files", None) or []):
            path = IdeogramBackend._resolve_output_path(raw_path, root)
            if not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    return IdeogramBackend._preview_to_b64(image.copy())
            except Exception as exc:
                print(f"[Ideogram Studio] Failed to read output {path}: {exc}")

        for artifact in getattr(result, "artifacts", None) or ():
            path = getattr(artifact, "path", None)
            if path:
                resolved = IdeogramBackend._resolve_output_path(path, root)
                if resolved.is_file():
                    try:
                        with Image.open(resolved) as image:
                            return IdeogramBackend._preview_to_b64(image.copy())
                    except Exception as exc:
                        print(f"[Ideogram Studio] Failed to read artifact {resolved}: {exc}")
            tensor = getattr(artifact, "video_tensor_uint8", None)
            if tensor is not None:
                try:
                    import numpy as np

                    frame = np.asarray(tensor)
                    if frame.ndim == 4:
                        frame = frame[:, 0]
                    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                        frame = np.transpose(frame, (1, 2, 0))
                    return IdeogramBackend._preview_to_b64(Image.fromarray(frame))
                except Exception:
                    pass
        return ""

    def start_generate(
        self,
        *,
        prompt: str,
        resolution: str | None = None,
        model_type: str | None = None,
        model_mode: str | None = None,
        seed: int = -1,
    ) -> str:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt JSON is empty.")
        from ideogram_studio.prompt_sanitize import prepare_prompt_for_generation

        try:
            prompt, prep_notes = prepare_prompt_for_generation(prompt)
            if prep_notes:
                print("[Ideogram Studio] Prompt prep:", "; ".join(prep_notes))
        except Exception as exc:
            raise ValueError(f"Invalid prompt JSON: {exc}") from exc

        job_id = uuid.uuid4().hex[:12]
        snapshot = JobSnapshot(job_id=job_id)
        self._set_job(snapshot)

        def worker() -> None:
            try:
                session = self._ensure_session()
                model = model_type or self._config.model_type
                settings = session.get_default_settings(model)
                settings.update(
                    {
                        "model_type": model,
                        "prompt": prompt,
                        "resolution": resolution or self._config.resolution,
                        "image_mode": 1,
                        "model_mode": model_mode or self._config.model_mode,
                        "seed": int(seed),
                    }
                )
                snapshot.phase = "loading"
                snapshot.status = "Starting generation…"
                self._set_job(snapshot)

                job = session.submit_task(settings)
                with self._jobs_lock:
                    self._active_jobs[job_id] = job
                for event in job.events.iter(timeout=0.25):
                    if event.kind == "progress":
                        data = event.data
                        snapshot.phase = str(getattr(data, "phase", "") or "generating")
                        snapshot.status = str(getattr(data, "status", "") or snapshot.phase)
                        snapshot.progress = int(getattr(data, "progress", 0) or 0)
                        self._set_job(snapshot)
                    elif event.kind == "preview":
                        data = event.data
                        snapshot.preview_b64 = self._preview_to_b64(getattr(data, "image", None))
                        snapshot.status = str(getattr(data, "status", "") or snapshot.status)
                        snapshot.progress = int(getattr(data, "progress", snapshot.progress) or snapshot.progress)
                        self._set_job(snapshot)

                result = job.result()
                snapshot.done = True
                if result.success:
                    snapshot.success = True
                    snapshot.phase = "done"
                    snapshot.status = "Complete"
                    snapshot.progress = 100
                    snapshot.output_paths = list(result.generated_files or [])
                    final_b64 = self._final_output_b64(result, self._root)
                    if final_b64:
                        snapshot.preview_b64 = final_b64
                    elif not snapshot.preview_b64:
                        print("[Ideogram Studio] Warning: could not load final output image for preview.")
                else:
                    snapshot.success = False
                    snapshot.phase = "error"
                    messages = [str(err.message) for err in (result.errors or []) if str(err.message).strip()]
                    snapshot.error = messages[0] if messages else "Generation failed."
                    snapshot.status = snapshot.error
            except Exception as exc:
                snapshot.done = True
                snapshot.success = False
                snapshot.phase = "error"
                snapshot.error = str(exc)
                snapshot.status = snapshot.error
                print(traceback.format_exc())
            finally:
                with self._jobs_lock:
                    self._active_jobs.pop(job_id, None)
            self._set_job(snapshot)

        thread = threading.Thread(target=worker, daemon=True, name=f"ideogram-gen-{job_id}")
        thread.start()
        return job_id
