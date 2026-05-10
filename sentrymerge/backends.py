"""VLM backends for sub-second visibility detection on a 30s dashcam clip.

Each backend takes a video path + (text query OR reference image) and returns
a list of ``{"start", "end", "confidence", "note"}`` dicts.

Three backends ship today:
  - GeminiBackend (cloud, native MP4)        — default if GEMINI_API_KEY set
  - OpenAIBackend (cloud, frame-sampled)     — default if OPENAI_API_KEY set
  - QwenBackend   (local, Qwen2.5-VL-7B)     — fallback, requires [local] extra

`resolve_backend(name=None)` picks one by name or by env-var auto-detection.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Protocol


GEMINI_DEFAULT_MODEL = "gemini-3-pro"
OPENAI_DEFAULT_MODEL = "gpt-5"
QWEN_DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

BACKEND_NAMES = ("gemini", "openai", "qwen")


_TEXT_PROMPT_TEMPLATE = (
    'You are watching a Tesla dashcam clip. The user is searching for: '
    '"{query}".\n\n'
    'Identify all time ranges (seconds from start of clip) where "{query}" is '
    'clearly visible or occurring.\n\n'
    'Return ONLY valid JSON of the form:\n'
    '{{"ranges": [{{"start": <float>, "end": <float>, '
    '"confidence": <0-1 float>, "note": "<short>"}}]}}\n\n'
    'Rules:\n'
    '- If never clearly visible, return {{"ranges": []}}.\n'
    '- "confidence" reflects how well the subject matches the query.\n'
    '- Tight intervals: only seconds where the subject is in frame and '
    'identifiable.\n'
)

_IMAGE_PROMPT = (
    'You are watching a Tesla dashcam clip. The user is searching for the '
    'subject shown in the attached reference image. Identify all time ranges '
    '(seconds from start of clip) where that subject is clearly visible.\n\n'
    'Return ONLY valid JSON of the form:\n'
    '{"ranges": [{"start": <float>, "end": <float>, '
    '"confidence": <0-1 float>, "note": "<short>"}]}\n\n'
    'Rules:\n'
    '- If never clearly visible, return {"ranges": []}.\n'
    '- "confidence" reflects how well the subject matches the reference image.\n'
    '- Tight intervals: only seconds where the subject is in frame and '
    'identifiable.\n'
)


def parse_ranges_json(text: str) -> list[dict]:
    """Parse a VLM JSON response, tolerant to common shape variance.

    Accepts ``{"ranges": [...]}`` or a bare list. Strips ```json fences if
    present. Filters entries missing ``start``/``end``/``confidence``.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        ranges = data.get("ranges", [])
    elif isinstance(data, list):
        ranges = data
    else:
        return []
    return [r for r in ranges if isinstance(r, dict)
            and "start" in r and "end" in r and "confidence" in r]


class VLMBackend(Protocol):
    """A VLM that returns visibility ranges for a query within a video clip."""

    name: str
    model: str

    def detect(
        self,
        video_path: Path,
        *,
        query: Optional[str] = None,
        image_path: Optional[Path] = None,
        verbose: bool = False,
    ) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiBackend:
    name = "gemini"

    def __init__(self, *, model: str = GEMINI_DEFAULT_MODEL,
                 api_key: Optional[str] = None):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Gemini backend requires `google-genai` (installed by default)."
            ) from e
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in ~/.sentrysearch/.env "
                "or the environment, or pass --vlm-backend openai/qwen."
            )
        self.model = model
        self._client = genai.Client(api_key=key)

    def detect(self, video_path, *, query=None, image_path=None, verbose=False):
        if (query is None) == (image_path is None):
            raise ValueError("exactly one of query, image_path must be set")
        from google.genai import types

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        parts: list = [types.Part.from_bytes(
            data=video_bytes, mime_type="video/mp4")]
        if query is not None:
            prompt = _TEXT_PROMPT_TEMPLATE.format(query=query)
        else:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            parts.append(types.Part.from_bytes(
                data=image_bytes, mime_type=_image_mime(image_path)))
            prompt = _IMAGE_PROMPT
        parts.append(prompt)

        if verbose:
            print(f"    Gemini call: {video_path.name} "
                  f"({len(video_bytes) / 1024:.0f}KB)", file=sys.stderr)

        resp = self._client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        return parse_ranges_json(resp.text)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIBackend:
    """OpenAI Chat Completions with frame sampling.

    GPT-4o / GPT-5 don't accept native MP4 — we sample frames at 1 fps via
    ffmpeg (matching Gemini's effective sampling rate) and pass each frame
    annotated with its timestamp so the model can return time ranges.
    """
    name = "openai"

    def __init__(self, *, model: str = OPENAI_DEFAULT_MODEL,
                 api_key: Optional[str] = None,
                 fps: float = 1.0, max_frames: int = 32):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI backend requires `openai`. Install with "
                "`uv sync --extra openai`."
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in ~/.sentrysearch/.env "
                "or the environment."
            )
        self.model = model
        self._fps = fps
        self._max_frames = max_frames
        self._client = OpenAI(api_key=key)

    def detect(self, video_path, *, query=None, image_path=None, verbose=False):
        if (query is None) == (image_path is None):
            raise ValueError("exactly one of query, image_path must be set")

        frames = _sample_frames_jpeg(
            video_path, fps=self._fps, max_frames=self._max_frames,
        )
        if verbose:
            print(f"    OpenAI call: {video_path.name} "
                  f"({len(frames)} frames @ {self._fps}fps)", file=sys.stderr)

        content: list = []
        if query is not None:
            content.append({"type": "text",
                            "text": _TEXT_PROMPT_TEMPLATE.format(query=query)})
        else:
            content.append({"type": "text", "text": _IMAGE_PROMPT})
            ref_b64 = base64.b64encode(image_path.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_image_mime(image_path)};base64,{ref_b64}",
                },
            })
        for i, jpg in enumerate(frames):
            t = i / self._fps
            content.append({"type": "text", "text": f"[frame at t={t:.1f}s]"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,"
                           f"{base64.b64encode(jpg).decode()}",
                },
            })

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return parse_ranges_json(resp.choices[0].message.content or "")


# ---------------------------------------------------------------------------
# Qwen2.5-VL local
# ---------------------------------------------------------------------------


class QwenBackend:
    """Qwen2.5-VL-Instruct local backend.

    Lazy-loaded: the model + processor are constructed on first ``detect()``
    call so importing sentrymerge is cheap on machines without the [local]
    extra. Subsequent calls reuse the cached model (class-level singleton).
    """
    name = "qwen"

    _model = None
    _processor = None
    _loaded_id: Optional[str] = None

    def __init__(self, *, model: str = QWEN_DEFAULT_MODEL,
                 quantize: Optional[bool] = None):
        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
                Qwen2_5_VLForConditionalGeneration, AutoProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "Qwen local backend requires the [local] extra. Install with "
                "`uv sync --extra local` "
                "(or `uv sync --extra local-quantized` on NVIDIA <16GB VRAM)."
            ) from e
        self.model = model
        self._quantize = quantize

    def _load(self):
        if QwenBackend._model is not None and QwenBackend._loaded_id == self.model:
            return
        import torch
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor,
        )

        kwargs: dict = {}
        if self._quantize:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError(
                    "--quantize requires the [local-quantized] extra "
                    "(bitsandbytes). NVIDIA-only."
                ) from e
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "mps"
        else:
            # CPU fallback — slow but functional
            kwargs["torch_dtype"] = torch.float32

        QwenBackend._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model, **kwargs,
        )
        QwenBackend._processor = AutoProcessor.from_pretrained(self.model)
        QwenBackend._loaded_id = self.model

    def detect(self, video_path, *, query=None, image_path=None, verbose=False):
        if (query is None) == (image_path is None):
            raise ValueError("exactly one of query, image_path must be set")

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "Qwen local backend requires `qwen-vl-utils`. Install with "
                "`uv sync --extra local`."
            ) from e
        import torch

        self._load()

        if query is not None:
            content = [
                {"type": "video", "video": str(video_path),
                 "fps": 1.0, "max_frames": 32},
                {"type": "text",
                 "text": _TEXT_PROMPT_TEMPLATE.format(query=query)},
            ]
        else:
            content = [
                {"type": "image", "image": str(image_path)},
                {"type": "video", "video": str(video_path),
                 "fps": 1.0, "max_frames": 32},
                {"type": "text", "text": _IMAGE_PROMPT},
            ]
        messages = [{"role": "user", "content": content}]

        if verbose:
            print(f"    Qwen call: {video_path.name}", file=sys.stderr)

        text = QwenBackend._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = QwenBackend._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(QwenBackend._model.device)

        with torch.no_grad():
            generated = QwenBackend._model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        out = QwenBackend._processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_ranges_json(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_mime(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")


def _sample_frames_jpeg(video_path: Path, *, fps: float,
                        max_frames: int) -> list[bytes]:
    """Extract JPEG frames at *fps* via ffmpeg, capped at *max_frames*."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(video_path),
             "-vf", f"fps={fps}",
             "-frames:v", str(max_frames),
             "-q:v", "3",
             str(td_path / "frame_%04d.jpg")],
            check=True,
        )
        return [p.read_bytes() for p in sorted(td_path.glob("frame_*.jpg"))]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_backend(
    name: Optional[str] = None,
    *,
    model: Optional[str] = None,
) -> VLMBackend:
    """Pick a VLM backend by explicit name, else auto-detect from env vars.

    Auto-detect order: GEMINI_API_KEY → gemini, OPENAI_API_KEY → openai,
    otherwise → qwen (local). An explicit *name* overrides auto-detection.
    *model* overrides the backend's default model id.
    """
    if name is None:
        if os.environ.get("GEMINI_API_KEY"):
            name = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "qwen"
    name = name.lower()
    if name == "gemini":
        return GeminiBackend(model=model or GEMINI_DEFAULT_MODEL)
    if name == "openai":
        return OpenAIBackend(model=model or OPENAI_DEFAULT_MODEL)
    if name == "qwen":
        return QwenBackend(model=model or QWEN_DEFAULT_MODEL)
    raise ValueError(
        f"unknown backend: {name!r} (expected one of {BACKEND_NAMES})"
    )
