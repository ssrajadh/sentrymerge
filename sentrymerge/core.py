"""Core SentryMerge logic: VLM detection, axis-direction ownership timeline,
and ffmpeg stitching.

Pipeline (see sentrymerge.cli for the wiring):

  1. Parse SentrySearch's last_search receipt (timestamp prefix + sister files).
  2. For each sister camera, ask Gemini Vision where the subject is visible
     -> per-camera time ranges with confidence.
  3. Build an ownership timeline ordered along the Tesla front-back axis,
     direction inferred from VLM timing. Each camera owns from its first
     detection until the next camera's first detection.
  4. ffmpeg-stitch the segments with a drawtext overlay.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


CAMERAS = ("front", "left_repeater", "right_repeater", "back")

# Position along the Tesla front-back axis. Side cameras share position 1
# because either side can come first depending on which side the object is on;
# the actual side ordering is decided by VLM detection times.
AXIS_POS = {"back": 0, "left_repeater": 1, "right_repeater": 1, "front": 2}

# Tesla SentryCam filename: YYYY-MM-DD_HH-MM-SS-<camera>.mp4
FNAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"-(front|back|left_repeater|right_repeater)\.mp4$"
)

VLM_MODEL_DEFAULT = "gemini-2.5-pro"


def parse_tesla_filename(path: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(timestamp_prefix, camera)`` for a Tesla clip path, else ``(None, None)``."""
    m = FNAME_RE.match(os.path.basename(path))
    return (m.group(1), m.group(2)) if m else (None, None)


def fmt_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def find_sister_files(timestamp: str, search_dirs: Iterable[Path]) -> dict[str, Path]:
    """Return ``{camera: Path}`` for each sister-camera file that exists on disk
    in any of *search_dirs*. First hit per camera wins."""
    sisters: dict[str, Path] = {}
    for d in search_dirs:
        for cam in CAMERAS:
            f = d / f"{timestamp}-{cam}.mp4"
            if cam not in sisters and f.exists():
                sisters[cam] = f
    return sisters


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------


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


def vlm_visibility_ranges(
    client,
    video_path: Path,
    *,
    query: Optional[str] = None,
    image_path: Optional[Path] = None,
    model: str = VLM_MODEL_DEFAULT,
    verbose: bool = False,
) -> list[dict]:
    """Ask the VLM for time ranges where the subject is visible.

    Provide exactly one of *query* (text) or *image_path* (reference image).
    Returns a list of ``{"start", "end", "confidence", "note"}`` dicts.
    """
    if (query is None) == (image_path is None):
        raise ValueError("exactly one of query, image_path must be set")

    from google.genai import types

    with open(video_path, "rb") as f:
        video_bytes = f.read()
    parts: list = [types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")]

    if query is not None:
        prompt = _TEXT_PROMPT_TEMPLATE.format(query=query)
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        # Best-effort mime detection; Gemini also accepts image/jpeg by default.
        suffix = image_path.suffix.lower()
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
        prompt = _IMAGE_PROMPT

    parts.append(prompt)

    if verbose:
        print(f"    VLM call: {video_path.name} "
              f"({len(video_bytes) / 1024:.0f}KB)", file=sys.stderr)

    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    data = json.loads(resp.text)
    # The model occasionally drops the {"ranges": ...} wrapper and returns the
    # bare list, despite the prompt — accept both shapes.
    if isinstance(data, dict):
        ranges = data.get("ranges", [])
    elif isinstance(data, list):
        ranges = data
    else:
        ranges = []
    return [r for r in ranges if isinstance(r, dict) and "start" in r
            and "end" in r and "confidence" in r]


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    camera: str
    confidence: float


@dataclass(frozen=True)
class Timeline:
    segments: list[Segment]
    direction: str  # "back→front", "front→back", "side-only", or "single-camera"
    warnings: list[str]


def build_ownership_timeline(
    per_cam_ranges: dict[str, list[dict]],
    total_duration: float,
    conf_threshold: float = 0.3,
) -> Timeline:
    """Order cameras along the front-back axis in the direction implied by VLM
    timing. Each camera owns from its earliest detection until the next
    camera's earliest detection (in the chosen physical order). Cameras out of
    order in time get squeezed to zero duration and dropped.

    Edge cases handled:
      - Only side cameras detect: fall back to pure temporal order.
      - Only one camera detects: emit a single segment and warn.
      - Only front + back detect (no side cam): warn but keep both — the
        physical impossibility is logged, not silently corrected.
    """
    cam_first: dict[str, dict] = {}
    for cam, ranges in per_cam_ranges.items():
        valid = [r for r in ranges if r["confidence"] >= conf_threshold]
        if not valid:
            continue
        cam_first[cam] = min(valid, key=lambda r: r["start"])

    warnings: list[str] = []
    if not cam_first:
        return Timeline(segments=[], direction="none", warnings=warnings)

    if len(cam_first) == 1:
        cam, r = next(iter(cam_first.items()))
        warnings.append(
            f"only {cam} detected the subject (no cross-camera handoff)"
        )
        end = min(r["end"], total_duration)
        seg = Segment(start=r["start"], end=end, camera=cam,
                      confidence=r["confidence"])
        return Timeline(segments=[seg] if end > r["start"] else [],
                        direction="single-camera", warnings=warnings)

    cams = list(cam_first)
    if all(AXIS_POS[c] == 1 for c in cams):
        # Only side cameras saw it. Direction is undefined on the front-back
        # axis; use temporal order.
        chain = sorted(cams, key=lambda c: cam_first[c]["start"])
        direction = "side-only"
    else:
        by_time = sorted(cams, key=lambda c: cam_first[c]["start"])
        earliest, latest = by_time[0], by_time[-1]
        if AXIS_POS[earliest] <= AXIS_POS[latest]:
            chain = sorted(
                cams, key=lambda c: (AXIS_POS[c], cam_first[c]["start"])
            )
            direction = "back→front"
        else:
            chain = sorted(
                cams, key=lambda c: (-AXIS_POS[c], cam_first[c]["start"])
            )
            direction = "front→back"

    if set(cams) == {"back", "front"}:
        warnings.append(
            "only back and front detected — no side-camera corroboration; "
            "object cannot physically traverse without passing a side camera, "
            "so one detection is likely a VLM hallucination"
        )

    segments: list[Segment] = []
    for i, cam in enumerate(chain):
        start = cam_first[cam]["start"]
        if i + 1 < len(chain):
            end = cam_first[chain[i + 1]]["start"]
        else:
            end = cam_first[cam]["end"]
        end = min(end, total_duration)
        if end > start:
            segments.append(Segment(
                start=start, end=end, camera=cam,
                confidence=cam_first[cam]["confidence"],
            ))

    return Timeline(segments=segments, direction=direction, warnings=warnings)


# ---------------------------------------------------------------------------
# ffmpeg stitch
# ---------------------------------------------------------------------------


def get_video_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.strip())


def _drawtext_safe(s: str) -> str:
    return s.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def build_stitch_command(
    segments: list[Segment],
    sisters: dict[str, Path],
    output: Path,
) -> list[str]:
    """Return an ``ffmpeg`` command that concatenates *segments* into *output*
    with a drawtext overlay per segment showing camera + range + VLM conf."""
    if not segments:
        raise ValueError("cannot build stitch command from empty segments")

    inputs: list[str] = []
    filters: list[str] = []
    for i, seg in enumerate(segments):
        path = sisters[seg.camera]
        dur = max(0.1, seg.end - seg.start)
        inputs += ["-ss", f"{seg.start}", "-t", f"{dur}", "-i", str(path)]
        label = (f"{seg.camera}   {fmt_mmss(seg.start)}-{fmt_mmss(seg.end)}   "
                 f"VLM conf={seg.confidence:.2f}")
        safe = _drawtext_safe(label)
        filters.append(
            f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,"
            f"drawtext=text='{safe}':fontcolor=white:fontsize=32:"
            f"box=1:boxcolor=black@0.75:boxborderw=12:x=22:y=22[v{i}]"
        )
    concat_in = "".join(f"[v{i}]" for i in range(len(segments)))
    fc = (";".join(filters)
          + f";{concat_in}concat=n={len(segments)}:v=1:a=0[out]")
    return ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", fc, "-map", "[out]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        str(output),
    ]
