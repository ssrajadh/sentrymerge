"""Core SentryMerge logic: axis-direction ownership timeline and ffmpeg
stitching. VLM detection itself lives in :mod:`sentrymerge.backends`.

Pipeline (see sentrymerge.cli for the wiring):

  1. Parse SentrySearch's last_search receipt (timestamp prefix + sister files).
  2. For each sister camera, call the configured VLM backend for visibility
     ranges (Gemini / OpenAI / Qwen-local).
  3. Build an ownership timeline ordered along the Tesla front-back axis,
     direction inferred from VLM timing. Each camera owns from its first
     detection until the next camera's first detection.
  4. ffmpeg-stitch the segments with a drawtext overlay.
"""
from __future__ import annotations

import os
import re
import subprocess
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
# Multi-run majority voting (mitigates VLM non-determinism)
# ---------------------------------------------------------------------------


def merge_votes(votes: list[list[dict]], *, dt: float = 0.1,
                min_votes: Optional[int] = None) -> list[dict]:
    """Combine *k* VLM runs into a single range list via per-slot majority vote.

    A slot of duration *dt* is considered visible iff at least *min_votes*
    runs detected something covering it. ``min_votes`` defaults to the
    majority (``ceil(k/2)``). The merged ranges are reconstructed from
    contiguous voted-in slots; confidence is averaged over all votes that
    covered the slot.
    """
    if not votes:
        return []
    k = len(votes)
    if k == 1:
        return list(votes[0])
    if min_votes is None:
        min_votes = (k + 1) // 2

    max_end = 0.0
    for run in votes:
        for r in run:
            max_end = max(max_end, r["end"])
    if max_end <= 0:
        return []

    n_slots = int(max_end / dt) + 1
    counts = [0] * n_slots
    conf_sums = [0.0] * n_slots

    for run in votes:
        # Within a single run, a slot only counts once even if multiple ranges
        # cover it (avoids inflating confidence by overlapping ranges).
        seen: set[int] = set()
        for r in run:
            i0 = max(0, int(r["start"] / dt))
            i1 = min(n_slots, int(r["end"] / dt) + 1)
            for i in range(i0, i1):
                if i not in seen:
                    seen.add(i)
                    counts[i] += 1
                    conf_sums[i] += r["confidence"]

    merged: list[dict] = []
    in_range = False
    start_t = 0.0
    accum_conf = 0.0
    accum_n = 0
    for i in range(n_slots):
        if counts[i] >= min_votes:
            if not in_range:
                start_t = i * dt
                accum_conf = 0.0
                accum_n = 0
                in_range = True
            accum_conf += conf_sums[i] / counts[i]
            accum_n += 1
        else:
            if in_range:
                merged.append({
                    "start": start_t,
                    "end": i * dt,
                    "confidence": accum_conf / max(1, accum_n),
                    "note": "",
                })
                in_range = False
    if in_range:
        merged.append({
            "start": start_t,
            "end": n_slots * dt,
            "confidence": accum_conf / max(1, accum_n),
            "note": "",
        })
    return merged


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
