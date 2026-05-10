# SentryMerge

Stitch a single cross-camera video of one event from a [SentrySearch](https://github.com/ssrajadh/sentrysearch) result. Frame-accurate handoffs between `back`, `left_repeater`, `right_repeater`, and `front` Tesla dashcam feeds, driven by a VLM visibility pass.

## Table of Contents

- [How it works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Stitch from last search](#stitch-from-last-search)
  - [Stitch from a fresh query](#stitch-from-a-fresh-query)
  - [Pick a specific event](#pick-a-specific-event)
  - [Reduce VLM wobble](#reduce-vlm-wobble-with---vlm-votes)
  - [Verbose mode](#verbose-mode)
- [Why this exists](#why-this-exists)
- [Direction labels](#direction-labels)
- [Cost](#cost)
- [Limitations & Future Work](#limitations--future-work)
- [Compatibility](#compatibility)
- [Requirements](#requirements)

## How it works

SentrySearch hits land at ~30s chunk granularity — too coarse to cleanly cut between cameras as a subject passes the car. SentryMerge groups hits by Tesla timestamp, picks the best multi-camera clip-set, asks Gemini 2.5 Pro for sub-second visibility ranges per camera, infers travel direction along the front-back axis, and stitches the clips into one video with ffmpeg. The output path is written to `~/.sentrysearch/last_clip.json` so a downstream tool (viewer, redactor, sharer) can pick it up.

SentryMerge does **only** the last-mile stitch. Retrieval stays in SentrySearch.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/) (if you don't have it):

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone and install:

```bash
git clone https://github.com/ssrajadh/sentrymerge.git
cd sentrymerge
uv tool install .
```

For development (editable install + tests), use `uv sync --group test` instead and run commands via `uv run sentrymerge ...` / `uv run pytest`.

3. Make sure `GEMINI_API_KEY` is set. SentryMerge reads it from `~/.sentrysearch/.env` (the same file `sentrysearch init` writes) or from the environment. If you've already run `sentrysearch init`, you're done.

4. Run a search and stitch:

```bash
sentrysearch search "honda fit" -n 20 --no-trim
sentrymerge --last
```

ffmpeg is required for stitching. If you don't have it system-wide, the bundled `imageio-ffmpeg` from sentrysearch is used automatically when both are installed.

## Usage

### Stitch from last search

```bash
$ sentrymerge --last
Clip-set: 2026-03-12_10-44-17  (4 cameras, top score 0.78)
VLM detections:
  back            00:08.4-00:14.2  conf=0.91
  left_repeater   00:12.1-00:19.8  conf=0.88
  right_repeater  (no detection)
  front           00:18.5-00:24.3  conf=0.84
Direction: back→front
Stitched 3 segments → merge.mp4
```

`--last` reads `~/.sentrysearch/last_search.json` (written by `sentrysearch search`) and stitches the best multi-camera clip-set from those results. The receipt schema is v1; corrupt or stale receipts are rejected and you'll be told to re-run search.

### Stitch from a fresh query

```bash
sentrymerge --query "honda fit"
sentrymerge --image ~/Downloads/reference.jpg
```

These re-run SentrySearch under the hood (lazy-imported, so `--last` keeps working without it). All other flags apply.

### Pick a specific event

If the top clip-set isn't the one you want, force one by its Tesla timestamp prefix:

```bash
sentrymerge --last --clip-set 2026-03-12_10-44-17
```

The prefix is just `YYYY-MM-DD_HH-MM-SS` — the camera suffix and extension are stripped.

### Reduce VLM wobble with `--vlm-votes`

Gemini 2.5 Pro at `temperature=0` still wobbles on visibility-range tasks (start/end shift by ~0.5s run-to-run, occasional false positives on confusable subjects). `--vlm-votes k` runs the VLM `k` times per camera and merges via per-slot majority vote at 0.1s resolution.

```bash
sentrymerge --last --vlm-votes 3   # 3× VLM cost, fewer wobbles
```

`k=1` is a pass-through (the default). For high-stakes runs, `k=3` is the usual trade. Cost scales linearly with `k`.

### Verbose mode

Add `-v` / `--verbose` to see the grouped clip-set candidates, raw VLM JSON per camera, the inferred axis direction, the ownership timeline, and the full ffmpeg command.

## Why this exists

SentrySearch indexes Tesla SentryCam clips at 30s chunk granularity. When an object passes the car, *multiple* sister-camera clips for the same timestamp light up — but chunk-level timing is too coarse to cut frame-accurately between cameras. "Honda Fit cleanly moves from `left_repeater` → `front`" needs sub-second handoffs, which means a per-camera visibility pass on top of the index.

The key insight: cars don't teleport. An object passing the Tesla traverses the front-back axis monotonically — either back→front (overtake) or front→back (oncoming). SentryMerge encodes that prior, picks the direction from the VLM's earliest-detection times, and squeezes out cameras whose times contradict the chosen ordering rather than forcing a physically impossible cut.

## Direction labels

The summary line tells you which physical model SentryMerge inferred:

- `back→front` — overtake. Subject came from behind, passed the car.
- `front→back` — oncoming. Subject came toward the car, receded behind.
- `side-only` — only side cameras detected; axis is tied. Falls back to temporal order.
- `single-camera` — one camera saw it; emits one segment with a warning.
- `none` — nothing detected above threshold.

When only `front` and `back` detect (no side), SentryMerge keeps both segments but warns — physically suspicious, often a VLM hallucination on one end.

## Cost

Per stitch, SentryMerge issues one Gemini 2.5 Pro call per camera that has a candidate clip — typically 3-4 calls per event, each on ~30s of 480p video. With `--vlm-votes k`, multiply by `k`.

For exact pricing, see [Gemini API pricing](https://ai.google.dev/pricing). At time of writing, a single stitch is on the order of cents.

ffmpeg work is local and free.

## Limitations & Future Work

- **One range per camera.** If the VLM returns multiple disjoint detections in the same chunk (the same subject appears twice), only the earliest above-threshold range is used. Fine for "one car passes" events.
- **No corroboration gate.** A confidently hallucinated detection on one camera will still produce a segment. A planned `--require-corroboration` flag would drop a camera's hit unless the temporally adjacent camera on the axis also detected something nearby.
- **No `--strict-axis` yet.** Front-only or front+back-without-side cases warn but don't drop. Trivially small change once it's wanted.
- **No receipt staleness check.** `--last` doesn't yet warn if the SentrySearch receipt is hours old.
- **Gemini Pro at temp=0 still wobbles.** `--vlm-votes` is the current mitigation. A deterministic detector (GroundingDINO or similar) is deferred until VLM brittleness is observed in real use after multi-vote.

## Compatibility

- Tesla SentryCam filenames: `YYYY-MM-DD_HH-MM-SS-<camera>.mp4`, where camera ∈ {`front`, `back`, `left_repeater`, `right_repeater`}.
- Sister files don't have to live in the same directory — SentryMerge searches every parent directory of any grouped hit.
- Output is always 1280×720 / 30 fps / yuv420p (concat filter requires consistent SAR/fps/format across inputs).

## Requirements

- Python 3.11+
- `ffmpeg` on PATH (or the bundled `imageio-ffmpeg` from sentrysearch)
- Gemini API key ([get one free](https://aistudio.google.com/apikey)), via `~/.sentrysearch/.env` or the environment
- SentrySearch installed and indexed if you want `--query` / `--image`. `--last` works without it as long as a valid `~/.sentrysearch/last_search.json` exists.
