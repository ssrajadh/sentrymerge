"""SentryMerge CLI — stitches cross-camera Tesla clips of the same event.

Reads SentrySearch's last_search receipt (or accepts ``--query``), groups hits
by Tesla timestamp prefix, runs Gemini Vision on each sister camera, builds an
axis-direction-aware ownership timeline, and emits a single stitched mp4.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import click
from dotenv import load_dotenv

from . import _toolkit_cache
from .backends import (
    BACKEND_NAMES, GEMINI_MODELS, OPENAI_MODELS, QWEN_MODEL_ALIASES,
    resolve_backend,
)
from .core import (
    build_ownership_timeline,
    build_stitch_command,
    find_sister_files,
    get_video_duration,
    merge_votes,
    parse_tesla_filename,
)


_ENV_PATH = os.path.join(os.path.expanduser("~"), ".sentrysearch", ".env")
load_dotenv(_ENV_PATH)
load_dotenv()


def _open_file(path: Path) -> None:
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass  # non-critical; the file is saved


def _group_by_clipset(results: list[dict]) -> dict[str, list[dict]]:
    """Group result dicts by Tesla timestamp prefix; tag each with camera."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        ts, cam = parse_tesla_filename(r["source_file"])
        if ts is None:
            continue
        r = dict(r)  # copy so we don't mutate caller data
        r["timestamp"] = ts
        r["camera"] = cam
        grouped[ts].append(r)
    return grouped


def _pick_clipset(grouped: dict[str, list[dict]],
                  forced: str | None) -> str | None:
    if forced is not None:
        return forced if forced in grouped else None
    multi = {t: hits for t, hits in grouped.items()
             if len({h["camera"] for h in hits}) >= 2}
    if multi:
        return max(multi,
                   key=lambda t: max(h["similarity_score"] for h in multi[t]))
    if grouped:
        return max(grouped,
                   key=lambda t: max(h["similarity_score"] for h in grouped[t]))
    return None


def _search_dirs(grouped: dict[str, list[dict]], timestamp: str) -> list[Path]:
    """Every directory that holds an indexed file for *timestamp*."""
    seen: dict[Path, None] = {}
    for h in grouped.get(timestamp, []):
        seen[Path(h["source_file"]).parent] = None
    return list(seen)


@click.command()
@click.option("--last", "use_last", is_flag=True,
              help="Use the most recent SentrySearch search receipt.")
@click.option("--query", default=None,
              help="Override the query (text). Mutually exclusive with --image.")
@click.option("--image", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Use an image as the query. Mutually exclusive with --query.")
@click.option("--clip-set", "clip_set", default=None,
              help="Force a specific Tesla timestamp prefix "
                   "(e.g. 2026-03-12_10-44-17).")
@click.option("--vlm-backend",
              type=click.Choice(BACKEND_NAMES, case_sensitive=False),
              default=None,
              help="VLM backend. Default: auto-detect from env "
                   "(GEMINI_API_KEY → gemini, OPENAI_API_KEY → openai, "
                   "else qwen local).")
@click.option("--vlm-model", default=None,
              help=(
                  "Override the backend's default model id.\n"
                  f"  gemini: {', '.join(GEMINI_MODELS)}\n"
                  f"  openai: {', '.join(OPENAI_MODELS)}\n"
                  f"  qwen:   {', '.join(QWEN_MODEL_ALIASES)} "
                  "(aliases imply --vlm-backend qwen), or any HF repo id.\n"
                  "Qwen auto-detects from hardware when omitted."
              ))
@click.option("--quantize/--no-quantize", default=None,
              help="Force 4-bit quantization on/off for the qwen backend. "
                   "Default: auto-detect (4-bit on NVIDIA <20GB VRAM).")
@click.option("--conf-threshold", default=0.3, show_default=True, type=float,
              help="Minimum VLM confidence for a detection to count.")
@click.option("--vlm-votes", default=1, show_default=True, type=click.IntRange(min=1),
              help="Run VLM k times per camera and majority-vote (mitigates "
                   "non-determinism). Multiplies cost by k.")
@click.option("-o", "--output", default="merge.mp4", show_default=True,
              type=click.Path(dir_okay=False),
              help="Output mp4 path.")
@click.option("--verbose", "-v", is_flag=True, help="Show debug info.")
def cli(use_last, query, image, clip_set, vlm_backend, vlm_model, quantize,
        conf_threshold, vlm_votes, output, verbose):
    """Stitch a cross-camera dashcam clip from a SentrySearch result."""
    # ---- resolve query + results ----------------------------------------
    sources = sum(bool(x) for x in (use_last, query, image))
    if sources != 1:
        raise click.UsageError(
            "specify exactly one of --last, --query, --image"
        )

    if use_last:
        receipt = _toolkit_cache.read_last_search()
        if receipt is None:
            raise click.UsageError(
                "no SentrySearch receipt found at "
                f"{_toolkit_cache._last_search_path()}.\n"
                "Run `sentrysearch search <query>` first."
            )
        query_text = receipt.query
        image_path = receipt.image_path
        results = receipt.results
        click.echo(f"Loaded receipt ({receipt.age_seconds}s old): "
                   f"{'image' if receipt.is_image_query else 'text'} query, "
                   f"{len(results)} results")
    else:
        # User supplied --query or --image directly: re-run search via
        # sentrysearch's public API (graceful import — keeps sentrymerge
        # usable when sentrysearch isn't installed, with --last only).
        try:
            from sentrysearch.search import (
                search_footage, search_footage_by_image,
            )
            from sentrysearch.store import SentryStore, detect_index
        except ImportError as e:
            raise click.UsageError(
                f"--query/--image requires sentrysearch to be installed "
                f"(import failed: {e}). Use --last instead."
            ) from None
        backend, model = detect_index()
        store = SentryStore(backend=backend or "gemini", model=model)
        if query is not None:
            query_text, image_path = query, None
            results = search_footage(query, store, n_results=500,
                                     verbose=verbose)
        else:
            query_text, image_path = None, Path(os.path.abspath(image))
            results = search_footage_by_image(
                image, store, n_results=500, verbose=verbose,
            )

    if not results:
        click.secho("No search results to merge.", fg="yellow")
        sys.exit(1)

    # ---- pick clip-set --------------------------------------------------
    grouped = _group_by_clipset(results)
    if not grouped:
        click.secho(
            "No Tesla-named files in results "
            "(expected YYYY-MM-DD_HH-MM-SS-<camera>.mp4).",
            fg="yellow",
        )
        sys.exit(1)

    timestamp = _pick_clipset(grouped, clip_set)
    if timestamp is None:
        click.secho(f"No clip-set found for --clip-set {clip_set!r}.", fg="red")
        sys.exit(1)

    label = query_text if query_text is not None else f"image:{image_path}"
    click.echo(f"Query: {label}")
    click.echo(f"Clip-set: {timestamp}")

    sisters = find_sister_files(timestamp, _search_dirs(grouped, timestamp))
    if not sisters:
        click.secho("No sister files on disk for this clip-set.", fg="red")
        sys.exit(1)
    click.echo(f"Sister files: {len(sisters)}/4 → {sorted(sisters)}")

    hits_by_cam = {h["camera"]: h["similarity_score"]
                   for h in grouped[timestamp]}
    click.echo("\nPer-camera index scores:")
    for cam in ("front", "left_repeater", "right_repeater", "back"):
        score = hits_by_cam.get(cam)
        on_disk = "yes" if cam in sisters else "no "
        score_str = f"{score:.3f}" if score is not None else "— (no hit)"
        click.echo(f"  {cam:<16} on_disk={on_disk}  index_score={score_str}")

    # ---- VLM pass -------------------------------------------------------
    try:
        backend = resolve_backend(vlm_backend, model=vlm_model, quantize=quantize)
    except (ImportError, RuntimeError, ValueError) as e:
        click.secho(str(e), fg="red")
        sys.exit(1)

    vote_suffix = f", {vlm_votes} votes/camera" if vlm_votes > 1 else ""
    click.echo(f"\nRunning {backend.name}/{backend.model} on {len(sisters)} "
               f"sister clips{vote_suffix}...")
    per_cam_ranges: dict[str, list[dict]] = {}
    for cam, path in sisters.items():
        votes = [
            backend.detect(
                path, query=query_text, image_path=image_path, verbose=verbose,
            )
            for _ in range(vlm_votes)
        ]
        ranges = merge_votes(votes) if vlm_votes > 1 else votes[0]
        per_cam_ranges[cam] = ranges
        rs = (", ".join(f"{r['start']:.1f}-{r['end']:.1f}@{r['confidence']:.2f}"
                        for r in ranges) or "— (not visible)")
        click.echo(f"  {cam:<16} → {rs}")

    # ---- timeline -------------------------------------------------------
    total_duration = max(get_video_duration(p) for p in sisters.values())
    timeline = build_ownership_timeline(
        per_cam_ranges, total_duration, conf_threshold=conf_threshold,
    )
    click.echo(f"\nDirection: {timeline.direction}")
    for w in timeline.warnings:
        click.secho(f"  Warning: {w}", fg="yellow")

    if not timeline.segments:
        click.secho(
            f"\nNo segments above VLM conf threshold {conf_threshold}. "
            f"Nothing to stitch.", fg="yellow",
        )
        sys.exit(1)

    click.echo(f"\nOwnership timeline ({len(timeline.segments)} segments):")
    for seg in timeline.segments:
        click.echo(
            f"  {seg.start:5.1f}-{seg.end:5.1f}s  "
            f"{seg.camera:<16}  conf={seg.confidence:.2f}"
        )

    # ---- ffmpeg stitch --------------------------------------------------
    output_path = Path(output).resolve()
    cmd = build_stitch_command(timeline.segments, sisters, output_path)
    click.echo(f"\nStitching → {output_path}")
    subprocess.run(cmd, check=True)

    try:
        _toolkit_cache.write_last_clip(output_path, saved_by="sentrymerge")
    except Exception as e:
        click.secho(f"(warning: could not write last-clip cache: {e})",
                    fg="yellow")

    click.secho(f"Wrote: {output_path}", fg="green")
    _open_file(output_path)


if __name__ == "__main__":
    cli()
