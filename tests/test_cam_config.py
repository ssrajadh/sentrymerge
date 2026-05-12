"""Tests for sentrymerge.cam_config: built-in loading, validation, and the
canonical-id mapping that supports filename formats whose suffixes aren't
the canonical camera ids (e.g. BlackVue _NF → front).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from sentrymerge import cam_config


def test_list_builtin_includes_tesla_and_hides_underscore_files():
    builtins = cam_config.list_builtin()
    assert "tesla" in builtins
    # The fill-in template `_example.toml` exists on disk but is hidden.
    assert not any(name.startswith("_") for name in builtins)


def test_tesla_parse_recognizes_canonical_filenames():
    cam = cam_config.load("tesla")
    assert cam.parse("2026-03-12_10-44-17-front.mp4") == \
        ("2026-03-12_10-44-17", "front")
    assert cam.parse("/abs/path/2026-03-12_10-44-17-left_repeater.mp4") == \
        ("2026-03-12_10-44-17", "left_repeater")
    assert cam.parse("not-a-tesla-clip.mp4") == (None, None)


def test_camera_aliases_map_raw_capture_to_canonical(tmp_path: Path):
    """A config with camera_aliases maps the regex's `camera` capture
    (e.g. `_F` / `_R`) to the canonical id used in `cameras` / `axis_pos`."""
    toml = tmp_path / "fr.toml"
    toml.write_text(textwrap.dedent('''
        name = "fr"
        cameras = ["rear", "front"]
        filename_pattern = '^(?P<timestamp>\\d{8}_\\d{6})_(?P<camera>F|R)\\.mp4$'
        [camera_aliases]
        F = "front"
        R = "rear"
        [axis_positions]
        rear = 0
        front = 1
    '''))
    cam = cam_config.load(str(toml))
    assert cam.parse("20240115_143000_F.mp4") == ("20240115_143000", "front")
    assert cam.parse("20240115_143000_R.mp4") == ("20240115_143000", "rear")


def test_detect_from_filenames_picks_tesla_for_tesla_names():
    files = [
        "/abs/2026-03-12_10-44-17-front.mp4",
        "/abs/2026-03-12_10-44-17-back.mp4",
        "/abs/2026-02-12_20-02-15-left_repeater.mp4",
    ]
    detected = cam_config.detect_from_filenames(files)
    assert detected is not None
    assert detected.name == "tesla"


def test_detect_from_filenames_returns_none_for_unknown_format():
    files = ["random.mp4", "garmin_clip_001.mp4", "iphone_video.mov"]
    assert cam_config.detect_from_filenames(files) is None


def test_unknown_builtin_raises_with_list():
    with pytest.raises(FileNotFoundError, match="Built-ins:.*tesla"):
        cam_config.load("not-a-real-cam")


def test_load_custom_toml_from_path(tmp_path: Path):
    toml = tmp_path / "garmin.toml"
    toml.write_text(textwrap.dedent('''
        name = "garmin"
        description = "Garmin single front cam"
        cameras = ["front"]
        filename_pattern = '^GRMN_(?P<timestamp>\\d{14})_(?P<camera>front)\\.mp4$'
        [axis_positions]
        front = 0
    '''))
    cam = cam_config.load(str(toml))
    assert cam.name == "garmin"
    assert cam.parse("GRMN_20240115143000_front.mp4") == \
        ("20240115143000", "front")


def test_missing_required_field_raises(tmp_path: Path):
    toml = tmp_path / "broken.toml"
    toml.write_text('name = "broken"\ncameras = ["a"]\n')
    with pytest.raises(ValueError, match="missing required field"):
        cam_config.load(str(toml))


def test_cameras_axis_mismatch_raises(tmp_path: Path):
    toml = tmp_path / "bad.toml"
    toml.write_text(textwrap.dedent('''
        name = "bad"
        cameras = ["a", "b"]
        filename_pattern = '^(?P<timestamp>\\d+)-(?P<camera>a|b)\\.mp4$'
        [axis_positions]
        a = 0
        c = 1
    '''))
    with pytest.raises(ValueError, match="must list the same ids"):
        cam_config.load(str(toml))


def test_regex_without_named_groups_raises(tmp_path: Path):
    toml = tmp_path / "bad.toml"
    toml.write_text(textwrap.dedent('''
        name = "bad"
        cameras = ["a"]
        filename_pattern = '^(.+)\\.mp4$'
        [axis_positions]
        a = 0
    '''))
    with pytest.raises(ValueError, match="named groups"):
        cam_config.load(str(toml))
