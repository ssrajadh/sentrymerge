"""Tests for sentrymerge.core.build_ownership_timeline.

These pin down axis-direction inference plus the risk-list edge cases:
single-camera, side-only, front+back-without-side, anomalous time ordering.
"""
from __future__ import annotations

from sentrymerge.core import build_ownership_timeline


def r(start: float, end: float, conf: float, note: str = "") -> dict:
    return {"start": start, "end": end, "confidence": conf, "note": note}


# ---------------------------------------------------------------------------
# Direction inference
# ---------------------------------------------------------------------------


def test_overtake_direction_back_to_front():
    """Object passes Tesla on the left: back sees first, then left, then front."""
    per_cam = {
        "back":          [r(0.0, 1.5, 0.90)],
        "left_repeater": [r(0.0, 1.2, 0.95)],
        "front":         [r(2.0, 6.5, 0.90)],
    }
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    assert tl.direction == "back→front"
    cams = [s.camera for s in tl.segments]
    # back is squeezed (back.start == left.start) — left then front.
    assert cams == ["left_repeater", "front"]
    assert tl.segments[0].start == 0.0
    assert tl.segments[0].end == 2.0
    assert tl.segments[1].start == 2.0
    assert tl.segments[1].end == 6.5


def test_oncoming_direction_front_to_back():
    """Oncoming car: front sees first, then side, then back."""
    per_cam = {
        "front":         [r(0.0, 2.0, 0.90)],
        "left_repeater": [r(1.5, 3.5, 0.85)],
        "back":          [r(3.0, 5.0, 0.80)],
    }
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    assert tl.direction == "front→back"
    cams = [s.camera for s in tl.segments]
    assert cams == ["front", "left_repeater", "back"]
    assert tl.segments[0].end == 1.5
    assert tl.segments[1].end == 3.0
    assert tl.segments[2].end == 5.0


# ---------------------------------------------------------------------------
# Risk-list edge cases
# ---------------------------------------------------------------------------


def test_single_camera_emits_one_segment_with_warning():
    per_cam = {"front": [r(1.0, 4.0, 0.85)]}
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    assert tl.direction == "single-camera"
    assert len(tl.segments) == 1
    assert tl.segments[0].camera == "front"
    assert tl.segments[0].start == 1.0
    assert tl.segments[0].end == 4.0
    assert any("only front" in w for w in tl.warnings)


def test_side_only_falls_back_to_temporal_order():
    """Both side cams detect; axis position is tied so direction is undefined."""
    per_cam = {
        "right_repeater": [r(2.0, 4.0, 0.80)],
        "left_repeater":  [r(0.0, 1.5, 0.90)],
    }
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    assert tl.direction == "side-only"
    cams = [s.camera for s in tl.segments]
    assert cams == ["left_repeater", "right_repeater"]


def test_front_back_without_side_warns_but_keeps_both():
    """Physically suspicious — flagged in warnings, output not silently mutated."""
    per_cam = {
        "back":  [r(0.0, 2.0, 0.85)],
        "front": [r(3.0, 5.0, 0.90)],
    }
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    assert tl.direction == "back→front"
    assert [s.camera for s in tl.segments] == ["back", "front"]
    assert any("no side-camera corroboration" in w for w in tl.warnings)


def test_squeezed_camera_dropped_when_axis_order_violates_time():
    """back's first detection is later than left's — squeezed to zero duration."""
    per_cam = {
        "back":          [r(1.0, 3.0, 0.80)],
        "left_repeater": [r(0.0, 2.0, 0.90)],
        "front":         [r(2.0, 5.0, 0.85)],
    }
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    # back wants [back.start=1.0, left.start=0.0] → end < start → dropped.
    cams = [s.camera for s in tl.segments]
    assert cams == ["left_repeater", "front"]
    assert tl.segments[0].start == 0.0
    assert tl.segments[0].end == 2.0


# ---------------------------------------------------------------------------
# Threshold + duration handling
# ---------------------------------------------------------------------------


def test_below_threshold_detections_dropped():
    per_cam = {
        "front":         [r(0.0, 5.0, 0.20)],   # below default 0.30
        "left_repeater": [r(0.0, 5.0, 0.50)],
    }
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    cams = [s.camera for s in tl.segments]
    assert cams == ["left_repeater"]


def test_total_duration_clamps_last_segment():
    per_cam = {"front": [r(0.0, 100.0, 0.90)]}
    tl = build_ownership_timeline(per_cam, total_duration=7.5)
    assert tl.segments[0].end == 7.5


def test_no_detections_returns_empty():
    tl = build_ownership_timeline({}, total_duration=10.0)
    assert tl.segments == []
    assert tl.direction == "none"


def test_all_below_threshold_returns_empty():
    per_cam = {"front": [r(0.0, 5.0, 0.10)]}
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    assert tl.segments == []


def test_uses_earliest_above_threshold_range_per_camera():
    """When a camera has multiple ranges, the earliest valid one is used."""
    per_cam = {"front": [r(5.0, 8.0, 0.85), r(0.0, 2.0, 0.50)]}
    tl = build_ownership_timeline(per_cam, total_duration=10.0)
    assert len(tl.segments) == 1
    assert tl.segments[0].start == 0.0
    assert tl.segments[0].end == 2.0
    assert tl.segments[0].confidence == 0.50
