"""Tests for sentrymerge.core.merge_votes — multi-run VLM majority voting."""
from __future__ import annotations

from sentrymerge.core import merge_votes


def r(start: float, end: float, conf: float) -> dict:
    return {"start": start, "end": end, "confidence": conf, "note": ""}


def test_single_vote_passes_through():
    votes = [[r(0.0, 2.0, 0.9)]]
    assert merge_votes(votes) == [r(0.0, 2.0, 0.9)]


def test_empty_votes_returns_empty():
    assert merge_votes([]) == []
    assert merge_votes([[]]) == []
    assert merge_votes([[], [], []]) == []


def test_three_votes_majority_keeps_overlap_drops_outliers():
    """Three runs agree on 1.0-3.0; one outlier extends to 4.0. Majority (≥2)
    keeps 1.0-4.0 because slot 3.0-4.0 needs only 2 votes — but only 1 vote
    covers it, so it gets dropped. Result: 1.0-3.0."""
    votes = [
        [r(1.0, 3.0, 0.9)],
        [r(1.0, 3.0, 0.85)],
        [r(1.0, 4.0, 0.7)],
    ]
    merged = merge_votes(votes, dt=0.1)
    assert len(merged) == 1
    assert abs(merged[0]["start"] - 1.0) < 0.05
    assert abs(merged[0]["end"] - 3.0) < 0.15


def test_majority_window_is_strict_intersection_when_one_dissents():
    votes = [
        [r(0.5, 2.0, 0.9)],
        [r(0.5, 2.0, 0.9)],
        [r(1.5, 3.0, 0.9)],
    ]
    # Majority (>= 2) at every slot in [0.5, 2.0]; slot at 2.0..3.0 has only 1.
    merged = merge_votes(votes, dt=0.1)
    assert len(merged) == 1
    assert abs(merged[0]["start"] - 0.5) < 0.05
    assert abs(merged[0]["end"] - 2.0) < 0.15


def test_no_overlap_yields_empty_when_majority_required():
    """Three disjoint detections: no slot has ≥2 votes → empty."""
    votes = [
        [r(0.0, 1.0, 0.9)],
        [r(2.0, 3.0, 0.9)],
        [r(4.0, 5.0, 0.9)],
    ]
    assert merge_votes(votes, dt=0.1) == []


def test_min_votes_override_can_be_more_strict():
    """Require unanimity from 2 votes that disagree → empty."""
    votes = [
        [r(0.0, 1.0, 0.9)],
        [r(0.5, 1.5, 0.9)],
    ]
    # Default majority for k=2 is 1 (ceil(2/2)) → wait, (k+1)//2 for k=2 = 1.
    # Actually (2+1)//2 = 1. Hmm — that means majority of 2 is 1? Let's check.
    # majority_of_k uses (k+1)//2: k=2 → 1, k=3 → 2, k=4 → 2, k=5 → 3.
    # So for k=2, default min_votes=1 means ANY single vote counts.
    # Use min_votes=2 to require unanimity.
    merged = merge_votes(votes, dt=0.1, min_votes=2)
    # Only 0.5..1.0 has both.
    assert len(merged) == 1
    assert abs(merged[0]["start"] - 0.5) < 0.05
    assert abs(merged[0]["end"] - 1.0) < 0.15


def test_overlapping_ranges_in_single_run_count_once():
    """A single run with two overlapping ranges shouldn't double-count slots."""
    votes = [
        [r(0.0, 2.0, 0.9), r(1.0, 3.0, 0.5)],   # overlapping in single run
        [r(0.0, 3.0, 0.7)],
    ]
    # With min_votes=2 (unanimity), slots 0..3 all have both runs voting → 0..3.
    merged = merge_votes(votes, dt=0.1, min_votes=2)
    assert len(merged) == 1
    assert abs(merged[0]["start"]) < 0.05
    assert abs(merged[0]["end"] - 3.0) < 0.15


def test_confidence_averaged_across_voting_runs():
    votes = [
        [r(0.0, 1.0, 1.0)],
        [r(0.0, 1.0, 0.5)],
    ]
    merged = merge_votes(votes, dt=0.1, min_votes=2)
    assert len(merged) == 1
    # Each slot averages (1.0 + 0.5) / 2 = 0.75; over many slots still 0.75.
    assert abs(merged[0]["confidence"] - 0.75) < 0.01
