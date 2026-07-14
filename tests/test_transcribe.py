"""filter_range windowing and the focus-mode transcript context pad."""
from __future__ import annotations

import transcribe


class TestFilterRange:
    def test_unbounded_returns_everything(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 10.0, "end": 11.0, "text": "b"}]
        assert transcribe.filter_range(segs, None, None) == segs

    def test_drops_segments_outside_the_window(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 10.0, "end": 11.0, "text": "b"}]
        assert transcribe.filter_range(segs, 5.0, 20.0) == [segs[1]]

    def test_keeps_segments_that_only_overlap_the_edge(self):
        segs = [{"start": 4.0, "end": 6.0, "text": "a"}]
        assert transcribe.filter_range(segs, 5.0, 20.0) == segs


class TestContextWindow:
    def test_unfocused_stays_unbounded(self):
        assert transcribe.context_window(None, None) == (None, None)

    def test_pads_both_sides_within_ratio(self):
        # 40s focus * 0.25 ratio = 10s pad on each side, within the clamp.
        lo, hi = transcribe.context_window(100.0, 140.0)
        assert lo == 90.0
        assert hi == 150.0

    def test_pad_is_floored_for_a_short_focus(self):
        # 4s focus * 0.25 = 1s, below the 10s floor.
        lo, hi = transcribe.context_window(100.0, 104.0)
        assert lo == 90.0
        assert hi == 114.0

    def test_pad_is_capped_for_a_long_focus(self):
        # 400s focus * 0.25 = 100s, above the 30s ceiling.
        lo, hi = transcribe.context_window(100.0, 500.0)
        assert lo == 70.0
        assert hi == 530.0

    def test_start_is_clamped_to_zero(self):
        lo, _hi = transcribe.context_window(5.0, 45.0)
        assert lo == 0.0

    def test_end_is_clamped_to_full_duration(self):
        _lo, hi = transcribe.context_window(100.0, 195.0, full_duration=200.0)
        assert hi == 200.0

    def test_open_start_stays_none(self):
        # lo defaults to 0 for the span calc: 100s span * 0.25 = 25s pad.
        lo, hi = transcribe.context_window(None, 100.0)
        assert lo is None
        assert hi == 125.0

    def test_open_end_stays_none(self):
        lo, hi = transcribe.context_window(100.0, None)
        assert hi is None
