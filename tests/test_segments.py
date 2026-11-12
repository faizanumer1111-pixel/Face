from fractions import Fraction

import pytest

from app.segments import (Segment, SegmentError, check_ready, format_timecode, frame_to_seconds, from_json,
                          normalize, parse_time, seconds_to_frame, stats)

F30 = Fraction(30)
NTSC = Fraction(30000, 1001)


def test_timecode_round_trip():
    for fps in (Fraction(24), Fraction(25), F30, NTSC, Fraction(60000, 1001)):
        for frame in (0, 1, 59, 60, 1799, 107892):
            assert seconds_to_frame(frame_to_seconds(frame, fps), fps) == frame


def test_parse_time_formats():
    assert parse_time("2.5", F30) == 75
    assert parse_time("1:02.5", F30) == 1875
    assert parse_time("0:01:02.5", F30) == 1875
    assert parse_time("#120", F30) == 120
    assert parse_time("f120", F30) == 120
    assert parse_time(120, F30) == 120
    assert parse_time(2.5, F30) == 75
    # NTSC: 2 seconds is frame 59.94 -> nearest boundary 60
    assert parse_time("2", NTSC) == 60
    with pytest.raises(SegmentError):
        parse_time("soon", F30)


def test_format_timecode():
    assert format_timecode(0) == "0:00.000"
    assert format_timecode(5.5) == "0:05.500"
    assert format_timecode(62.25) == "1:02.250"
    assert format_timecode(3723.5) == "1:02:03.500"


def test_segment_edges_are_frame_accurate():
    # 2.0s-5.5s at 30 fps is frames 60..164
    (s,) = from_json([{"start": "2.0", "end": "5.5"}], F30)
    assert (s.start, s.end, s.length) == (60, 165, 105)


def test_merge_overlapping_and_touching_same_settings():
    segs = [Segment(0, 30), Segment(20, 50), Segment(50, 60), Segment(100, 120)]
    out = normalize(segs, 300)
    assert [(s.start, s.end) for s in out] == [(0, 60), (100, 120)]


def test_touching_different_settings_stay_separate():
    out = normalize([Segment(0, 30, "everyone"), Segment(30, 60, "main")], 300)
    assert [(s.start, s.end, s.mode) for s in out] == [(0, 30, "everyone"), (30, 60, "main")]


def test_overlap_with_different_settings_is_an_error():
    with pytest.raises(SegmentError, match="overlap"):
        normalize([Segment(0, 40, "everyone"), Segment(30, 60, "main")], 300)
    # same mode but different people also conflicts
    with pytest.raises(SegmentError):
        normalize([Segment(0, 40, "cast", cast=[{"ref": "a", "face": None}]),
                   Segment(30, 60, "cast", cast=[{"ref": "b", "face": None}])], 300)


def test_clamp_sort_and_drop_empty():
    out = normalize([Segment(250, 400), Segment(-5, 10), Segment(50, 50)], 300)
    assert [(s.start, s.end) for s in out] == [(0, 10), (250, 300)]


def test_cast_dropped_unless_cast_mode():
    (s,) = normalize([Segment(0, 10, "everyone", cast=[{"ref": "x", "face": None}])], 100)
    assert s.cast == []


def test_old_one_person_form_becomes_cast():
    (s,) = from_json([{"start": 0, "end": 10, "mode": "one", "ref": "r1"}], F30)
    assert s.mode == "cast" and s.cast == [{"ref": "r1", "face": None}]


def test_stats():
    st = stats([Segment(0, 60), Segment(100, 280)], 1800)
    assert st == {"frames_to_process": 240, "total_frames": 1800, "percent_skipped": 86.7}


def test_check_ready():
    with pytest.raises(SegmentError, match="at least one segment"):
        check_ready([], set(), {"f"}, "f")
    with pytest.raises(SegmentError, match="nobody is picked"):
        check_ready([Segment(0, 10, "cast")], set(), {"f"}, "f")
    with pytest.raises(SegmentError, match="Add a face"):
        check_ready([Segment(0, 10, "everyone")], set(), set(), None)
    with pytest.raises(SegmentError, match="deleted"):
        check_ready([Segment(0, 10, "cast", cast=[{"ref": "r1", "face": "gone"}])], {"r1"}, {"f"}, "f")
    check_ready([Segment(0, 10, "cast", cast=[{"ref": "r1", "face": None}]), Segment(20, 30, "main")],
                {"r1"}, {"f"}, "f")


def test_bad_mode_and_threshold():
    with pytest.raises(SegmentError):
        from_json([{"start": 0, "end": 10, "mode": "nobody"}], F30)
    with pytest.raises(SegmentError):
        from_json([{"start": 0, "end": 10, "mode": "cast", "threshold": 1.5}], F30)
