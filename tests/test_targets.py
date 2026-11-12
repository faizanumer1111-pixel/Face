import numpy as np

from app.faces import Face
from app.targets import CastSelector, EveryoneSelector, KpsSmoother, MainFaceSelector, ReferenceSelector, iou, make_selector

SHAPE = (1080, 1920, 3)


def face(x, y, size, emb=None):
    f = Face(bbox=np.array([x, y, x + size, y + size], dtype=np.float32), kps=np.zeros((5, 2)), score=0.9)
    if emb is not None:
        f.embedding = emb / np.linalg.norm(emb)
    return f


def unit(seed):
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def mix(a, b, sim):
    """A unit vector with cosine `sim` to a (b gives the orthogonal direction)."""
    b = b - np.dot(b, a) * a
    b /= np.linalg.norm(b)
    return sim * a + np.sqrt(1 - sim**2) * b


def test_iou():
    a = np.array([0, 0, 10, 10.0])
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30.0])) == 0.0
    assert abs(iou(a, np.array([5, 0, 15, 10.0])) - 1 / 3) < 1e-6


def test_everyone_swaps_all():
    faces = [face(0, 0, 100), face(500, 0, 50), face(900, 0, 80)]
    assert EveryoneSelector().select(faces, SHAPE) == faces
    assert EveryoneSelector().select([], SHAPE) == []


def test_main_picks_largest():
    small, big = face(100, 100, 100), face(1200, 100, 300)
    assert MainFaceSelector().select([small, big], SHAPE) == [big]


def test_main_tie_goes_to_center():
    edge = face(50, 400, 200)
    center = face(860, 400, 195)  # within 10% of the largest, but centred
    assert MainFaceSelector().select([edge, center], SHAPE) == [center]


def test_main_hysteresis_needs_several_frames():
    sel = MainFaceSelector(switch_ratio=1.3, switch_frames=5)
    a = face(200, 300, 200)
    assert sel.select([a, face(1300, 300, 150)], SHAPE)[0] is a
    # b becomes 1.4x a's area: a keeps the swap for 4 frames, b takes over on the 5th
    picks = []
    for _ in range(6):
        a_now, b_now = face(200, 300, 200), face(1300, 300, 237)
        picks.append("b" if sel.select([a_now, b_now], SHAPE)[0] is b_now else "a")
    assert picks == ["a", "a", "a", "a", "b", "b"]


def test_main_small_flicker_does_not_switch():
    sel = MainFaceSelector()
    a = face(200, 300, 200)
    assert sel.select([a, face(1300, 300, 150)], SHAPE)[0] is a
    for i in range(20):
        # b is bigger only every other frame, so its streak never builds up
        size = 260 if i % 2 else 150
        a_now = face(200, 300, 200)
        assert sel.select([a_now, face(1300, 300, size)], SHAPE)[0] is a_now


def test_main_cut_to_new_big_face_switches_at_once():
    sel = MainFaceSelector()
    sel.select([face(200, 300, 300)], SHAPE)
    new = face(1200, 200, 280)  # different place, similar size: a new shot
    assert sel.select([new], SHAPE) == [new]


def test_main_missed_detection_does_not_jump_to_background_face():
    sel = MainFaceSelector(max_gap=3)
    sel.select([face(200, 300, 300), face(1500, 100, 60)], SHAPE)
    bystander = face(1500, 100, 60)
    # main face missing for 3 frames: don't swap the small bystander
    for _ in range(3):
        assert sel.select([face(1500, 100, 60)], SHAPE) == []
    # gone for longer than max_gap: whoever's left becomes the main face
    assert sel.select([bystander], SHAPE) == [bystander]


def test_reference_matches_only_that_person():
    me, other = unit(1), unit(2)
    sel = ReferenceSelector(me, threshold=0.35)
    a, b = face(100, 100, 200, mix(me, other, 0.7)), face(900, 100, 200, other)
    assert sel.select([a, b], SHAPE) == [a]


def test_reference_at_most_one_face():
    me, other = unit(1), unit(2)
    sel = ReferenceSelector(me, threshold=0.35)
    a, b = face(100, 100, 200, mix(me, other, 0.6)), face(900, 100, 200, mix(me, other, 0.8))
    assert sel.select([a, b], SHAPE) == [b]


def test_reference_tracks_through_similarity_dip():
    me, other = unit(1), unit(2)
    sel = ReferenceSelector(me, threshold=0.35, fps=30, dip_seconds=1.0)
    stranger = face(1200, 100, 200, other)
    assert len(sel.select([face(100, 100, 200, mix(me, other, 0.7)), stranger], SHAPE)) == 1
    # profile view: similarity 0.2 (below threshold, above floor), same place
    for i in range(30):
        dip = face(100 + i, 100, 200, mix(me, other, 0.2))
        assert sel.select([dip, face(1200, 100, 200, other)], SHAPE) == [dip]
    # longer than dip_seconds: let go
    late = face(130, 100, 200, mix(me, other, 0.2))
    assert sel.select([late], SHAPE) == []


def test_reference_does_not_track_onto_a_stranger():
    me, other = unit(1), unit(2)
    sel = ReferenceSelector(me, threshold=0.35)
    sel.select([face(100, 100, 200, mix(me, other, 0.7))], SHAPE)
    # someone else in the same spot, similarity under the floor
    assert sel.select([face(105, 100, 200, mix(me, other, 0.05))], SHAPE) == []


def test_reference_survives_a_missed_detection():
    me, other = unit(1), unit(2)
    sel = ReferenceSelector(me, threshold=0.35, max_missing=5)
    sel.select([face(100, 100, 200, mix(me, other, 0.7))], SHAPE)
    assert sel.select([], SHAPE) == []
    dip = face(102, 100, 200, mix(me, other, 0.2))
    assert sel.select([dip], SHAPE) == [dip]


def test_make_selector():
    assert isinstance(make_selector("everyone"), EveryoneSelector)
    assert isinstance(make_selector("main"), MainFaceSelector)
    assert isinstance(make_selector("one", unit(1)), CastSelector)
    assert isinstance(make_selector("cast", cast=[(unit(1), "f1")]), CastSelector)


def test_sources_are_set():
    faces = [face(0, 0, 100), face(500, 0, 300)]
    assert {f.source for f in make_selector("everyone", source="me").select(faces, SHAPE)} == {"me"}
    (f,) = make_selector("main", source="me").select([face(0, 0, 100), face(500, 0, 300)], SHAPE)
    assert f.source == "me"


def test_cast_gives_each_person_their_own_face():
    a, b, c = unit(1), unit(2), unit(3)
    sel = CastSelector([(a, "face-for-a"), (b, "face-for-b")], threshold=0.35)
    fa, fb, fc = face(0, 0, 200, mix(a, c, 0.7)), face(600, 0, 200, mix(b, c, 0.7)), face(1200, 0, 200, c)
    picked = sel.select([fa, fb, fc], SHAPE)
    assert {(id(f), f.source) for f in picked} == {(id(fa), "face-for-a"), (id(fb), "face-for-b")}


def test_cast_conflict_goes_to_the_closer_match():
    a, b = unit(1), unit(2)
    sel = CastSelector([(a, "A"), (mix(a, b, 0.6), "B")], threshold=0.35)
    only = face(0, 0, 200, a)  # looks exactly like a, less like the second person
    (f,) = sel.select([only], SHAPE)
    assert f.source == "A"


def test_smoother_steadies_jitter_but_follows_movement():
    sm = KpsSmoother()
    base = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], np.float32)
    f = face(0, 0, 120)
    f.kps = base.copy()
    sm.smooth([f])
    jittery = face(1, 0, 120)
    jittery.kps = base + 2.0  # 2 px of jitter on a 120 px face
    sm.smooth([jittery])
    assert np.abs(jittery.kps - base).max() < 1.0  # mostly averaged away
    moved = face(30, 0, 120)
    moved.kps = base + 30.0  # a real head movement
    sm.smooth([moved])
    assert np.abs(moved.kps - (base + 30.0)).max() < 2.0  # follows it straight away
