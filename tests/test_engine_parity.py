"""Checks the app's face engine against the original InsightFace implementation.

Runs only where the insightface package is installed (it isn't an app dependency):
    pip install insightface==0.7.3 && pytest tests/test_engine_parity.py
"""

from pathlib import Path

import numpy as np
import pytest

insightface = pytest.importorskip("insightface")

from app.config import MODELS_DIR  # noqa: E402
from app.faces import FaceEngine, missing_models  # noqa: E402

pytestmark = pytest.mark.skipif(bool(missing_models(MODELS_DIR)), reason="models not downloaded")


@pytest.fixture(scope="module")
def group_photo():
    import cv2

    path = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
    img = cv2.imread(str(path))
    assert img is not None
    return img


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    root = tmp_path_factory.mktemp("insightface")
    pack = root / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        (pack / name).symlink_to(MODELS_DIR / name)
    fa = FaceAnalysis(name="buffalo_l", root=str(root), allowed_modules=["detection", "recognition"],
                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = get_model(str(MODELS_DIR / "inswapper_128.onnx"), providers=["CPUExecutionProvider"])
    return fa, swapper


@pytest.fixture(scope="module")
def engine():
    return FaceEngine(MODELS_DIR)


def test_detection_matches(group_photo, reference, engine):
    fa, _ = reference
    ref = sorted(fa.get(group_photo), key=lambda f: f.bbox[0])
    ours = sorted(engine.detect(group_photo, 640), key=lambda f: f.bbox[0])
    assert len(ours) == len(ref) > 3
    for a, b in zip(ours, ref):
        assert np.allclose(a.bbox, b.bbox, atol=0.5)
        assert np.allclose(a.kps, b.kps, atol=0.5)


def test_embeddings_match(group_photo, reference, engine):
    fa, _ = reference
    ref = sorted(fa.get(group_photo), key=lambda f: f.bbox[0])
    ours = sorted(engine.detect(group_photo, 640), key=lambda f: f.bbox[0])
    for a, b in zip(ours, ref):
        emb = engine.embed(group_photo, a)
        assert float(np.dot(emb, b.normed_embedding)) > 0.999


def test_swap_matches(group_photo, reference, engine):
    fa, swapper = reference
    faces = sorted(fa.get(group_photo), key=lambda f: f.bbox[0])
    source, target = faces[0], faces[-1]
    expected = swapper.get(group_photo.copy(), target, source, paste_back=True)

    ours_faces = sorted(engine.detect(group_photo, 640), key=lambda f: f.bbox[0])
    src_emb = engine.embed(group_photo, ours_faces[0])
    out = group_photo.copy()
    engine.swap(out, ours_faces[-1], engine.latent(src_emb))

    diff = np.abs(out.astype(np.int16) - expected.astype(np.int16))
    assert diff.mean() < 0.05
    assert diff.max() <= 3
    # and the swap actually changed the face
    assert np.abs(out.astype(np.int16) - group_photo.astype(np.int16)).sum() > 0
