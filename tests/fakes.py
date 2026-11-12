"""A stand-in face engine for tests that don't need the real models.

It "detects" two faces at fixed places (a big one on the left, a small one on the
right), gives each a fixed identity, and "swaps" by inverting the pixels in the box.
"""

import numpy as np

from app.faces import Face


def _unit(seed):
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


LEFT_ID = _unit(1)
RIGHT_ID = _unit(2)


class FakeEngine:
    providers = {"det": "cpu", "rec": "cpu", "swap": "cpu"}

    def __init__(self):
        self.detect_calls = 0
        self.swap_calls = 0

    def boxes(self, shape):
        h, w = shape[:2]
        left = np.array([w * 0.10, h * 0.20, w * 0.40, h * 0.80], dtype=np.float32)
        right = np.array([w * 0.65, h * 0.35, w * 0.80, h * 0.65], dtype=np.float32)
        return [(left, LEFT_ID), (right, RIGHT_ID)]

    def detect(self, img, det_size=640):
        self.detect_calls += 1
        faces = []
        for bbox, _ in self.boxes(img.shape):
            kps = np.zeros((5, 2), dtype=np.float32)
            faces.append(Face(bbox=bbox, kps=kps, score=0.9))
        return faces

    def embed(self, img, face):
        for bbox, ident in self.boxes(img.shape):
            if np.allclose(bbox, face.bbox):
                face.embedding = ident.copy()
                return face.embedding
        raise AssertionError("unknown face")

    def latent(self, embedding):
        return embedding.reshape(1, -1)

    def swap(self, img, face, latent):
        self.swap_calls += 1
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        img[y1:y2, x1:x2] = 255 - img[y1:y2, x1:x2]
