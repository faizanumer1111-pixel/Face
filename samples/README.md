Put your own test files here (they're ignored by git):

- `test_video.mp4` (or `.mov`): a short clip, ideally with two people in it
- `test_face.jpg` (or `.png`): a clear, front-facing photo of the face to put in

Then `.venv/bin/python -m pytest tests/test_real_render.py -s` runs the end-to-end checks
with the real face models.
