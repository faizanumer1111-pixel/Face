import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config
from app.server import create_app
from tests.fakes import FakeEngine

ROOT = Path(__file__).resolve().parent.parent
H = {config.CLIENT_HEADER: "1"}


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data", engine_factory=FakeEngine)
    app.state.engines.get()  # wait for the (fake) models
    with TestClient(app, base_url=f"http://127.0.0.1:{config.PORT}") as c:
        yield c


def face_photo(tmp_path):
    p = tmp_path / "me.jpg"
    img = np.full((400, 400, 3), 180, np.uint8)
    cv2.circle(img, (200, 200), 120, (90, 120, 200), -1)
    cv2.imwrite(str(p), img)
    return p


def wait_job(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise AssertionError("render didn't finish")


def test_rejects_other_host_names(tmp_path):
    app = create_app(data_dir=tmp_path / "data", engine_factory=FakeEngine)
    with TestClient(app, base_url="http://evil.example.com") as c:
        assert c.get("/api/status").status_code == 400
    with TestClient(app, base_url="http://localhost:7860") as c:
        assert c.get("/api/status").status_code == 200


def test_writes_need_the_page_header(client, small_video):
    with open(small_video, "rb") as f:
        r = client.post("/api/projects", files={"video": ("ad.mp4", f, "video/mp4")})
    assert r.status_code == 403
    assert client.get("/api/status").status_code == 200  # reads are fine


def test_full_flow(client, small_video, tmp_path):
    with open(small_video, "rb") as f:
        r = client.post("/api/projects", files={"video": ("my ad.mp4", f, "video/mp4")}, headers=H)
    assert r.status_code == 200, r.text
    p = r.json()
    pid = p["id"]
    assert p["video"]["total_frames"] == 75

    # browser can seek: range requests work
    r = client.get(p["video_url"], headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100

    with open(face_photo(tmp_path), "rb") as f:
        r = client.post(f"/api/projects/{pid}/face", files={"photo": ("me.jpg", f, "image/jpeg")}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["face"]["selected"] == 0  # the biggest face

    r = client.get(f"/api/projects/{pid}/frames/10/faces")
    faces = r.json()["faces"]
    assert len(faces) == 2

    # segments: times as text, overlap merge, savings stats
    segs = [{"start": "0.4", "end": "0.8", "mode": "everyone"}, {"start": "#15", "end": "#25", "mode": "everyone"},
            {"start": 40, "end": 50, "mode": "one"}]
    r = client.put(f"/api/projects/{pid}/segments", json={"segments": segs}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"]) for s in body["segments"]] == [(10, 25), (40, 50)]
    assert body["stats"]["frames_to_process"] == 25
    assert body["segments"][0]["start_tc"] == "0:00.400"

    # rendering is refused until the person for the "one" segment is picked
    r = client.post(f"/api/projects/{pid}/render", json={}, headers=H)
    assert r.status_code == 422 and "nobody is picked" in r.json()["detail"]

    seg_id = body["segments"][1]["id"]
    r = client.post(f"/api/projects/{pid}/references", json={"frame": 45, "bbox": faces[0]["bbox"],
                                                              "segment_id": seg_id}, headers=H)
    assert r.status_code == 200, r.text
    ref = r.json()["ref"]
    assert r.json()["project"]["segments"][1]["ref"] == ref

    # overlapping segments with different modes are refused with a clear message
    bad = body["segments"] + [{"start": 20, "end": 45, "mode": "main"}]
    r = client.put(f"/api/projects/{pid}/segments", json={"segments": bad}, headers=H)
    assert r.status_code == 422 and "overlap" in r.json()["detail"]

    # preview a frame
    r = client.post(f"/api/projects/{pid}/preview", json={"frame": 45, "mode": "one", "ref": ref}, headers=H)
    assert r.status_code == 200, r.text
    pv = r.json()
    assert pv["faces"] == 2 and pv["swapped"] == 1 and pv["after"].startswith("data:image/jpeg;base64,")

    # render the full video
    r = client.post(f"/api/projects/{pid}/render", json={"output": "full"}, headers=H)
    assert r.status_code == 200, r.text
    job = wait_job(client, r.json()["job"]["id"])
    assert job["state"] == "done", job
    assert job["progress"]["frames_swapped"] == 25
    name = job["result"]["output"]
    assert name.startswith("my-ad_swapped_")

    outs = client.get("/api/outputs").json()["outputs"]
    assert outs[0]["name"] == name
    r = client.get(f"/api/outputs/{name}")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert "attachment" in r.headers["content-disposition"]

    # clips only
    r = client.post(f"/api/projects/{pid}/render", json={"output": "clips"}, headers=H)
    job = wait_job(client, r.json()["job"]["id"])
    assert job["state"] == "done" and job["progress"]["frames_total"] == 25

    # project reloads with its settings
    p = client.get(f"/api/projects/{pid}").json()
    assert p["settings"]["output"] == "clips"
    assert [x["id"] for x in client.get("/api/projects").json()["projects"]] == [pid]


def test_no_face_in_photo(client, small_video, tmp_path):
    class NoFaces(FakeEngine):
        def detect(self, img, det_size=640):
            return []

    app = create_app(data_dir=tmp_path / "d2", engine_factory=NoFaces)
    with TestClient(app, base_url="http://127.0.0.1:7860") as c:
        with open(small_video, "rb") as f:
            pid = c.post("/api/projects", files={"video": ("a.mp4", f)}, headers=H).json()["id"]
        with open(face_photo(tmp_path), "rb") as f:
            r = c.post(f"/api/projects/{pid}/face", files={"photo": ("me.jpg", f)}, headers=H)
        assert r.status_code == 422
        assert r.json()["detail"] == "No face found in your photo — use a clear, front-facing shot."


def test_bad_paths_are_rejected(client):
    assert client.get("/api/projects/../../etc/passwd").status_code == 404
    assert client.get("/api/outputs/..%2Fsecret.mp4").status_code == 404
    assert client.get("/api/projects/zzz").status_code == 404


def test_cancel_render(tmp_path, small_video):
    class Slow(FakeEngine):
        def detect(self, img, det_size=640):
            time.sleep(0.05)
            return super().detect(img, det_size)

    app = create_app(data_dir=tmp_path / "d3", engine_factory=Slow)
    with TestClient(app, base_url="http://127.0.0.1:7860") as c:
        with open(small_video, "rb") as f:
            pid = c.post("/api/projects", files={"video": ("a.mp4", f)}, headers=H).json()["id"]
        with open(face_photo(tmp_path), "rb") as f:
            c.post(f"/api/projects/{pid}/face", files={"photo": ("me.jpg", f)}, headers=H)
        c.put(f"/api/projects/{pid}/segments", json={"segments": [{"start": 0, "end": 75}]}, headers=H)
        job = c.post(f"/api/projects/{pid}/render", json={}, headers=H).json()["job"]
        time.sleep(0.5)
        c.post(f"/api/jobs/{job['id']}/cancel", headers=H)
        job = wait_job(c, job["id"])
        assert job["state"] == "cancelled"
        assert c.get("/api/outputs").json()["outputs"] == []


def _lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packet is sent
            ip = s.getsockname()[0]
        except OSError:
            return None
    return None if ip.startswith("127.") else ip


def test_server_only_listens_on_loopback(tmp_path):
    """Start the real server and check other machines can't connect to it."""
    assert config.HOST == "127.0.0.1"
    if socket.socket().connect_ex(("127.0.0.1", config.PORT)) == 0:
        pytest.skip(f"port {config.PORT} is already in use")
    ip = _lan_ip()
    if ip is None:
        pytest.skip("no network interface besides loopback")
    proc = subprocess.Popen([sys.executable, "-m", "app.server", "--no-browser"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", config.PORT)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise AssertionError("server didn't start")
        with socket.socket() as s:
            s.settimeout(2)
            assert s.connect_ex((ip, config.PORT)) != 0, f"server is reachable on {ip}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
