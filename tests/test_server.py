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


def add_face(client, tmp_path, name="Model A.jpg"):
    with open(face_photo(tmp_path), "rb") as f:
        r = client.post("/api/faces", files={"photo": (name, f, "image/jpeg")}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["added"]


def test_full_flow(client, small_video, tmp_path):
    with open(small_video, "rb") as f:
        r = client.post("/api/projects", files={"video": ("my ad.mp4", f, "video/mp4")}, headers=H)
    assert r.status_code == 200, r.text
    p = r.json()
    pid = p["id"]
    assert p["video"]["total_frames"] == 75
    assert p["settings"]["vr_format"] == "flat"

    # browser can seek: range requests work
    r = client.get(p["video_url"], headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100

    # face library: the fake engine finds two faces in any photo, so one photo adds two faces
    added = add_face(client, tmp_path)
    assert [a["name"] for a in added] == ["Model A", "Model A 2"]
    face_a, face_b = added[0]["id"], added[1]["id"]
    assert client.get(added[0]["crop_url"]).status_code == 200
    assert client.patch(f"/api/faces/{face_b}", json={"name": "Model B"}, headers=H).json()["name"] == "Model B"
    assert client.get(f"/api/projects/{pid}").json()["default_face"] in (face_a, face_b)

    faces = client.get(f"/api/projects/{pid}/frames/10/faces").json()["faces"]
    assert len(faces) == 2

    # segments: times as text, overlap merge, savings stats
    segs = [{"start": "0.4", "end": "0.8", "mode": "everyone", "face": face_a},
            {"start": "#15", "end": "#25", "mode": "everyone", "face": face_a},
            {"start": 40, "end": 50, "mode": "cast"}]
    r = client.put(f"/api/projects/{pid}/segments", json={"segments": segs}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["start"], s["end"]) for s in body["segments"]] == [(10, 25), (40, 50)]
    assert body["stats"]["frames_to_process"] == 25
    assert body["segments"][0]["start_tc"] == "0:00.400"

    # rendering is refused until somebody is cast in the "picked people" segment
    r = client.post(f"/api/projects/{pid}/render", json={}, headers=H)
    assert r.status_code == 422 and "nobody is picked" in r.json()["detail"]

    # casting board: drop face A on the left person and face B on the right person
    left_box = max(faces, key=lambda f: f["bbox"][2] - f["bbox"][0])["bbox"]
    right_box = min(faces, key=lambda f: f["bbox"][2] - f["bbox"][0])["bbox"]
    r = client.post(f"/api/projects/{pid}/cast", json={"frame": 45, "bbox": left_box, "face": face_a}, headers=H)
    assert r.status_code == 200, r.text
    r = client.post(f"/api/projects/{pid}/cast", json={"frame": 45, "bbox": right_box, "face": face_b}, headers=H)
    cast = r.json()["segments"][1]["cast"]
    assert [c["face"] for c in cast] == [face_a, face_b]
    # recasting the same person replaces their face instead of adding them twice
    r = client.post(f"/api/projects/{pid}/cast", json={"frame": 46, "bbox": left_box, "face": face_b}, headers=H)
    assert [c["face"] for c in r.json()["segments"][1]["cast"]] == [face_b, face_b]
    labels = client.get(f"/api/projects/{pid}/frames/45/faces").json()
    assert labels["mode"] == "cast" and {f["cast"] for f in labels["faces"]} == {face_b}

    # casting outside a segment explains what to do
    r = client.post(f"/api/projects/{pid}/cast", json={"frame": 70, "bbox": left_box, "face": face_a}, headers=H)
    assert r.status_code == 409 and "segment" in r.json()["detail"]

    # overlapping segments with different modes are refused with a clear message
    bad = client.get(f"/api/projects/{pid}").json()["segments"] + [{"start": 20, "end": 45, "mode": "main"}]
    r = client.put(f"/api/projects/{pid}/segments", json={"segments": bad}, headers=H)
    assert r.status_code == 422 and "overlap" in r.json()["detail"]

    # settings for the new swap options
    r = client.patch(f"/api/projects/{pid}/settings", json={"enhance": 0.3, "occlusion": False, "det_size": 960},
                     headers=H)
    assert r.json()["settings"]["enhance"] == 0.3 and r.json()["settings"]["det_size"] == 960

    # preview a frame inside the cast segment
    r = client.post(f"/api/projects/{pid}/preview", json={"frame": 45}, headers=H)
    assert r.status_code == 200, r.text
    pv = r.json()
    assert pv["faces"] == 2 and pv["swapped"] == 2 and pv["after"].startswith("data:image/jpeg;base64,")

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
    assert client.get(f"/api/outputs/{name}/thumb.jpg").status_code == 200

    # clips only
    r = client.post(f"/api/projects/{pid}/render", json={"output": "clips"}, headers=H)
    job = wait_job(client, r.json()["job"]["id"])
    assert job["state"] == "done" and job["progress"]["frames_total"] == 25

    # deleting a face that's cast blocks the next render with a clear message
    client.delete(f"/api/faces/{face_b}", headers=H)
    r = client.post(f"/api/projects/{pid}/render", json={}, headers=H)
    assert r.status_code == 422 and "deleted" in r.json()["detail"]

    # project reloads with its settings
    p = client.get(f"/api/projects/{pid}").json()
    assert p["settings"]["output"] == "clips"
    assert [x["id"] for x in client.get("/api/projects").json()["projects"]] == [pid]
    assert client.delete(f"/api/outputs/{name}", headers=H).status_code == 200


def test_vr_project_gets_a_preview_copy(client, tmp_path):
    from tests.test_vr import _vr_clip

    clip = _vr_clip(tmp_path, "launch_180_LR.mp4")
    with open(clip, "rb") as f:
        p = client.post("/api/projects", files={"video": ("launch_180_LR.mp4", f, "video/mp4")}, headers=H).json()
    assert p["settings"]["vr_format"] == "vr180_sbs"
    assert any("VR 180" in n for n in p["notes"])
    add_face(client, tmp_path)
    client.put(f"/api/projects/{p['id']}/segments", json={"segments": [{"start": 2, "end": 6}]}, headers=H)
    job = client.post(f"/api/projects/{p['id']}/render", json={}, headers=H).json()["job"]
    job = wait_job(client, job["id"])
    assert job["state"] == "done", job
    assert job["result"]["output"].endswith("_180_LR.mp4")


def test_no_face_in_photo(client, small_video, tmp_path):
    class NoFaces(FakeEngine):
        def detect(self, img, det_size=640):
            return []

    app = create_app(data_dir=tmp_path / "d2", engine_factory=NoFaces)
    with TestClient(app, base_url="http://127.0.0.1:7860") as c:
        with open(small_video, "rb") as f:
            pid = c.post("/api/projects", files={"video": ("a.mp4", f)}, headers=H).json()["id"]
        with open(face_photo(tmp_path), "rb") as f:
            r = c.post("/api/faces", files={"photo": ("me.jpg", f)}, headers=H)
        assert r.status_code == 422
        assert r.json()["detail"] == "No face found in your photo — use a clear, front-facing shot."


def test_bad_paths_are_rejected(client):
    assert client.get("/api/projects/../../etc/passwd").status_code == 404
    assert client.get("/api/outputs/..%2Fsecret.mp4").status_code == 404
    assert client.get("/api/projects/zzz").status_code == 404


def test_render_queue(tmp_path, small_video):
    """Queue the same project several times; each render keeps the settings it was queued with."""
    class Slow(FakeEngine):
        def detect(self, img, det_size=640):
            time.sleep(0.03)
            return super().detect(img, det_size)

    app = create_app(data_dir=tmp_path / "dq", engine_factory=Slow)
    with TestClient(app, base_url="http://127.0.0.1:7860") as c:
        with open(small_video, "rb") as f:
            pid = c.post("/api/projects", files={"video": ("promo.mp4", f)}, headers=H).json()["id"]
        a, b = add_face(c, tmp_path, "Ava.jpg")
        c.put(f"/api/projects/{pid}/segments",
              json={"segments": [{"start": 0, "end": 30, "face": a["id"]}]}, headers=H)
        j1 = c.post(f"/api/projects/{pid}/render", json={"output": "full"}, headers=H).json()["job"]
        # switch the face and the output, then queue the same project again
        c.put(f"/api/projects/{pid}/segments",
              json={"segments": [{"start": 0, "end": 30, "face": b["id"]}]}, headers=H)
        j2 = c.post(f"/api/projects/{pid}/render", json={"output": "clips"}, headers=H).json()["job"]
        j3 = c.post(f"/api/projects/{pid}/render", json={"output": "clips"}, headers=H).json()["job"]
        assert (j1["position"], j2["position"], j3["position"]) == (1, 2, 3)
        assert j1["label"] == "promo.mp4 with Ava" and j2["label"] == "promo.mp4 with Ava 2"

        # adding a new video while renders are running works
        with open(small_video, "rb") as f:
            assert c.post("/api/projects", files={"video": ("next.mp4", f)}, headers=H).status_code == 200

        # reorder: j3 goes ahead of j2, then j2 is taken out of the queue
        listing = c.post(f"/api/jobs/{j3['id']}/move", json={"to": 1}, headers=H).json()["jobs"]
        assert [j["id"] for j in listing[:3]] == [j1["id"], j3["id"], j2["id"]]
        c.post(f"/api/jobs/{j2['id']}/cancel", headers=H)

        done1 = wait_job(c, j1["id"])
        done3 = wait_job(c, j3["id"])
        assert done1["state"] == done3["state"] == "done"
        assert c.get(f"/api/jobs/{j2['id']}").json()["state"] == "cancelled"
        # each kept what it was queued with
        assert done1["result"]["mode"] == "full" and done1["result"]["faces"] == ["Ava"]
        assert done3["result"]["mode"] == "clips" and done3["result"]["faces"] == ["Ava 2"]
        outs = c.get("/api/outputs").json()["outputs"]
        assert len(outs) == 2 and {o["mode"] for o in outs} == {"full", "clips"}

        states = [j["state"] for j in c.get("/api/jobs").json()["jobs"]]
        assert sorted(states) == ["cancelled", "done", "done"]
        assert c.post("/api/jobs/clear", headers=H).json()["jobs"] == []


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
            c.post("/api/faces", files={"photo": ("me.jpg", f)}, headers=H)
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
