import threading
import time

from app.jobs import JobQueue


def wait_for(cond, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_runs_in_order_one_at_a_time():
    q = JobQueue()
    ran, running, peak = [], [0], [0]
    gate = threading.Event()

    def make(name):
        def run(job):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            gate.wait(5)
            ran.append(name)
            running[0] -= 1
            return {"name": name}
        return run

    jobs = [q.submit("p", make(n), label=n) for n in "abc"]
    wait_for(lambda: jobs[0].state == "running")
    assert [q.position(j) for j in jobs] == [1, 2, 3]
    gate.set()
    wait_for(lambda: all(j.state == "done" for j in jobs))
    assert ran == ["a", "b", "c"] and peak[0] == 1
    # finished ones are listed newest first, after anything still active
    assert [j["label"] for j in q.listing()] == ["c", "b", "a"]


def test_move_and_cancel_waiting_renders():
    q = JobQueue()
    gate = threading.Event()
    ran = []

    def make(name):
        def run(job):
            gate.wait(5)
            ran.append(name)
        return run

    a, b, c, d = (q.submit("p", make(n), label=n) for n in "abcd")
    wait_for(lambda: a.state == "running")
    q.move(d.id, 1)  # d jumps ahead of b and c
    assert [j["label"] for j in q.listing()] == ["a", "d", "b", "c"]
    q.cancel(b.id)
    assert b.state == "cancelled"
    gate.set()
    wait_for(lambda: c.state == "done")
    assert ran == ["a", "d", "c"]
    q.clear_finished()
    assert q.listing() == []


def test_moving_a_running_render_is_refused():
    q = JobQueue()
    gate = threading.Event()
    a = q.submit("p", lambda job: gate.wait(5))
    wait_for(lambda: a.state == "running")
    try:
        q.move(a.id, 2)
        raise AssertionError("should refuse")
    except ValueError:
        pass
    gate.set()


def test_failure_is_reported_and_queue_keeps_going():
    q = JobQueue()

    def boom(job):
        raise RuntimeError("encoder exploded")

    bad = q.submit("p", boom)
    good = q.submit("p", lambda job: {"ok": True})
    wait_for(lambda: good.state == "done")
    assert bad.state == "failed" and "encoder exploded" in bad.message
