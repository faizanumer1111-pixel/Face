"use strict";

// FaceSwap Studio front end. Talks only to the local server it was served from.

const $ = (id) => document.getElementById(id);
const MODE_LABEL = { one: "One person", everyone: "Everyone", main: "Whoever's on screen" };

const S = {
  status: null,
  project: null,
  fps: 30,
  total: 0,
  pendingIn: null,
  picking: null, // segment id while picking a person
  job: null,
  pollTimer: null,
  scrubbing: false,
};

// ------------------------------------------------------------------ helpers

async function api(path, { method = "GET", json, form } = {}) {
  const opts = { method, headers: { "X-FaceSwap-Studio": "1" } };
  if (json !== undefined) {
    opts.body = JSON.stringify(json);
    opts.headers["Content-Type"] = "application/json";
  } else if (form) {
    opts.body = form;
  }
  const res = await fetch(path, opts);
  const isJson = (res.headers.get("content-type") || "").includes("application/json");
  const data = isJson ? await res.json() : null;
  if (!res.ok) throw new Error((data && data.detail) || `Something went wrong (${res.status}).`);
  return data;
}

function toast(msg, isError = false) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " error" : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), isError ? 7000 : 3500);
}

function busy(text) {
  $("busy-text").textContent = text || "Working…";
  $("busy").hidden = !text;
}

async function guarded(fn) {
  try {
    return await fn();
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(null);
  }
}

function fmtTime(seconds) {
  const ms = Math.round(seconds * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const r = String(ms % 1000).padStart(3, "0");
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}.${r}` : `${m}:${ss}.${r}`;
}

function fmtDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return "";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

// ------------------------------------------------------------------ status

async function refreshStatus() {
  try {
    S.status = await api("/api/status");
  } catch {
    return;
  }
  const pill = $("engine-status");
  pill.classList.toggle("error", S.status.engine === "error");
  if (S.status.engine === "loading") {
    pill.textContent = "Loading face models…";
    setTimeout(refreshStatus, 1500);
  } else if (S.status.engine === "error") {
    pill.textContent = "Face models missing — run scripts/setup.sh";
    pill.title = S.status.error || "";
  } else {
    const swap = S.status.providers && S.status.providers.swap;
    const where = swap === "coreml" ? "Apple GPU" : swap === "cuda" ? "NVIDIA GPU" : "CPU";
    pill.textContent = `Ready · swapping on ${where}`;
    pill.title = `Encoder: ${S.status.encoder}`;
  }
}

// ---------------------------------------------------------------- projects

async function loadRecent() {
  const { projects } = await api("/api/projects");
  const sel = $("recent-projects");
  sel.innerHTML = '<option value="">Recent projects…</option>';
  for (const p of projects) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = `${p.name} · ${p.created}`;
    sel.appendChild(o);
  }
  if (S.project) sel.value = S.project.id;
}

function setProject(p, { keepVideo = false } = {}) {
  const prevId = S.project && S.project.id;
  S.project = p;
  try { localStorage.setItem("faceswap.project", p ? p.id : ""); } catch {}
  if (!p) {
    renderAll();
    return;
  }
  S.fps = p.video.fps;
  S.total = p.video.total_frames;
  const video = $("video");
  if (!keepVideo || prevId !== p.id) {
    video.src = p.video_url;
    S.pendingIn = null;
  }
  $("default-mode").value = p.settings.default_mode || "everyone";
  document.querySelector(`input[name=output][value=${p.settings.output || "full"}]`).checked = true;
  $("max-quality").checked = !!p.settings.max_quality;
  $("det-size").value = String(p.settings.det_size || 640);
  renderAll();
}

async function openProject(id) {
  if (!id) return;
  await guarded(async () => {
    const p = await api(`/api/projects/${id}`);
    setProject(p);
    await loadRecent();
    await resumeJob();
  });
}

function newProject() {
  S.project = null;
  try { localStorage.setItem("faceswap.project", ""); } catch {}
  $("video").removeAttribute("src");
  $("video").load();
  $("recent-projects").value = "";
  renderAll();
}

// ----------------------------------------------------------------- uploads

async function uploadVideo(file) {
  if (!file) return;
  if (S.job && ["queued", "running"].includes(S.job.state)) {
    toast("Wait for the current render to finish, or cancel it first.", true);
    return;
  }
  busy("Uploading and preparing your video…");
  await guarded(async () => {
    const form = new FormData();
    form.append("video", file);
    const p = await api("/api/projects", { method: "POST", form });
    setProject(p);
    await loadRecent();
    for (const n of p.notes || []) toast(n);
  });
}

async function uploadFace(file) {
  if (!file) return;
  if (!S.project) {
    toast("Add the video first, then the face photo.", true);
    return;
  }
  busy("Finding the face in your photo…");
  await guarded(async () => {
    const form = new FormData();
    form.append("photo", file);
    const p = await api(`/api/projects/${S.project.id}/face`, { method: "POST", form });
    setProject(p, { keepVideo: true });
  });
}

function wireDrop(zoneId, inputId, handler) {
  const zone = $(zoneId);
  const input = $(inputId);
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    if (e.dataTransfer.files.length) handler(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) handler(input.files[0]);
    input.value = "";
  });
}

// --------------------------------------------------------------- rendering

function renderAll() {
  const p = S.project;
  $("editor").hidden = !p;
  $("render-card").hidden = !p;
  renderVideoSummary();
  renderFace();
  if (!p) return;
  renderSegments();
  renderTimeline();
  updateTimecode();
  updateRenderButton();
}

function renderVideoSummary() {
  const el = $("video-summary");
  const p = S.project;
  if (!p) {
    el.innerHTML = 'Drop a video here or <button class="link" data-pick="file-video">choose a file</button>';
    return;
  }
  const v = p.video;
  el.innerHTML = "";
  const name = document.createElement("div");
  name.innerHTML = `<strong></strong>`;
  name.querySelector("strong").textContent = p.name;
  const meta = document.createElement("div");
  meta.textContent = `${v.width}×${v.height} · ${v.fps} fps · ${fmtTime(v.duration)} · ${v.total_frames} frames${v.has_audio ? "" : " · no audio"}`;
  const again = document.createElement("div");
  again.innerHTML = 'Use a different video: <button class="link" data-pick="file-video">choose a file</button>';
  el.append(name, meta);
  if (p.notes && p.notes.length) {
    const ul = document.createElement("ul");
    ul.className = "notes";
    for (const n of p.notes) {
      const li = document.createElement("li");
      li.textContent = n;
      ul.appendChild(li);
    }
    el.appendChild(ul);
  }
  el.appendChild(again);
}

function renderFace() {
  const p = S.project;
  const f = p && p.face;
  $("face-pick").hidden = !f;
  const summary = $("face-summary");
  if (!f) {
    summary.innerHTML = p
      ? 'Drop a photo here or <button class="link" data-pick="file-face">choose a file</button>'
      : '<span class="muted">Add the video first.</span>';
    return;
  }
  summary.innerHTML = 'Use a different photo: <button class="link" data-pick="file-face">choose a file</button>';
  const img = $("face-photo");
  const src = f.photo_url + "?v=" + encodeURIComponent(p.id + f.width + f.faces.length);
  if (img.dataset.src !== src) {
    img.dataset.src = src;
    img.src = src;
  }
  $("face-crop").src = f.crop_url + "&t=" + Date.now();
  $("face-hint").textContent = f.faces.length > 1
    ? `${f.faces.length} faces in this photo. Click a box to use a different one.`
    : "This face will be put into the video.";
  const draw = () => {
    const boxes = $("face-boxes");
    boxes.innerHTML = "";
    const sx = img.clientWidth / f.width;
    const sy = img.clientHeight / f.height;
    f.faces.forEach((face, i) => {
      const [x1, y1, x2, y2] = face.bbox;
      const b = document.createElement("div");
      b.className = "fbox" + (i === f.selected ? " selected" : "");
      Object.assign(b.style, { left: `${x1 * sx}px`, top: `${y1 * sy}px`, width: `${(x2 - x1) * sx}px`, height: `${(y2 - y1) * sy}px` });
      b.title = i === f.selected ? "This face is being used" : "Use this face";
      b.addEventListener("click", () => selectFace(i));
      boxes.appendChild(b);
    });
  };
  if (img.complete && img.naturalWidth) draw();
  img.onload = draw;
}

async function selectFace(i) {
  if (!S.project || S.project.face.selected === i) return;
  await guarded(async () => {
    const p = await api(`/api/projects/${S.project.id}/face/select`, { method: "POST", json: { index: i } });
    setProject(p, { keepVideo: true });
  });
}

function segmentAt(frame) {
  return (S.project.segments || []).find((s) => s.start <= frame && frame < s.end) || null;
}

function renderSegments() {
  const p = S.project;
  const segs = p.segments || [];
  const tbody = $("segments");
  tbody.innerHTML = "";
  $("no-segments").hidden = segs.length > 0;
  segs.forEach((s, i) => {
    const tr = document.createElement("tr");
    tr.dataset.id = s.id;
    const len = s.end - s.start;
    const ref = s.ref && p.references[s.ref];
    tr.innerHTML = `
      <td><i class="swatch mode-${s.mode}"></i>${i + 1}</td>
      <td><input class="tc" data-field="start" aria-label="Start time"></td>
      <td><input class="tc" data-field="end" aria-label="End time"></td>
      <td class="muted">${(len / S.fps).toFixed(2)}s · ${len} frames</td>
      <td><select data-field="mode" aria-label="Who gets swapped">
        <option value="one">One person</option>
        <option value="everyone">Everyone</option>
        <option value="main">Whoever's on screen</option>
      </select></td>
      <td class="who"></td>
      <td class="row-actions">
        <button data-act="go" title="Jump to this segment">Go to</button>
        <button data-act="preview" title="Preview the first frame of this segment">Preview</button>
        <button data-act="delete" title="Delete this segment">Delete</button>
      </td>`;
    tr.querySelector('[data-field=start]').value = s.start_tc;
    tr.querySelector('[data-field=end]').value = s.end_tc;
    tr.querySelector('[data-field=mode]').value = s.mode;
    const who = tr.querySelector(".who");
    if (s.mode === "one") {
      if (ref) {
        const img = document.createElement("img");
        img.src = ref.crop_url;
        img.alt = "Picked person";
        who.appendChild(img);
      } else {
        const w = document.createElement("span");
        w.className = "warn";
        w.textContent = "Nobody picked yet";
        who.appendChild(w);
      }
      const pick = document.createElement("button");
      pick.dataset.act = "pick";
      pick.textContent = ref ? "Change" : "Pick person";
      pick.title = "Pause on a frame that shows them, then click their face";
      who.appendChild(pick);
      const strict = document.createElement("label");
      strict.className = "strict";
      strict.title = "Lower swaps the person in more shots; higher avoids swapping look-alikes";
      strict.innerHTML = `Match <input type="range" min="0.15" max="0.7" step="0.01" data-field="threshold"><span></span>`;
      const range = strict.querySelector("input");
      range.value = s.threshold;
      strict.querySelector("span").textContent = Number(s.threshold).toFixed(2);
      range.addEventListener("input", () => { strict.querySelector("span").textContent = Number(range.value).toFixed(2); });
      who.appendChild(strict);
    } else {
      who.innerHTML = `<span class="muted">${s.mode === "everyone" ? "Every face in frame" : "The biggest face on screen"}</span>`;
    }
    tbody.appendChild(tr);
  });
  const st = p.stats;
  $("savings").textContent = segs.length
    ? `Processing ${st.frames_to_process.toLocaleString()} of ${st.total_frames.toLocaleString()} frames (${st.percent_skipped}% skipped)`
    : "";
  highlightCurrentRow();
}

function renderTimeline() {
  const segs = S.project.segments || [];
  const wrap = $("tl-segs");
  wrap.innerHTML = "";
  for (const s of segs) {
    const d = document.createElement("div");
    d.className = `tl-seg mode-${s.mode}`;
    d.style.left = `${(s.start / S.total) * 100}%`;
    d.style.width = `${((s.end - s.start) / S.total) * 100}%`;
    d.title = `${MODE_LABEL[s.mode]}: ${s.start_tc} – ${s.end_tc}`;
    wrap.appendChild(d);
  }
  const tin = $("tl-in");
  tin.hidden = S.pendingIn == null;
  if (S.pendingIn != null) tin.style.left = `${(S.pendingIn / S.total) * 100}%`;
  $("in-hint").textContent = S.pendingIn != null
    ? `In point set at ${fmtTime(S.pendingIn / S.fps)}. Move to the end and press O.`
    : "";
}

function updateRenderButton() {
  const p = S.project;
  const running = S.job && ["queued", "running"].includes(S.job.state);
  const ready = p && p.face && (p.segments || []).length > 0;
  $("btn-render").disabled = !ready || running;
  $("btn-render").title = !p ? "" : !p.face ? "Add the face photo first" : !(p.segments || []).length ? "Mark at least one segment" : "";
}

// ------------------------------------------------------------------ player

function currentFrame() {
  const v = $("video");
  return Math.max(0, Math.min(S.total - 1, Math.floor(v.currentTime * S.fps + 1e-3)));
}

function seekFrame(f) {
  f = Math.max(0, Math.min(S.total - 1, f));
  const v = $("video");
  v.pause();
  v.currentTime = (f + 0.5) / S.fps;
  clearBoxes();
  updateTimecode(f);
}

function updateTimecode(f = currentFrame()) {
  if (!S.project) return;
  $("timecode").textContent = `${fmtTime(f / S.fps)} · frame ${f} / ${S.total - 1}`;
  $("tl-playhead").style.left = `${((f + 0.5) / S.total) * 100}%`;
  highlightCurrentRow(f);
}

function highlightCurrentRow(f = currentFrame()) {
  const seg = S.project && segmentAt(f);
  for (const tr of $("segments").children) tr.classList.toggle("current", !!seg && tr.dataset.id === seg.id);
}

function togglePlay() {
  const v = $("video");
  if (v.paused) {
    clearBoxes();
    v.play();
  } else {
    v.pause();
  }
}

function frameLoop() {
  const v = $("video");
  const tick = () => {
    updateTimecode();
    if (!v.paused) schedule();
  };
  const schedule = () => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(tick));
  schedule();
}

function timelineFrame(e) {
  const r = $("timeline").getBoundingClientRect();
  const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  return Math.floor(x * S.total);
}

// ---------------------------------------------------------------- segments

function segmentsPayload(segs) {
  return segs.map((s) => ({ id: s.id, start: s.start, end: s.end, mode: s.mode, ref: s.ref, threshold: s.threshold }));
}

async function saveSegments(segs) {
  const p = await api(`/api/projects/${S.project.id}/segments`, { method: "PUT", json: { segments: segs } });
  setProject(p, { keepVideo: true });
}

function setIn() {
  if (!S.project) return;
  S.pendingIn = currentFrame();
  renderTimeline();
}

async function setOut() {
  if (!S.project) return;
  if (S.pendingIn == null) {
    toast("Set the start first: pause where the swap should begin and press I.", true);
    return;
  }
  const end = currentFrame() + 1;
  if (end <= S.pendingIn) {
    toast("The end is before the start. Move past the in point, then press O.", true);
    return;
  }
  const mode = $("default-mode").value;
  const segs = segmentsPayload(S.project.segments || []);
  segs.push({ start: S.pendingIn, end, mode, threshold: 0.35 });
  await guarded(async () => {
    await saveSegments(segs);
    S.pendingIn = null;
    renderTimeline();
  });
}

async function editSegment(id, field, value) {
  const segs = segmentsPayload(S.project.segments);
  const s = segs.find((x) => x.id === id);
  if (!s) return;
  s[field] = field === "threshold" ? Number(value) : value;
  await guarded(() => saveSegments(segs)).then(() => renderAll());
}

async function deleteSegment(id) {
  const segs = segmentsPayload(S.project.segments).filter((x) => x.id !== id);
  await guarded(() => saveSegments(segs));
}

// ----------------------------------------------------------- face picking

function videoRect() {
  const v = $("video");
  const cw = v.clientWidth;
  const ch = v.clientHeight;
  const vw = S.project.video.width;
  const vh = S.project.video.height;
  const scale = Math.min(cw / vw, ch / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, scale };
}

function clearBoxes() {
  $("overlay").innerHTML = "";
}

async function startPick(segId) {
  const seg = S.project.segments.find((s) => s.id === segId);
  const v = $("video");
  v.pause();
  let f = currentFrame();
  if (!seg || f < seg.start || f >= seg.end) {
    // Pick from a frame inside the segment.
    f = seg ? seg.start : f;
    seekFrame(f);
    await new Promise((r) => setTimeout(r, 150));
  }
  busy("Finding faces on this frame…");
  await guarded(async () => {
    const det = $("det-size").value;
    const { faces } = await api(`/api/projects/${S.project.id}/frames/${f}/faces?det_size=${det}`);
    if (!faces.length) {
      toast("No faces on this frame. Move to a frame where the person is clearly visible, then click Pick person again.", true);
      return;
    }
    S.picking = { segId, frame: f, faces };
    drawPickBoxes(faces);
  });
}

function drawPickBoxes(faces) {
  const ov = $("overlay");
  ov.innerHTML = "";
  ov.classList.add("picking");
  $("pick-banner").hidden = false;
  const r = videoRect();
  for (const face of faces) {
    const [x1, y1, x2, y2] = face.bbox;
    const b = document.createElement("div");
    b.className = "vbox";
    Object.assign(b.style, {
      left: `${r.x + x1 * r.scale}px`, top: `${r.y + y1 * r.scale}px`,
      width: `${(x2 - x1) * r.scale}px`, height: `${(y2 - y1) * r.scale}px`,
    });
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      choosePerson(face.bbox);
    });
    ov.appendChild(b);
  }
}

function stopPick() {
  S.picking = null;
  $("overlay").classList.remove("picking");
  $("pick-banner").hidden = true;
  clearBoxes();
}

async function choosePerson(bbox) {
  const pick = S.picking;
  stopPick();
  if (!pick) return;
  busy("Saving this person…");
  await guarded(async () => {
    const res = await api(`/api/projects/${S.project.id}/references`, {
      method: "POST", json: { frame: pick.frame, bbox, segment_id: pick.segId },
    });
    setProject(res.project, { keepVideo: true });
    toast("Got it. Only this person will be swapped in that segment.");
  });
}

// ----------------------------------------------------------------- preview

async function preview(frame = currentFrame()) {
  if (!S.project.face) {
    toast("Add the face photo first.", true);
    return;
  }
  const seg = segmentAt(frame);
  const body = seg
    ? { frame, mode: seg.mode, ref: seg.ref, threshold: seg.threshold }
    : { frame, mode: $("default-mode").value };
  if (body.mode === "one" && !body.ref) {
    toast(seg ? "Pick the person for this segment first." : "Preview One person mode inside a segment where you've picked the person.", true);
    return;
  }
  body.det_size = Number($("det-size").value);
  busy("Swapping this frame…");
  await guarded(async () => {
    const r = await api(`/api/projects/${S.project.id}/preview`, { method: "POST", json: body });
    $("cmp-before").src = r.before;
    $("cmp-after").src = r.after;
    let info = `Frame ${frame} · ${MODE_LABEL[body.mode]} · ${r.faces} face${r.faces === 1 ? "" : "s"} found, ${r.swapped} swapped · ${r.ms} ms`;
    if (r.similarities && r.similarities.length) info += ` · match scores ${r.similarities.join(", ")} (needs ${Number(body.threshold).toFixed(2)})`;
    if (!r.faces) info += " — no faces here";
    $("preview-info").textContent = info;
    setSlider(50);
    $("preview-dialog").showModal();
  });
}

function setSlider(v) {
  $("cmp-slider").value = v;
  $("cmp-after").style.clipPath = `inset(0 0 0 ${v}%)`;
  $("cmp-line").style.left = `${v}%`;
}

// ------------------------------------------------------------------ render

async function startRender() {
  const output = document.querySelector("input[name=output]:checked").value;
  await guarded(async () => {
    const { job } = await api(`/api/projects/${S.project.id}/render`, {
      method: "POST",
      json: { output, max_quality: $("max-quality").checked, det_size: Number($("det-size").value) },
    });
    S.job = job;
    showJob();
    pollJob();
  });
}

async function cancelRender() {
  if (!S.job) return;
  await guarded(async () => {
    S.job = await api(`/api/jobs/${S.job.id}/cancel`, { method: "POST" });
    showJob();
  });
}

function pollJob() {
  clearTimeout(S.pollTimer);
  if (!S.job) return;
  S.pollTimer = setTimeout(async () => {
    try {
      S.job = await api(`/api/jobs/${S.job.id}`);
    } catch {
      S.job = null;
    }
    showJob();
    if (S.job && ["queued", "running"].includes(S.job.state)) {
      pollJob();
    } else if (S.job && S.job.state === "done") {
      toast("Render finished. It's in Finished videos.");
      loadOutputs();
    }
  }, 700);
}

function showJob() {
  const j = S.job;
  const running = j && ["queued", "running"].includes(j.state);
  $("btn-cancel").hidden = !running;
  $("progress").hidden = !j;
  updateRenderButton();
  if (!j) {
    $("progress-text").textContent = "";
    return;
  }
  const p = j.progress || {};
  const pct = p.frames_total ? (100 * p.frames_written) / p.frames_total : 0;
  $("progress-bar").style.width = `${j.state === "done" ? 100 : pct}%`;
  let text = "";
  if (j.state === "queued") text = "Waiting to start…";
  else if (j.state === "running") {
    if (!p.frames_total) text = "Starting…";
    else if (p.frames_swapped < p.swap_total) {
      text = `Swapping faces: frame ${p.frames_swapped.toLocaleString()} of ${p.swap_total.toLocaleString()}`;
      if (p.swap_fps) text += ` · ${p.swap_fps.toFixed(1)} fps`;
      if (p.eta_seconds != null && p.frames_swapped > 0) text += ` · about ${fmtDuration(p.eta_seconds)} left`;
    } else text = "Finishing the video…";
  } else if (j.state === "done") {
    const r = j.result || {};
    text = `Done in ${fmtDuration(r.seconds)}: ${r.output}`;
    if (p.frames_without_target) text += ` · ${p.frames_without_target} segment frame(s) had nobody to swap`;
  } else if (j.state === "cancelled") text = "Render cancelled.";
  else if (j.state === "failed") text = `Render failed: ${j.message}`;
  $("progress-text").textContent = text;
}

async function resumeJob() {
  try {
    const { jobs } = await api("/api/jobs");
    const j = S.project && jobs.find((x) => x.project_id === S.project.id);
    if (j) {
      S.job = j;
      showJob();
      pollJob();
    } else if (S.job && S.job.project_id !== (S.project && S.project.id)) {
      S.job = null;
      showJob();
    }
  } catch {}
}

// ----------------------------------------------------------------- outputs

async function loadOutputs() {
  let outputs = [];
  try {
    ({ outputs } = await api("/api/outputs"));
  } catch {
    return;
  }
  const ul = $("outputs");
  ul.innerHTML = "";
  $("no-outputs").hidden = outputs.length > 0;
  const mac = S.status && S.status.platform === "darwin";
  for (const o of outputs) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = o.name;
    const meta = document.createElement("span");
    meta.className = "muted";
    meta.textContent = `${o.size_mb} MB · ${o.created}${o.mode ? " · " + (o.mode === "clips" ? "clips only" : "full video") : ""}`;
    const spacer = document.createElement("div");
    spacer.className = "spacer";
    const dl = document.createElement("a");
    dl.className = "button";
    dl.href = `/api/outputs/${encodeURIComponent(o.name)}`;
    dl.setAttribute("download", o.name);
    dl.textContent = "Download";
    li.append(name, meta, spacer, dl);
    if (mac) {
      const rv = document.createElement("button");
      rv.textContent = "Reveal in Finder";
      rv.addEventListener("click", () => guarded(() => api(`/api/outputs/${encodeURIComponent(o.name)}/reveal`, { method: "POST" })));
      li.appendChild(rv);
    }
    ul.appendChild(li);
  }
}

// ------------------------------------------------------------------ wiring

function wire() {
  wireDrop("drop-video", "file-video", uploadVideo);
  wireDrop("drop-face", "file-face", uploadFace);
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-pick]");
    if (t) $(t.dataset.pick).click();
  });

  $("recent-projects").addEventListener("change", (e) => openProject(e.target.value));
  $("btn-new").addEventListener("click", newProject);

  const v = $("video");
  v.addEventListener("loadedmetadata", () => updateTimecode());
  v.addEventListener("seeked", () => updateTimecode());
  v.addEventListener("play", () => { $("btn-play").textContent = "Pause"; frameLoop(); });
  v.addEventListener("pause", () => { $("btn-play").textContent = "Play"; updateTimecode(); });
  v.addEventListener("click", () => { if (!S.picking) togglePlay(); });

  $("btn-play").addEventListener("click", togglePlay);
  $("btn-back").addEventListener("click", () => seekFrame(currentFrame() - 1));
  $("btn-fwd").addEventListener("click", () => seekFrame(currentFrame() + 1));
  $("btn-in").addEventListener("click", setIn);
  $("btn-out").addEventListener("click", setOut);
  $("btn-preview").addEventListener("click", () => preview());
  $("default-mode").addEventListener("change", (e) => {
    if (S.project) guarded(() => api(`/api/projects/${S.project.id}/settings`, { method: "PATCH", json: { default_mode: e.target.value } }));
  });

  const tl = $("timeline");
  tl.addEventListener("pointerdown", (e) => {
    if (!S.project) return;
    S.scrubbing = true;
    tl.setPointerCapture(e.pointerId);
    seekFrame(timelineFrame(e));
  });
  tl.addEventListener("pointermove", (e) => { if (S.scrubbing) seekFrame(timelineFrame(e)); });
  tl.addEventListener("pointerup", () => { S.scrubbing = false; });

  const tbody = $("segments");
  tbody.addEventListener("change", (e) => {
    const tr = e.target.closest("tr");
    const field = e.target.dataset.field;
    if (!tr || !field) return;
    if (field === "mode" && e.target.value === "one") {
      editSegment(tr.dataset.id, "mode", "one").then(() => toast("Now pick the person: pause on a frame showing them and click Pick person."));
      return;
    }
    editSegment(tr.dataset.id, field, e.target.value);
  });
  tbody.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.classList.contains("tc")) e.target.blur();
  });
  tbody.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const id = btn.closest("tr").dataset.id;
    const seg = S.project.segments.find((s) => s.id === id);
    if (btn.dataset.act === "delete") deleteSegment(id);
    else if (btn.dataset.act === "go") seekFrame(seg.start);
    else if (btn.dataset.act === "preview") {
      const f = currentFrame();
      const frame = f >= seg.start && f < seg.end ? f : seg.start;
      seekFrame(frame);
      preview(frame);
    } else if (btn.dataset.act === "pick") startPick(id);
  });

  $("btn-render").addEventListener("click", startRender);
  $("btn-cancel").addEventListener("click", cancelRender);
  for (const r of document.querySelectorAll("input[name=output]")) {
    r.addEventListener("change", () => S.project && api(`/api/projects/${S.project.id}/settings`, { method: "PATCH", json: { output: r.value } }).catch(() => {}));
  }
  $("max-quality").addEventListener("change", (e) => S.project && api(`/api/projects/${S.project.id}/settings`, { method: "PATCH", json: { max_quality: e.target.checked } }).catch(() => {}));
  $("det-size").addEventListener("change", (e) => S.project && api(`/api/projects/${S.project.id}/settings`, { method: "PATCH", json: { det_size: Number(e.target.value) } }).catch(() => {}));

  $("preview-close").addEventListener("click", () => $("preview-dialog").close());
  $("cmp-slider").addEventListener("input", (e) => setSlider(e.target.value));

  window.addEventListener("resize", () => { if (S.picking) drawPickBoxes(S.picking.faces); });

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (!S.project || $("preview-dialog").open) return;
    const step = e.shiftKey ? 10 : 1;
    switch (e.key) {
      case " ": e.preventDefault(); togglePlay(); break;
      case "i": case "I": setIn(); break;
      case "o": case "O": setOut(); break;
      case ",": case "<": seekFrame(currentFrame() - step); break;
      case ".": case ">": seekFrame(currentFrame() + step); break;
      case "ArrowLeft": e.preventDefault(); seekFrame(currentFrame() - step); break;
      case "ArrowRight": e.preventDefault(); seekFrame(currentFrame() + step); break;
      case "Escape":
        if (S.picking) stopPick();
        else if (S.pendingIn != null) { S.pendingIn = null; renderTimeline(); }
        break;
      default: return;
    }
  });
}

async function init() {
  wire();
  await refreshStatus();
  await loadOutputs();
  await guarded(loadRecent);
  let last = "";
  try { last = localStorage.getItem("faceswap.project") || ""; } catch {}
  if (last) {
    try {
      setProject(await api(`/api/projects/${last}`));
      await loadRecent();
      await resumeJob();
    } catch {
      renderAll();  // that project was deleted
    }
  } else renderAll();
  if (!S.project) {
    try {
      const { jobs } = await api("/api/jobs");
      if (jobs.length) await openProject(jobs[0].project_id);
    } catch {}
  }
}

init();
