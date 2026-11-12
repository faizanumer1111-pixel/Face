"use strict";

// FaceSwap Studio front end. Talks only to the local server it was served from.

const $ = (id) => document.getElementById(id);
const MODE_LABEL = { cast: "Picked people", everyone: "Everyone", main: "Whoever's on screen" };

const S = {
  status: null,
  project: null,
  faces: [], // face library
  fps: 30,
  total: 0,
  pendingIn: null,
  job: null,
  pollTimer: null,
  boxesTimer: null,
  boxesFor: null, // frame the face boxes belong to
  scrubbing: false,
  dragFace: null,
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
  $("busy-text").textContent = text || "Working";
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

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c) n.append(c);
  return n;
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
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${String(s % 60).padStart(2, "0")} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

const faceById = (id) => S.faces.find((f) => f.id === id);
const defaultFace = () => (S.project && faceById(S.project.default_face)) || S.faces[0] || null;

// ------------------------------------------------------------------ status

async function refreshStatus() {
  try {
    S.status = await api("/api/status");
  } catch {
    return;
  }
  const box = $("engine");
  box.classList.toggle("ready", S.status.engine === "ready");
  box.classList.toggle("error", S.status.engine === "error");
  if (S.status.engine === "loading") {
    $("engine-text").textContent = "Loading face models";
    setTimeout(refreshStatus, 1500);
  } else if (S.status.engine === "error") {
    $("engine-text").textContent = "Face models missing: run scripts/setup.sh";
    box.title = S.status.error || "";
  } else {
    const swap = S.status.providers && S.status.providers.swap;
    const where = swap === "coreml" ? "Mac GPU" : swap === "cuda" ? "NVIDIA GPU" : "CPU";
    $("engine-text").textContent = `Ready, swapping on the ${where}`;
    box.title = `Video encoder: ${S.status.encoder}`;
    const f = S.status.features || {};
    const missing = [];
    if (!f.enhance) { $("opt-enhance").disabled = true; missing.push("sharper faces"); }
    if (!f.occlusion) { $("opt-occlusion").disabled = true; missing.push("hands in front"); }
    $("look-note").hidden = !missing.length;
    $("look-note").textContent = missing.length ? `Run scripts/setup.sh again to turn on ${missing.join(" and ")}.` : "";
    scheduleBoxes();
  }
  const sel = $("vr-format");
  if (!sel.options.length && S.status.formats) {
    for (const [k, label] of Object.entries(S.status.formats)) sel.append(el("option", { value: k, text: label }));
    if (S.project) sel.value = S.project.settings.vr_format;
  }
}

// ------------------------------------------------------------- face library

async function loadFaces() {
  try {
    ({ faces: S.faces } = await api("/api/faces"));
  } catch {
    S.faces = [];
  }
  renderTalent();
}

async function addFaces(files) {
  const list = [...files].filter((f) => f.type.startsWith("image/") || /\.heic$/i.test(f.name));
  if (!list.length) return;
  busy(list.length > 1 ? `Finding faces in ${list.length} photos` : "Finding the face in your photo");
  let added = 0;
  for (const file of list) {
    try {
      const form = new FormData();
      form.append("photo", file);
      const r = await api("/api/faces", { method: "POST", form });
      S.faces = r.faces;
      added += r.added.length;
      if (r.added.length > 1) toast(`${file.name}: found ${r.added.length} faces. Delete any you don't need.`);
    } catch (e) {
      toast(`${file.name}: ${e.message}`, true);
    }
  }
  busy(null);
  renderTalent();
  if (S.project) await reloadProject();
  if (added) toast(added === 1 ? "Face added. Drag it onto a person in the video." : `${added} faces added.`);
}

function renderTalent() {
  const ul = $("talent");
  ul.innerHTML = "";
  const def = defaultFace();
  $("board-tip").textContent = S.faces.length
    ? "Drag a face onto a person in the video. The starred face is used wherever you haven't picked one."
    : "Drop in photos of the faces you want to use. Then drag a face onto a person in the video.";
  for (const f of S.faces) {
    const isDefault = def && def.id === f.id;
    const card = el("li", { class: "headshot", draggable: "true", "data-id": f.id },
      el("img", { src: f.crop_url, alt: "" }),
      el("button", {
        class: "star", "aria-pressed": isDefault ? "true" : "false",
        title: isDefault ? "Starred: used wherever you haven't picked a face" : "Star: use this face by default",
        "aria-label": `Use ${f.name} by default`, text: isDefault ? "★" : "☆",
        onclick: (e) => { e.stopPropagation(); setDefaultFace(f.id); },
      }),
      el("button", {
        class: "remove", title: "Delete this face", "aria-label": `Delete ${f.name}`, text: "×",
        onclick: (e) => { e.stopPropagation(); deleteFace(f); },
      }),
      el("span", { class: "name", title: "Double-click to rename", text: f.name, ondblclick: (e) => renameFace(e, f) }),
    );
    card.addEventListener("dragstart", (e) => {
      S.dragFace = f.id;
      card.classList.add("dragging");
      e.dataTransfer.setData("text/x-face-id", f.id);
      e.dataTransfer.effectAllowed = "copy";
      onFaceDragStart();
    });
    card.addEventListener("dragend", () => {
      S.dragFace = null;
      card.classList.remove("dragging");
      onFaceDragEnd();
    });
    ul.appendChild(card);
  }
}

function renameFace(e, f) {
  const span = e.currentTarget;
  const input = el("input", { class: "name", value: f.name, "aria-label": "Face name", maxlength: "40" });
  span.replaceWith(input);
  input.focus();
  input.select();
  const done = async (save) => {
    if (save && input.value.trim() && input.value.trim() !== f.name) {
      await guarded(async () => {
        await api(`/api/faces/${f.id}`, { method: "PATCH", json: { name: input.value.trim() } });
        await loadFaces();
        if (S.project) renderSegments();
        scheduleBoxes(true);
      });
    } else renderTalent();
  };
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") input.blur();
    if (ev.key === "Escape") { input.dataset.cancel = "1"; input.blur(); }
  });
  input.addEventListener("blur", () => done(!input.dataset.cancel));
}

async function deleteFace(f) {
  if (!confirm(`Delete ${f.name}? Segments that use this face will need another one.`)) return;
  await guarded(async () => {
    ({ faces: S.faces } = await api(`/api/faces/${f.id}`, { method: "DELETE" }));
    renderTalent();
    if (S.project) await reloadProject();
  });
}

async function setDefaultFace(id) {
  if (!S.project) {
    toast("Open or add a video first; the star is saved with each project.");
    return;
  }
  await guarded(async () => {
    setProject(await api(`/api/projects/${S.project.id}/settings`, { method: "PATCH", json: { default_face: id } }), true);
    renderTalent();
  });
}

// ---------------------------------------------------------------- projects

async function loadRecent() {
  const { projects } = await api("/api/projects");
  const sel = $("project-select");
  sel.innerHTML = "";
  sel.append(el("option", { value: "", text: projects.length ? "Recent projects" : "No projects yet" }));
  for (const p of projects) sel.append(el("option", { value: p.id, text: `${p.name}, ${p.created.slice(0, 16)}` }));
  if (S.project) sel.value = S.project.id;
}

function setProject(p, keepVideo = false) {
  const prevId = S.project && S.project.id;
  S.project = p;
  try { localStorage.setItem("faceswap.project", p ? p.id : ""); } catch {}
  if (p) {
    S.fps = p.video.fps;
    S.total = p.video.total_frames;
    if (!keepVideo || prevId !== p.id) {
      $("video").src = p.video_url;
      S.pendingIn = null;
      S.boxesFor = null;
    }
    const s = p.settings;
    $("default-mode").value = s.default_mode || "everyone";
    document.querySelector(`input[name=output][value=${s.output || "full"}]`).checked = true;
    $("max-quality").checked = !!s.max_quality;
    $("det-size").value = String(s.det_size || 640);
    $("opt-enhance").value = Math.round((s.enhance ?? 0.6) * 100);
    $("opt-enhance-out").textContent = `${$("opt-enhance").value}%`;
    $("opt-occlusion").checked = !!s.occlusion;
    $("opt-color").checked = !!s.color_match;
    $("opt-steady").checked = !!s.steady;
    if ($("vr-format").options.length) $("vr-format").value = s.vr_format;
  }
  renderAll();
}

async function reloadProject() {
  if (S.project) setProject(await api(`/api/projects/${S.project.id}`), true);
}

async function openProject(id) {
  if (!id) return;
  await guarded(async () => {
    setProject(await api(`/api/projects/${id}`));
    await loadRecent();
    await resumeJob();
  });
}

function newProject() {
  S.project = null;
  try { localStorage.setItem("faceswap.project", ""); } catch {}
  $("video").removeAttribute("src");
  $("video").load();
  $("project-select").value = "";
  renderAll();
}

async function uploadVideo(file) {
  if (!file) return;
  if (S.job && ["queued", "running"].includes(S.job.state)) {
    toast("A render is running. Wait for it or cancel it first.", true);
    return;
  }
  busy("Bringing in your video");
  await guarded(async () => {
    const form = new FormData();
    form.append("video", file);
    const p = await api("/api/projects", { method: "POST", form });
    setProject(p);
    await loadRecent();
    for (const n of p.notes || []) toast(n);
  });
}

async function patchSettings(body) {
  if (!S.project) return;
  try {
    setProject(await api(`/api/projects/${S.project.id}/settings`, { method: "PATCH", json: body }), true);
  } catch (e) {
    toast(e.message, true);
  }
}

// --------------------------------------------------------------- rendering

function renderAll() {
  const p = S.project;
  $("stage-empty").hidden = !!p;
  $("video-wrap").hidden = !p;
  $("controls").hidden = !p;
  $("segments-panel").hidden = !p;
  $("render-panel").hidden = !p;
  if (!p) return;
  const v = p.video;
  const fmt = S.status && S.status.formats ? S.status.formats[p.settings.vr_format] : "";
  $("video-meta").textContent = `${p.name}: ${v.width}×${v.height}, ${v.fps} fps, ${fmtTime(v.duration)}`
    + (v.has_audio ? "" : ", no sound") + (p.settings.vr_format !== "flat" && fmt ? `. ${fmt}` : "");
  renderSegments();
  renderTimeline();
  updateTimecode();
  updateRenderButton();
}

function segmentAt(frame) {
  return (S.project.segments || []).find((s) => s.start <= frame && frame < s.end) || null;
}

function faceChoice(seg) {
  // Face used by an Everyone / Whoever's on screen segment
  const wrap = el("label", { class: "face-select" });
  const current = faceById(seg.face) || defaultFace();
  if (current) wrap.append(el("img", { src: current.crop_url, alt: "" }));
  const sel = el("select", { "aria-label": "Face to put in", "data-field": "face" });
  const def = defaultFace();
  sel.append(el("option", { value: "", text: def ? `Starred face (${def.name})` : "Starred face (add one)" }));
  for (const f of S.faces) sel.append(el("option", { value: f.id, text: f.name }));
  sel.value = seg.face && faceById(seg.face) ? seg.face : "";
  wrap.append(sel);
  return wrap;
}

function renderSegments() {
  const p = S.project;
  const segs = p.segments || [];
  const ol = $("segments");
  ol.innerHTML = "";
  $("no-segments").hidden = segs.length > 0;
  segs.forEach((s, i) => {
    const len = s.end - s.start;
    const modes = el("div", { class: "modes", role: "group", "aria-label": "Who gets swapped" });
    for (const m of ["cast", "everyone", "main"]) {
      modes.append(el("button", { "data-mode": m, "aria-pressed": s.mode === m ? "true" : "false" },
        el("i", { class: `mode-${m}` }), MODE_LABEL[m]));
    }
    const who = el("div", { class: "who" }, modes);
    if (s.mode === "cast") {
      const list = el("div", { class: "cast-list" });
      for (const c of s.cast) {
        const ref = p.references[c.ref];
        const face = faceById(c.face) || defaultFace();
        list.append(el("span", { class: "cast-chip" },
          ref ? el("img", { src: ref.crop_url, alt: "Person in the video" }) : null,
          el("span", { class: "arrow", text: "→" }),
          face ? el("img", { src: face.crop_url, alt: face.name, title: face.name }) : el("span", { text: "?" }),
          el("button", { "data-uncast": c.ref, "aria-label": "Remove this person", title: "Remove", text: "×" })));
      }
      who.append(list);
      if (!s.cast.length) who.append(el("p", { class: "hint", text: "Pause inside this segment and drag a face onto each person to swap." }));
      const strict = el("label", { class: "strict", title: "Lower catches people at more angles; higher avoids look-alikes" },
        "Match", el("input", { type: "range", min: "0.15", max: "0.7", step: "0.01", value: String(s.threshold), "data-field": "threshold" }),
        el("span", { text: Number(s.threshold).toFixed(2) }));
      strict.querySelector("input").addEventListener("input", (e) => { strict.querySelector("span").textContent = Number(e.target.value).toFixed(2); });
      who.append(strict);
    } else {
      who.append(faceChoice(s));
    }
    const row = el("li", { class: "tape-row", "data-id": s.id },
      el("div", { class: `tab mode-${s.mode}`, text: String(i + 1), "aria-label": `Segment ${i + 1}` }),
      el("div", { class: "times" },
        el("label", {}, "Start", el("input", { "data-field": "start", value: s.start_tc, "aria-label": "Start time" })),
        el("label", {}, "End", el("input", { "data-field": "end", value: s.end_tc, "aria-label": "End time" })),
        el("span", { class: "len", text: `${(len / S.fps).toFixed(2)} s, ${len} frames` })),
      who,
      el("div", { class: "row-actions" },
        el("button", { "data-act": "go", text: "Go to" }),
        el("button", { "data-act": "preview", text: "Preview" }),
        el("button", { "data-act": "delete", text: "Delete" })));
    ol.append(row);
  });
  const st = p.stats;
  $("savings").innerHTML = "";
  if (segs.length) {
    $("savings").append("Swapping ", el("strong", { text: st.frames_to_process.toLocaleString() }),
      ` of ${st.total_frames.toLocaleString()} frames. The other ${st.percent_skipped}% are copied straight through.`);
  }
  highlightCurrentRow();
}

function renderTimeline() {
  const wrap = $("tl-segs");
  wrap.innerHTML = "";
  for (const s of S.project.segments || []) {
    wrap.append(el("div", {
      class: `tl-seg mode-${s.mode}`,
      style: `left:${(s.start / S.total) * 100}%;width:${((s.end - s.start) / S.total) * 100}%`,
      title: `${MODE_LABEL[s.mode]}, ${s.start_tc} to ${s.end_tc}`,
    }));
  }
  const tin = $("tl-in");
  tin.hidden = S.pendingIn == null;
  if (S.pendingIn != null) tin.style.left = `${(S.pendingIn / S.total) * 100}%`;
}

function updateRenderButton() {
  const p = S.project;
  const running = S.job && ["queued", "running"].includes(S.job.state);
  const ready = p && S.faces.length && (p.segments || []).length;
  $("btn-render").disabled = !ready || running;
  $("btn-render").title = !p ? "" : !S.faces.length ? "Add a face first" : !(p.segments || []).length ? "Mark at least one segment" : "";
  $("btn-render").parentElement.classList.toggle("recording", !!running);
}

// ------------------------------------------------------------------ player

function currentFrame() {
  return Math.max(0, Math.min(S.total - 1, Math.floor($("video").currentTime * S.fps + 1e-3)));
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
  $("tc-time").textContent = fmtTime(f / S.fps);
  $("tc-frame").textContent = `frame ${f.toLocaleString()} of ${(S.total - 1).toLocaleString()}`;
  $("tl-playhead").style.left = `${((f + 0.5) / S.total) * 100}%`;
  $("timeline").setAttribute("aria-valuenow", String(f));
  $("timeline").setAttribute("aria-valuemax", String(S.total - 1));
  highlightCurrentRow(f);
}

function highlightCurrentRow(f = currentFrame()) {
  const seg = S.project && segmentAt(f);
  for (const row of $("segments").children) row.classList.toggle("current", !!seg && row.dataset.id === seg.id);
}

function togglePlay() {
  const v = $("video");
  if (v.paused) { clearBoxes(); v.play(); } else v.pause();
}

function frameLoop() {
  const v = $("video");
  const tick = () => { updateTimecode(); if (!v.paused) schedule(); };
  const schedule = () => (v.requestVideoFrameCallback ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(tick));
  schedule();
}

function timelineFrame(e) {
  const r = $("timeline").getBoundingClientRect();
  return Math.floor(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * S.total);
}

// -------------------------------------------------------- face boxes / casting

function videoRect() {
  const v = $("video");
  const cw = v.clientWidth;
  const ch = v.clientHeight;
  const { width: vw, height: vh } = S.project.video;
  const scale = Math.min(cw / vw, ch / vh);
  return { x: (cw - vw * scale) / 2, y: (ch - vh * scale) / 2, scale };
}

function clearBoxes() {
  $("overlay").innerHTML = "";
  S.boxesFor = null;
  closePicker();
}

function scheduleBoxes(force = false) {
  clearTimeout(S.boxesTimer);
  if (!S.project || !$("video").paused || !S.status || S.status.engine !== "ready") return;
  S.boxesTimer = setTimeout(() => showBoxes(force), 220);
}

async function showBoxes(force = false) {
  const f = currentFrame();
  if (!force && S.boxesFor === f) return;
  let data;
  try {
    data = await api(`/api/projects/${S.project.id}/frames/${f}/faces?det_size=${$("det-size").value}`);
  } catch {
    return;
  }
  if (currentFrame() !== f || !$("video").paused) return;
  S.boxesFor = f;
  const ov = $("overlay");
  ov.innerHTML = "";
  const r = videoRect();
  for (const face of data.faces) {
    const [x1, y1, x2, y2] = face.bbox;
    const cast = face.cast ? faceById(face.cast) : null;
    const box = el("button", {
      class: "vbox" + (cast ? ` ${data.mode}` : ""),
      style: `left:${r.x + x1 * r.scale}px;top:${r.y + y1 * r.scale}px;width:${(x2 - x1) * r.scale}px;height:${(x2 - x1) * r.scale * ((y2 - y1) / (x2 - x1))}px`,
      "aria-label": cast ? `Person with ${cast.name}'s face. Click to change.` : "Person. Click to choose a face.",
      title: data.segment ? "Drop a face here, or click to choose" : "Mark a segment here first (I and O)",
    });
    if (cast) box.append(el("span", { class: "tag" }, el("img", { src: cast.crop_url, alt: "" }), cast.name));
    box.addEventListener("click", (e) => { e.stopPropagation(); openPicker(box, face.bbox, f, !!data.segment); });
    box.addEventListener("dragover", (e) => { if (S.dragFace) { e.preventDefault(); box.classList.add("drop-ok"); } });
    box.addEventListener("dragleave", () => box.classList.remove("drop-ok"));
    box.addEventListener("drop", (e) => {
      e.preventDefault();
      box.classList.remove("drop-ok");
      const id = e.dataTransfer.getData("text/x-face-id") || S.dragFace;
      if (id) castFace(f, face.bbox, id, !!data.segment);
    });
    ov.append(box);
  }
  if (!data.faces.length && S.dragFace) stageHint("No faces on this frame. Move to a frame where the person is visible.");
}

function stageHint(text) {
  const h = $("stage-hint");
  h.textContent = text || "";
  h.hidden = !text;
}

function onFaceDragStart() {
  if (!S.project) {
    stageHint("Add a video first, then drop faces onto people.");
    return;
  }
  $("video").pause();
  const seg = segmentAt(currentFrame());
  stageHint(seg ? "Drop the face on a person" : "This frame isn't in a segment yet. Mark one with I and O, then drop faces on people.");
  scheduleBoxes();
}

function onFaceDragEnd() {
  setTimeout(() => stageHint(null), 400);
}

async function castFace(frame, bbox, faceId, inSegment) {
  if (!inSegment) {
    toast("Mark a segment around this moment first: press I at the start and O at the end.", true);
    return;
  }
  busy("Casting");
  await guarded(async () => {
    setProject(await api(`/api/projects/${S.project.id}/cast`, { method: "POST", json: { frame, bbox, face: faceId } }), true);
    const face = faceById(faceId);
    toast(`${face ? face.name : "That face"} is on this person for the whole segment.`);
    await showBoxes(true);
  });
}

function openPicker(anchor, bbox, frame, inSegment) {
  const pk = $("picker");
  pk.innerHTML = "";
  if (!S.faces.length) {
    pk.append(el("p", { text: "Add a face first: drop a photo on the faces panel." }));
  } else if (!inSegment) {
    pk.append(el("p", { text: "This frame isn't in a segment. Mark one with I and O first." }));
  } else {
    pk.append(el("p", { text: "Put which face on this person?" }));
    const ul = el("ul");
    for (const f of S.faces) {
      ul.append(el("li", {}, el("button", { onclick: () => { closePicker(); castFace(frame, bbox, f.id, true); } },
        el("img", { src: f.crop_url, alt: "" }), f.name)));
    }
    pk.append(ul);
  }
  const r = anchor.getBoundingClientRect();
  pk.hidden = false;
  const w = pk.offsetWidth;
  pk.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2))}px`;
  pk.style.top = `${Math.min(window.innerHeight - pk.offsetHeight - 8, r.bottom + 34)}px`;
  const first = pk.querySelector("button");
  if (first) first.focus();
}

function closePicker() {
  $("picker").hidden = true;
}

// ---------------------------------------------------------------- segments

function payload(segs) {
  return segs.map((s) => ({ id: s.id, start: s.start, end: s.end, mode: s.mode, face: s.face, cast: s.cast, threshold: s.threshold }));
}

async function saveSegments(segs) {
  setProject(await api(`/api/projects/${S.project.id}/segments`, { method: "PUT", json: { segments: segs } }), true);
  scheduleBoxes(true);
}

function setIn() {
  if (!S.project) return;
  S.pendingIn = currentFrame();
  renderTimeline();
  stageHint(`In point set at ${fmtTime(S.pendingIn / S.fps)}. Move to the end and press O.`);
  setTimeout(() => stageHint(null), 2500);
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
  const segs = payload(S.project.segments || []);
  segs.push({ start: S.pendingIn, end, mode: $("default-mode").value, threshold: 0.35, cast: [] });
  await guarded(async () => {
    await saveSegments(segs);
    S.pendingIn = null;
    renderTimeline();
    if ($("default-mode").value === "cast") toast("Segment added. Now drag a face onto each person to swap.");
  });
}

async function editSegment(id, change) {
  const segs = payload(S.project.segments);
  const s = segs.find((x) => x.id === id);
  if (!s) return;
  Object.assign(s, change);
  await guarded(() => saveSegments(segs));
  renderAll();
}

// ----------------------------------------------------------------- preview

async function preview(frame = currentFrame()) {
  if (!S.faces.length) {
    toast("Add a face first: drop a photo on the faces panel.", true);
    return;
  }
  busy("Swapping this frame");
  await guarded(async () => {
    const r = await api(`/api/projects/${S.project.id}/preview`, { method: "POST", json: { frame, det_size: Number($("det-size").value) } });
    $("cmp-before").src = r.before;
    $("cmp-after").src = r.after;
    let info = `Frame ${frame}, ${MODE_LABEL[r.mode] || r.mode}. Found ${r.faces} face${r.faces === 1 ? "" : "s"}, swapped ${r.swapped}. Took ${(r.ms / 1000).toFixed(1)} s.`;
    if (!r.faces) info += " Nobody to swap on this frame.";
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
  await guarded(async () => {
    const { job } = await api(`/api/projects/${S.project.id}/render`, {
      method: "POST",
      json: { output: document.querySelector("input[name=output]:checked").value, max_quality: $("max-quality").checked, det_size: Number($("det-size").value) },
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
    try { S.job = await api(`/api/jobs/${S.job.id}`); } catch { S.job = null; }
    showJob();
    if (S.job && ["queued", "running"].includes(S.job.state)) pollJob();
    else if (S.job && S.job.state === "done") { toast("Render finished. It's in Rendered videos."); loadOutputs(); }
  }, 700);
}

function showJob() {
  const j = S.job;
  const running = j && ["queued", "running"].includes(j.state);
  $("btn-cancel").hidden = !running;
  $("progress").hidden = !j;
  updateRenderButton();
  if (!j) { $("progress-text").textContent = ""; return; }
  const p = j.progress || {};
  const pct = p.frames_total ? (100 * p.frames_written) / p.frames_total : 0;
  $("progress-bar").style.width = `${j.state === "done" ? 100 : pct}%`;
  let text = "";
  if (j.state === "queued") text = "Waiting to start";
  else if (j.state === "running") {
    if (!p.frames_total) text = "Starting";
    else if (p.frames_swapped < p.swap_total) {
      text = `Swapping frame ${p.frames_swapped.toLocaleString()} of ${p.swap_total.toLocaleString()}`;
      if (p.swap_fps) text += ` at ${p.swap_fps.toFixed(1)} frames a second`;
      if (p.eta_seconds != null && p.frames_swapped > 0) text += `. About ${fmtDuration(p.eta_seconds)} left.`;
    } else text = "Saving the video";
  } else if (j.state === "done") {
    const r = j.result || {};
    text = `Done in ${fmtDuration(r.seconds)}.`;
    if (p.frames_without_target) text += ` ${p.frames_without_target} segment frame(s) had nobody to swap.`;
  } else if (j.state === "cancelled") text = "Render cancelled.";
  else if (j.state === "failed") text = `Render failed: ${j.message}`;
  $("progress-text").textContent = text;
}

async function resumeJob() {
  try {
    const { jobs } = await api("/api/jobs");
    const j = S.project && jobs.find((x) => x.project_id === S.project.id);
    if (j) { S.job = j; showJob(); pollJob(); }
    else if (S.job && S.job.project_id !== (S.project && S.project.id)) { S.job = null; showJob(); }
  } catch {}
}

// ----------------------------------------------------------------- outputs

async function loadOutputs() {
  let outputs = [];
  try { ({ outputs } = await api("/api/outputs")); } catch { return; }
  const ul = $("outputs");
  ul.innerHTML = "";
  $("no-outputs").hidden = outputs.length > 0;
  const mac = S.status && S.status.platform === "darwin";
  const formats = (S.status && S.status.formats) || {};
  for (const o of outputs) {
    const name = encodeURIComponent(o.name);
    const kind = o.mode === "clips" ? "Segments only" : "Full video";
    const fmt = o.format && o.format !== "flat" ? `, ${formats[o.format] || o.format}` : "";
    ul.append(el("li", { class: "out-card" },
      el("img", { src: `/api/outputs/${name}/thumb.jpg`, alt: "", loading: "lazy" }),
      el("div", { class: "out-body" },
        el("span", { class: "out-name", text: o.name }),
        el("span", { class: "out-meta", text: `${kind}${fmt}. ${o.size_mb} MB, ${o.created}` })),
      el("div", { class: "out-actions" },
        el("a", { class: "button", href: `/api/outputs/${name}`, download: o.name, text: "Download" }),
        mac ? el("button", { text: "Reveal in Finder", onclick: () => guarded(() => api(`/api/outputs/${name}/reveal`, { method: "POST" })) }) : null,
        el("button", { class: "quiet", text: "Delete", onclick: async () => {
          if (!confirm(`Delete ${o.name}?`)) return;
          await guarded(() => api(`/api/outputs/${name}`, { method: "DELETE" }));
          loadOutputs();
        } }))));
  }
}

// ------------------------------------------------------------------ wiring

function wireDrop(zone, accept, handler) {
  zone.addEventListener("dragover", (e) => {
    if (S.dragFace || ![...e.dataTransfer.types].includes("Files")) return;
    e.preventDefault();
    zone.classList.add("over");
  });
  zone.addEventListener("dragleave", (e) => { if (!zone.contains(e.relatedTarget)) zone.classList.remove("over"); });
  zone.addEventListener("drop", (e) => {
    if (S.dragFace) return;
    e.preventDefault();
    zone.classList.remove("over");
    const files = [...e.dataTransfer.files].filter(accept);
    if (files.length) handler(files);
  });
}

function wire() {
  const isVideo = (f) => f.type.startsWith("video/") || /\.(mp4|mov|m4v|mkv)$/i.test(f.name);
  const isImage = (f) => f.type.startsWith("image/") || /\.heic$/i.test(f.name);
  wireDrop($("stage-empty"), isVideo, (files) => uploadVideo(files[0]));
  wireDrop($("board"), isImage, addFaces);
  $("file-video").addEventListener("change", (e) => { if (e.target.files.length) uploadVideo(e.target.files[0]); e.target.value = ""; });
  $("file-face").addEventListener("change", (e) => { addFaces(e.target.files); e.target.value = ""; });
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-pick]");
    if (t) $(t.dataset.pick).click();
    if (!e.target.closest("#picker") && !e.target.closest(".vbox")) closePicker();
  });
  // A video dropped anywhere on the stage replaces the current one
  $("stage").addEventListener("dragover", (e) => { if (!S.dragFace && [...e.dataTransfer.types].includes("Files")) e.preventDefault(); });
  $("stage").addEventListener("drop", (e) => {
    if (S.dragFace) return;
    const files = [...e.dataTransfer.files].filter(isVideo);
    if (files.length) { e.preventDefault(); uploadVideo(files[0]); }
  });

  $("project-select").addEventListener("change", (e) => openProject(e.target.value));
  $("btn-new").addEventListener("click", newProject);

  const v = $("video");
  v.addEventListener("loadedmetadata", () => { updateTimecode(); scheduleBoxes(); });
  v.addEventListener("seeked", () => { updateTimecode(); scheduleBoxes(); });
  v.addEventListener("play", () => { $("btn-play").classList.add("playing"); clearBoxes(); frameLoop(); });
  v.addEventListener("pause", () => { $("btn-play").classList.remove("playing"); updateTimecode(); scheduleBoxes(); });
  v.addEventListener("click", togglePlay);

  $("btn-play").addEventListener("click", togglePlay);
  $("btn-back").addEventListener("click", () => seekFrame(currentFrame() - 1));
  $("btn-fwd").addEventListener("click", () => seekFrame(currentFrame() + 1));
  $("btn-in").addEventListener("click", setIn);
  $("btn-out").addEventListener("click", setOut);
  $("btn-preview").addEventListener("click", () => preview());
  $("default-mode").addEventListener("change", (e) => patchSettings({ default_mode: e.target.value }));
  $("vr-format").addEventListener("change", (e) => patchSettings({ vr_format: e.target.value }).then(() => scheduleBoxes(true)));

  const tl = $("timeline");
  tl.addEventListener("pointerdown", (e) => { if (!S.project) return; S.scrubbing = true; tl.setPointerCapture(e.pointerId); seekFrame(timelineFrame(e)); });
  tl.addEventListener("pointermove", (e) => { if (S.scrubbing) seekFrame(timelineFrame(e)); });
  tl.addEventListener("pointerup", () => { S.scrubbing = false; });
  tl.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); e.stopPropagation(); seekFrame(currentFrame() - (e.shiftKey ? 10 : 1)); }
    if (e.key === "ArrowRight") { e.preventDefault(); e.stopPropagation(); seekFrame(currentFrame() + (e.shiftKey ? 10 : 1)); }
  });

  const ol = $("segments");
  ol.addEventListener("change", (e) => {
    const row = e.target.closest(".tape-row");
    const field = e.target.dataset.field;
    if (!row || !field) return;
    const value = field === "threshold" ? Number(e.target.value) : field === "face" ? (e.target.value || null) : e.target.value;
    editSegment(row.dataset.id, { [field]: value });
  });
  ol.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.matches(".times input")) e.target.blur(); });
  ol.addEventListener("click", (e) => {
    const row = e.target.closest(".tape-row");
    if (!row) return;
    const seg = S.project.segments.find((s) => s.id === row.dataset.id);
    const modeBtn = e.target.closest("[data-mode]");
    if (modeBtn && modeBtn.dataset.mode !== seg.mode) {
      editSegment(seg.id, { mode: modeBtn.dataset.mode }).then(() => {
        if (modeBtn.dataset.mode === "cast") toast("Pause inside the segment and drag a face onto each person to swap.");
      });
      return;
    }
    const uncast = e.target.closest("[data-uncast]");
    if (uncast) { editSegment(seg.id, { cast: seg.cast.filter((c) => c.ref !== uncast.dataset.uncast) }); return; }
    const act = e.target.closest("[data-act]");
    if (!act) return;
    if (act.dataset.act === "delete") {
      guarded(() => saveSegments(payload(S.project.segments).filter((x) => x.id !== seg.id)));
    } else if (act.dataset.act === "go") seekFrame(seg.start);
    else if (act.dataset.act === "preview") {
      const f = currentFrame();
      const frame = f >= seg.start && f < seg.end ? f : seg.start;
      seekFrame(frame);
      preview(frame);
    }
  });

  $("btn-render").addEventListener("click", startRender);
  $("btn-cancel").addEventListener("click", cancelRender);
  for (const r of document.querySelectorAll("input[name=output]")) r.addEventListener("change", () => patchSettings({ output: r.value }));
  $("max-quality").addEventListener("change", (e) => patchSettings({ max_quality: e.target.checked }));
  $("det-size").addEventListener("change", (e) => patchSettings({ det_size: Number(e.target.value) }).then(() => scheduleBoxes(true)));
  $("opt-enhance").addEventListener("input", (e) => { $("opt-enhance-out").textContent = `${e.target.value}%`; });
  $("opt-enhance").addEventListener("change", (e) => patchSettings({ enhance: Number(e.target.value) / 100 }));
  $("opt-occlusion").addEventListener("change", (e) => patchSettings({ occlusion: e.target.checked }));
  $("opt-color").addEventListener("change", (e) => patchSettings({ color_match: e.target.checked }));
  $("opt-steady").addEventListener("change", (e) => patchSettings({ steady: e.target.checked }));

  $("preview-close").addEventListener("click", () => $("preview-dialog").close());
  $("cmp-slider").addEventListener("input", (e) => setSlider(e.target.value));
  window.addEventListener("resize", () => { if (S.boxesFor != null) { S.boxesFor = null; scheduleBoxes(); } });

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (!S.project || $("preview-dialog").open) return;
    const step = e.shiftKey ? 10 : 1;
    switch (e.key) {
      case " ": if (tag === "button") return; e.preventDefault(); togglePlay(); break;
      case "i": case "I": setIn(); break;
      case "o": case "O": setOut(); break;
      case ",": case "<": seekFrame(currentFrame() - step); break;
      case ".": case ">": seekFrame(currentFrame() + step); break;
      case "ArrowLeft": e.preventDefault(); seekFrame(currentFrame() - step); break;
      case "ArrowRight": e.preventDefault(); seekFrame(currentFrame() + step); break;
      case "Escape": closePicker(); if (S.pendingIn != null) { S.pendingIn = null; renderTimeline(); } break;
      default: return;
    }
  });
}

async function init() {
  wire();
  await refreshStatus();
  await loadFaces();
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
      renderAll();
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
