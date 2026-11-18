/* Lead Role studio client. No framework: one state object, small render functions. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  const S = {
    config: null,
    jobId: null, jobPromise: null,
    videoFile: null, videoUrl: null, playable: false, duration: 0, fps: 30,
    meta: null, uploading: false, uploadPct: 0,
    faces: [], activeFace: "a", assign: {},   // assign: face id -> "largest" | "all" | "photo" | person id from a scan
    targets: {},                               // face id -> {photos, usable, busy, seq} photo of the person it replaces
    pickFor: null, camFor: null,
    inT: 0, outT: 0, t: 0,
    people: null, scanning: false,
    tier: localStorage.getItem("lr.tier") || "fast",
    frame: { aspect: "original", mode: "follow", cx: 0.5, cy: 0.5 },   // reframe (crop) settings
    lastCrop: null,
    previewing: false,
    mode: "source",            // source | preview | result
    takes: [], take: null, comparing: false, comparePct: 50,
    pollTimer: null,
  };

  // ------------------------------------------------------------------ helpers
  function tc(t) {
    t = Math.max(0, t || 0);
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
    const cs = Math.floor((t * 100) % 100);
    const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0"), ff = String(cs).padStart(2, "0");
    return h ? `${h}:${mm}:${ss}.${ff}` : `${mm}:${ss}.${ff}`;
  }
  function secs(t) {
    if (t == null || !isFinite(t)) return "-";
    t = Math.round(t);
    return t >= 60 ? `${Math.floor(t / 60)}m ${String(t % 60).padStart(2, "0")}s` : `${t}s`;
  }
  let toastTimer;
  function toast(msg, err = false, ms = 4500) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("err", err);
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), ms);
  }
  function busy(btn, on) {
    btn.classList.toggle("busy", on);
    btn.setAttribute("aria-busy", on ? "true" : "false");
  }

  async function api(method, url, body) {
    const opts = { method, headers: {} };
    if (body instanceof FormData) opts.body = body;
    else if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    let res;
    try { res = await fetch(url, opts); } catch { throw new Error("Can't reach the studio server. Check your connection and try again."); }
    let data = null;
    try { data = await res.json(); } catch { /* empty body */ }
    if (!res.ok) {
      let msg = data && (data.error || data.detail);
      if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
      throw new Error(msg || `Request failed (${res.status})`);
    }
    return data;
  }
  // Long jobs come back as {pending: token}; keep asking until they finish.
  async function task(method, url, body) {
    let r = await api(method, url, body);
    while (r && r.pending) r = await api("GET", `/api/tasks/${r.pending}`);
    return r.result;
  }
  function upload(url, form, onPct) {
    return new Promise((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open("PUT", url);
      x.upload.onprogress = (e) => e.lengthComputable && onPct(e.loaded / e.total);
      x.onload = () => {
        let data = null;
        try { data = JSON.parse(x.responseText); } catch { /* not json */ }
        if (x.status >= 200 && x.status < 300) resolve(data);
        else reject(new Error((data && (data.error || data.detail)) || `Upload failed (${x.status})`));
      };
      x.onerror = () => reject(new Error("Upload interrupted. Check your connection and drop the video again."));
      x.send(form);
    });
  }
  function ensureJob() {
    if (!S.jobPromise) {
      S.jobPromise = api("POST", "/api/jobs").then((r) => (S.jobId = r.job_id))
        .catch((e) => { S.jobPromise = null; throw e; });
    }
    return S.jobPromise;
  }
  const isVideo = (f) => (f.type || "").startsWith("video/") || /\.(mp4|mov|m4v|mkv|webm|avi)$/i.test(f.name || "");
  const isImage = (f) => (f.type || "").startsWith("image/") || /\.(jpe?g|png|webp|bmp|heic)$/i.test(f.name || "");

  // ------------------------------------------------------------------ config
  async function loadConfig() {
    try {
      S.config = await api("GET", "/api/config");
    } catch (e) {
      toast(e.message, true);
      return;
    }
    $("runtime").textContent = `Face work runs on ${S.config.accelerator}`;
    $("takesHint").textContent = S.config.backend === "modal"
      ? "Takes render at the same time on separate GPUs. Change anything and press Render again to start the next one."
      : "Takes render one after another on this computer. Change anything and press Render again to queue the next one.";
    if (!S.config.tiers.some((t) => t.key === S.tier)) S.tier = S.config.tiers[0].key;
    renderTiers();
    renderFormat();
    update();
  }

  // ------------------------------------------------------------------ 1. faces
  // Each face set (A-D) is one person you bring, with 1-5 photos of them.
  const SLOTS = () => (S.config && S.config.face_slots) || "abcd";
  const maxPhotos = () => (S.config && S.config.max_photos) || 5;
  const faceName = (f) => `Face ${(f.id || f).toUpperCase()}`;
  const roleVar = (id) => `var(--role-${id})`;
  const getFace = (id) => S.faces.find((f) => f.id === id);

  function addFaceSet(pick) {
    const used = new Set(S.faces.map((f) => f.id));
    const id = [...SLOTS()].find((c) => !used.has(c));
    if (!id) return toast(`Up to ${SLOTS().length} faces can be swapped in one render.`, true);
    S.faces.push({ id, photos: [], usable: 0, busy: false, seq: 0 });
    S.activeFace = id;
    S.assign[id] = "largest";  // simplest default: A takes the biggest face, B the next biggest...
    if (S.faces.length > 1) {  // "Everyone" only works with a single face
      for (const k of Object.keys(S.assign)) if (S.assign[k] === "all") S.assign[k] = "largest";
    }
    renderFaces();
    renderWho();
    renderCast();
    update();
    if (pick) pickPhotos(id);
  }

  function removeFaceSet(id) {
    const f = getFace(id);
    if (!f || S.faces.length < 2) return;
    f.photos.forEach((p) => URL.revokeObjectURL(p.url));
    S.faces = S.faces.filter((x) => x !== f);
    delete S.assign[id];
    delete S.targets[id];
    if (S.activeFace === id) S.activeFace = S.faces[0].id;
    if (S.jobId) api("DELETE", `/api/jobs/${S.jobId}/faces/${id}`).catch(() => {});
    renderFaces();
    renderWho();
    renderCast();
    update();
  }

  function pickPhotos(id) {
    S.pickFor = id;
    $("faceInput").click();
  }

  async function downscale(file, max = 1600) {
    try {
      const bmp = await createImageBitmap(file);
      const k = Math.min(1, max / Math.max(bmp.width, bmp.height));
      const c = document.createElement("canvas");
      c.width = Math.round(bmp.width * k);
      c.height = Math.round(bmp.height * k);
      c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
      return await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
    } catch {
      return file; // let the server try (e.g. formats this browser can't decode)
    }
  }

  async function addPhotos(files, id) {
    const f = getFace(id) || getFace(S.activeFace) || S.faces[0];
    const imgs = [...files].filter(isImage);
    if (!imgs.length) return toast("Those aren't images. Use a JPG or PNG photo of a face.", true);
    const room = maxPhotos() - f.photos.length;
    if (room <= 0) return toast(`${faceName(f)} already has ${maxPhotos()} photos. Remove one to add another.`, true);
    if (imgs.length > room) toast(`Only the first ${room} photo${room > 1 ? "s were" : " was"} added to ${faceName(f)} (limit ${maxPhotos()}).`);
    for (const file of imgs.slice(0, room)) {
      const blob = await downscale(file);
      f.photos.push({ blob, url: URL.createObjectURL(blob), state: "busy", thumb: null, error: null });
    }
    S.activeFace = f.id;
    renderFaces();
    syncFaces(f);
    warm();
  }

  async function syncFaces(f) {
    const seq = ++f.seq;
    if (!f.photos.length) {
      f.usable = 0;
      f.busy = false;
      renderFaces();
      update();
      if (S.jobId) api("DELETE", `/api/jobs/${S.jobId}/faces/${f.id}`).catch(() => {});
      return;
    }
    f.busy = true;
    f.photos.forEach((p) => (p.state = "busy"));
    renderFaces();
    update();
    try {
      const id = await ensureJob();
      const form = new FormData();
      f.photos.forEach((p, i) => form.append("files", p.blob, `${i}.jpg`));
      const r = await task("POST", `/api/jobs/${id}/faces/${f.id}`, form);
      if (seq !== f.seq || getFace(f.id) !== f) return; // superseded or removed meanwhile
      r.photos.forEach((res, i) => {
        const p = f.photos[i];
        if (!p) return;
        p.state = res.ok ? "ok" : "bad";
        p.thumb = res.thumb || null;
        p.error = res.error || null;
      });
      f.usable = r.usable;
      if (!r.usable) toast(`No usable face found for ${faceName(f)}. Try a clear, front-facing photo.`, true);
    } catch (e) {
      if (seq !== f.seq) return;
      f.photos.forEach((p) => (p.state = "bad"));
      f.usable = 0;
      toast(e.message, true);
    }
    f.busy = false;
    renderFaces();
    update();
  }

  function renderFaces() {
    const box = $("faceSets");
    box.textContent = "";
    const multi = S.faces.length > 1;
    S.faces.forEach((f) => {
      const set = el("div", "face-set");
      set.dataset.face = f.id;
      set.style.setProperty("--role", roleVar(f.id));

      const head = el("div", "face-set-head");
      head.append(el("span", "role-tag", f.id.toUpperCase()), el("span", "face-set-name", faceName(f)));
      const target = el("span", "face-set-target");
      const pid = S.assign[f.id];
      const person = pid && S.people && S.people.find((p) => p.id === pid);
      const tphoto = S.targets[f.id] && S.targets[f.id].photos.find((x) => x.thumb);
      if (pid === "all") target.textContent = "replaces everyone";
      else if (pid === "largest") target.textContent = S.faces.indexOf(f) ? `replaces the #${S.faces.indexOf(f) + 1} biggest face` : "replaces the biggest face";
      else if (pid === "photo" && tphoto) { const img = el("img"); img.src = tphoto.thumb; img.alt = ""; target.append("replaces", img); }
      else if (person) {
        const img = el("img");
        img.src = person.thumb;
        img.alt = "";
        target.append("replaces", img);
      }
      head.append(target);
      if (multi) {
        const rm = el("button", "face-set-remove", "×");
        rm.type = "button";
        rm.title = `Remove ${faceName(f)}`;
        rm.setAttribute("aria-label", `Remove ${faceName(f)}`);
        rm.onclick = () => removeFaceSet(f.id);
        head.append(rm);
      }

      const drop = el("div", `key-surface face-drop${f.photos.length ? " has-faces" : ""}`);
      drop.tabIndex = 0;
      drop.setAttribute("role", "button");
      drop.setAttribute("aria-label", f.photos.length ? `Add a photo to ${faceName(f)}` : `Choose a photo for ${faceName(f)}`);
      for (const c of ["tl", "tr", "bl", "br"]) drop.append(el("span", `marker ${c}`));
      const copy = el("span", "key-copy");
      const title = f.photos.length ? "Add another angle" : multi ? `Drop a photo for ${faceName(f)}` : "Drop a photo of your face";
      copy.append(el("strong", "", title), el("span", "", "or click to choose one"));
      drop.append(copy);
      drop.onclick = () => pickPhotos(f.id);
      drop.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickPhotos(f.id); }
      };
      drop.hidden = f.photos.length >= maxPhotos();

      const list = el("ul", "face-list");
      list.setAttribute("aria-live", "polite");
      f.photos.forEach((p, i) => {
        const li = el("li", `face-item ${p.state}`);
        const img = el("img");
        img.src = p.thumb || p.url;
        img.alt = `${faceName(f)}, photo ${i + 1}`;
        li.append(img);
        if (p.state !== "busy") li.append(el("span", "face-state", p.state === "ok" ? "✓" : "!"));
        const rm = el("button", "remove", "×");
        rm.type = "button";
        rm.setAttribute("aria-label", `Remove ${faceName(f)} photo ${i + 1}`);
        rm.onclick = () => {
          URL.revokeObjectURL(p.url);
          f.photos.splice(i, 1);
          renderFaces();
          syncFaces(f);
        };
        li.append(rm);
        if (p.error) li.title = p.error;
        list.append(li);
      });
      const err = f.photos.find((p) => p.error);
      if (err) list.append(el("li", "face-note", err.error));

      const cam = el("button", "btn ghost", "Use camera");
      cam.type = "button";
      cam.onclick = () => openCamera(f.id);
      set.append(head, drop, list, cam);
      box.append(set);
    });
    $("addSetBtn").hidden = S.faces.length >= SLOTS().length;
  }

  // camera
  let camStream = null;
  async function openCamera(id) {
    S.camFor = id;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return toast("This browser can't use the camera here. Upload a photo instead.", true);
    }
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } }, audio: false });
    } catch {
      return toast("Camera access was blocked. Allow it in your browser settings or upload a photo.", true);
    }
    $("camVideo").srcObject = camStream;
    $("h-camera").textContent = S.faces.length > 1 ? `Take a photo for ${faceName(id)}` : "Take a photo";
    $("camera").hidden = false;
    $("camShoot").focus();
  }
  function closeCamera() {
    if (camStream) camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
    $("camera").hidden = true;
  }
  async function shoot() {
    const v = $("camVideo");
    if (!v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
    closeCamera();
    addPhotos([new File([blob], "camera.jpg", { type: "image/jpeg" })], S.camFor);
  }

  // ------------------------------------------------------------------ 2. video
  const video = $("video");

  async function setVideo(file) {
    if (!isVideo(file)) return toast("That isn't a video file. Use an MP4 or MOV.", true);
    if (S.uploading) return toast("Wait for the current upload to finish.", true);
    if (S.videoUrl) URL.revokeObjectURL(S.videoUrl);
    const fresh = S.takes.length > 0;  // a running take still reads the old video: give this one its own workspace
    Object.assign(S, {
      videoFile: file, videoUrl: URL.createObjectURL(file), playable: false, meta: null,
      people: null, duration: 0, inT: 0, outT: 0, t: 0, filmstripDone: false,
    });
    if (S.mode === "result") S.take = null;
    if (fresh) { S.jobId = null; S.jobPromise = null; }
    $("filmstrip").getContext("2d").clearRect(0, 0, $("filmstrip").width, $("filmstrip").height);
    $("scanBtn").textContent = "Find people";
    showSource();
    renderCast();
    renderWho();
    renderTakes();
    $("renderPanel").hidden = true;
    $("videoDrop").hidden = true;
    video.hidden = false;
    video.src = S.videoUrl;
    warm();
    update();

    S.uploading = true;
    $("uploadLine").hidden = false;
    setUploadPct(0);
    try {
      const id = await ensureJob();
      if (fresh) {  // new workspace: send the faces (and photos of people to replace) again
        S.faces.filter((f) => f.photos.length).forEach((f) => syncFaces(f));
        Object.keys(S.targets).filter((k) => S.targets[k].photos.length && S.faces.some((f) => f.id === k)).forEach((k) => syncTarget(k));
      }
      const form = new FormData();
      form.append("file", file, file.name || "video.mp4");
      S.meta = await upload(`/api/jobs/${id}/video`, form, setUploadPct);
      S.fps = S.meta.fps || 30;
      renderFormat();
      if (!S.playable) {
        // the browser can't decode this format: drive the timeline from the server's frames
        setDuration(S.meta.duration);
        video.hidden = true;
        $("posterFallback").hidden = false;
        showFallbackFrame(0);
        toast("Your browser can't play this format, but you can still trim, preview and render it.");
        buildFilmstrip();
      } else if (Math.abs(S.meta.duration - S.duration) > 0.25) {
        setDuration(S.meta.duration);
        buildFilmstrip();
      }
    } catch (e) {
      toast(e.message, true, 7000);
      S.meta = null;
    } finally {
      S.uploading = false;
      $("uploadLine").hidden = true;
      update();
    }
  }
  function setUploadPct(p) {
    S.uploadPct = p;
    $("uploadLine").firstElementChild.style.width = `${Math.round(p * 100)}%`;
    if (S.uploading) $("whyDisabled").textContent = `Uploading video ${Math.round(p * 100)}%`;
  }
  function setDuration(d) {
    renderFormat();
    const keepSel = S.outT > 0 && S.outT <= d;
    S.duration = d;
    if (!keepSel) { S.inT = 0; S.outT = Math.min(d, maxSec()); }
    if (S.outT - S.inT > maxSec()) S.outT = S.inT + maxSec();
    layoutTimeline();
    if (!keepSel && d > maxSec()) toast(`This video is ${tc(d)} long. One render covers up to ${Math.round(maxSec() / 60)} minutes, so the first ${Math.round(maxSec() / 60)} minutes are selected. Move the section or render the rest as more takes.`, false, 9000);
    $("tcDur").textContent = tc(d);
    update();
  }

  video.addEventListener("loadedmetadata", () => {
    if (S.mode !== "source") return;
    S.playable = true;
    $("posterFallback").hidden = true;
    if (!S.duration) setDuration(video.duration);
    if (!S.filmstripDone) buildFilmstrip();
  });
  video.addEventListener("error", () => {
    if (S.mode === "source") S.playable = false;
  });
  video.addEventListener("play", () => $("playBtn").classList.add("playing"));
  video.addEventListener("pause", () => $("playBtn").classList.remove("playing"));

  function curT() { return S.mode === "source" && S.playable ? video.currentTime : S.t; }
  function seek(t) {
    t = Math.min(Math.max(0, t), S.duration || 0);
    S.t = t;
    if (S.mode !== "source") showSource();
    if (S.playable) video.currentTime = t;
    else showFallbackFrame(t);
    layoutTimeline();
  }
  let fallbackTimer;
  function showFallbackFrame(t) {
    if (!S.jobId || !S.meta) return;
    clearTimeout(fallbackTimer);
    fallbackTimer = setTimeout(() => {
      $("posterFallback").src = `/api/jobs/${S.jobId}/frame?t=${t.toFixed(3)}&w=1280`;
    }, 120);
    $("tcNow").textContent = tc(t);
  }
  function togglePlay() {
    if (S.mode === "result") return video.paused ? video.play() : video.pause();
    if (S.mode !== "source") showSource();
    if (!S.playable) return toast("This format can't play in your browser. Use the timeline to move around.");
    if (video.paused) {
      if (video.currentTime < S.inT || video.currentTime >= S.outT - 0.05) video.currentTime = S.inT;
      video.play();
    } else video.pause();
  }
  function step(frames) {
    if (S.mode === "source" && S.playable) video.pause();
    seek(curT() + frames / (S.fps || 30));
  }

  // tick: keep the playhead smooth and loop the section
  function tick() {
    if (S.mode === "source" && S.playable && S.duration) {
      const t = video.currentTime;
      if (!video.paused && $("loopSel").checked && t >= S.outT) video.currentTime = S.inT;
      S.t = t;
      $("playhead").style.left = `${(t / S.duration) * 100}%`;
      $("tcNow").textContent = tc(t);
    }
    if (S.comparing) syncCompare();
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ timeline
  const tl = $("timeline");
  const minSel = () => Math.min(0.5, Math.max(2 / (S.fps || 30), 0.1), (S.duration || 1) / 2);
  function layoutTimeline() {
    const d = S.duration || 1;
    tl.classList.toggle("empty", !S.duration);
    const a = (S.inT / d) * 100, b = (S.outT / d) * 100;
    $("sel").style.left = `${a}%`;
    $("sel").style.width = `${Math.max(0, b - a)}%`;
    $("maskL").style.width = `${a}%`;
    $("maskR").style.width = `${100 - b}%`;
    $("playhead").style.left = `${(curT() / d) * 100}%`;
    $("hIn").setAttribute("aria-valuetext", tc(S.inT));
    $("hOut").setAttribute("aria-valuetext", tc(S.outT));
    const len = S.outT - S.inT;
    if (document.activeElement !== $("inField")) $("inField").value = S.duration ? tc(S.inT).replace(/^00:/, "0:") : "";
    if (document.activeElement !== $("outField")) $("outField").value = S.duration ? tc(S.outT).replace(/^00:/, "0:") : "";
    const whole = S.duration && S.inT === 0 && S.outT === S.duration;
    $("wholeBtn").hidden = !S.duration || !!whole || S.duration > maxSec();
    $("selReadout").innerHTML = S.duration
      ? (whole ? `Rendering the whole video (${len.toFixed(1)} s). To render part of it, type times above or drag the green handles.`
        : `Rendering <b>${tc(S.inT)}</b> to <b>${tc(S.outT)}</b>, ${len.toFixed(1)} s of ${tc(S.duration)}.`)
      : "Drop a video to choose the section you want.";
  }
  function xToT(clientX) {
    const r = tl.getBoundingClientRect();
    return Math.min(Math.max(0, (clientX - r.left) / r.width), 1) * S.duration;
  }
  const maxSec = () => (S.config && S.config.max_seconds) || 1200;
  function setIn(t, strict) {
    t = Math.max(0, t);
    if (strict && t > S.outT - minSel()) {
      toast("The start has to come before the end. Move the playhead earlier, or set the end first.", true);
      return false;
    }
    S.inT = Math.min(t, S.outT - minSel());
    if (S.outT - S.inT > maxSec()) S.outT = Math.min(S.duration, S.inT + maxSec());  // the end follows the cap
    layoutTimeline();
    return true;
  }
  function setOut(t, strict) {
    t = Math.min(S.duration, t);
    if (strict && t < S.inT + minSel()) {
      toast(S.inT === 0 ? "Play or drag to where the section should end, then press End here."
        : "The end has to come after the start. Move the playhead later, or set the start first.", true);
      return false;
    }
    S.outT = Math.max(t, S.inT + minSel());
    if (S.outT - S.inT > maxSec()) {
      S.inT = Math.max(0, S.outT - maxSec());  // the start follows the cap
      toast(`One render covers up to ${Math.round(maxSec() / 60)} minutes, so the start moved to ${tc(S.inT)}. Render the rest as another take.`, false, 6000);
    }
    layoutTimeline();
    return true;
  }
  function wholeVideo() {
    S.inT = 0;
    S.outT = Math.min(S.duration, maxSec());
    layoutTimeline();
    if (S.duration > maxSec()) toast(`The first ${Math.round(maxSec() / 60)} minutes are selected. One render covers up to ${Math.round(maxSec() / 60)} minutes; render the rest as another take.`, false, 7000);
  }
  // "12", "12.5", "1:05", "1:05.5", "01:02:03" -> seconds
  function parseTime(str) {
    const parts = String(str).trim().replace(",", ".").split(":").map(Number);
    if (!parts.length || parts.some((n) => !isFinite(n) || n < 0)) return null;
    return parts.reduce((acc, n) => acc * 60 + n, 0);
  }
  function bindTimeField(input, apply) {
    const commit = () => {
      const t = parseTime(input.value);
      if (t == null) { toast("Type a time like 12.5 or 1:05.", true); layoutTimeline(); return; }
      if (!apply(t, true)) layoutTimeline();
      else seek(input === $("inField") ? S.inT : S.outT);
    };
    input.addEventListener("change", commit);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); input.blur(); } e.stopPropagation(); });
  }

  function dragger(target, onMove) {
    target.addEventListener("pointerdown", (e) => {
      if (!S.duration) return;
      e.preventDefault();
      e.stopPropagation();
      target.setPointerCapture(e.pointerId);
      onMove(e);
      const move = (ev) => onMove(ev);
      const up = () => {
        target.removeEventListener("pointermove", move);
        target.removeEventListener("pointerup", up);
        target.removeEventListener("pointercancel", up);
      };
      target.addEventListener("pointermove", move);
      target.addEventListener("pointerup", up);
      target.addEventListener("pointercancel", up);
    });
  }
  dragger($("hIn"), (e) => { setIn(xToT(e.clientX)); seek(S.inT); });
  dragger($("hOut"), (e) => { setOut(xToT(e.clientX)); seek(S.outT - 1 / (S.fps || 30)); });
  dragger(tl, (e) => { if (S.mode === "source" && S.playable) video.pause(); seek(xToT(e.clientX)); });
  for (const [h, which] of [[$("hIn"), "in"], [$("hOut"), "out"]]) {
    h.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      e.stopPropagation();
      const d = (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 1 : 1 / (S.fps || 30));
      if (which === "in") { setIn(S.inT + d); seek(S.inT); } else { setOut(S.outT + d); seek(S.outT); }
    });
  }

  // filmstrip: drawn from the local file when the browser can decode it, else from server frames
  let filmTimer;
  async function buildFilmstrip() {
    const canvas = $("filmstrip");
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !S.duration) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    const n = Math.max(4, Math.round(w / (h * 1.6)));
    const tw = canvas.width / n, th = canvas.height;
    const token = (S.filmToken = Math.random());
    const draw = (src, sw, sh, i) => {
      const k = Math.max(tw / sw, th / sh);
      const cw = tw / k, ch = th / k;
      ctx.drawImage(src, (sw - cw) / 2, (sh - ch) / 2, cw, ch, i * tw, 0, tw + 1, th);
    };
    if (S.playable) {
      const v = document.createElement("video");
      v.muted = true;
      v.preload = "auto";
      v.src = S.videoUrl;
      try {
        await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = rej; setTimeout(rej, 8000); });
        for (let i = 0; i < n; i++) {
          if (token !== S.filmToken) return;
          await new Promise((res) => { v.onseeked = res; v.currentTime = (S.duration * (i + 0.5)) / n; setTimeout(res, 2500); });
          draw(v, v.videoWidth, v.videoHeight, i);
        }
        S.filmstripDone = true;
        v.removeAttribute("src");
        v.load();
        return;
      } catch { /* fall through to server frames */ }
    }
    if (!S.meta || !S.jobId) return;
    await Promise.all([...Array(n).keys()].map((i) => new Promise((res) => {
      const img = new Image();
      img.onload = () => { if (token === S.filmToken) draw(img, img.naturalWidth, img.naturalHeight, i); res(); };
      img.onerror = res;
      img.src = `/api/jobs/${S.jobId}/frame?t=${((S.duration * (i + 0.5)) / n).toFixed(3)}&w=${Math.round(tw)}`;
    })));
    S.filmstripDone = true;
  }
  window.addEventListener("resize", () => {
    clearTimeout(filmTimer);
    filmTimer = setTimeout(() => { if (S.duration) buildFilmstrip(); }, 350);
  });

  // ------------------------------------------------------------------ 3. who gets replaced
  function setWho(faceId, mode) {
    if (mode === "all") for (const f of S.faces) if (f.id !== faceId) S.assign[f.id] = "largest";
    S.assign[faceId] = mode;
    renderWho();
    renderCast();
    renderFaces();
    update();
    if (mode === "photo" && !(S.targets[faceId] && S.targets[faceId].photos.length)) pickTarget(faceId);
  }
  function pickTarget(faceId) {
    S.pickTargetFor = faceId;
    $("targetInput").click();
  }
  async function addTargetPhotos(files, faceId) {
    const imgs = [...files].filter(isImage);
    if (!imgs.length) return toast("Drop a photo (JPG or PNG) of the person to replace. A screenshot from the video works.", true);
    const t = (S.targets[faceId] = S.targets[faceId] || { photos: [], usable: 0, busy: false, seq: 0 });
    const room = maxPhotos() - t.photos.length;
    if (room <= 0) return toast(`That's already ${maxPhotos()} photos. Remove one to add another.`, true);
    for (const file of imgs.slice(0, room)) {
      const blob = await downscale(file);
      t.photos.push({ blob, url: URL.createObjectURL(blob), state: "busy", thumb: null, error: null });
    }
    S.assign[faceId] = "photo";
    renderWho();
    syncTarget(faceId);
    warm();
  }
  async function syncTarget(faceId) {
    const t = S.targets[faceId];
    const seq = ++t.seq;
    if (!t.photos.length) {
      t.usable = 0; t.busy = false;
      renderWho(); renderFaces(); update();
      if (S.jobId) api("DELETE", `/api/jobs/${S.jobId}/targets/${faceId}`).catch(() => {});
      return;
    }
    t.busy = true;
    t.photos.forEach((p) => (p.state = "busy"));
    renderWho(); update();
    try {
      const id = await ensureJob();
      const form = new FormData();
      t.photos.forEach((p, i) => form.append("files", p.blob, `${i}.jpg`));
      const r = await task("POST", `/api/jobs/${id}/targets/${faceId}`, form);
      if (seq !== t.seq) return;
      r.photos.forEach((res, i) => { const p = t.photos[i]; if (!p) return; p.state = res.ok ? "ok" : "bad"; p.thumb = res.thumb || null; p.error = res.error || null; });
      t.usable = r.usable;
      if (!r.usable) toast("No face found in that photo. Use a clear shot of the person's face, a screenshot from the video is ideal.", true, 6000);
    } catch (e) {
      if (seq !== t.seq) return;
      t.photos.forEach((p) => (p.state = "bad")); t.usable = 0; toast(e.message, true);
    }
    t.busy = false;
    renderWho(); renderFaces(); update();
  }
  function renderWho() {
    const box = $("who");
    box.textContent = "";
    const multi = S.faces.length > 1;
    S.faces.forEach((f, idx) => {
      const row = el("div", "who-row");
      row.style.setProperty("--role", roleVar(f.id));
      const tag = el("span", "role-tag", f.id.toUpperCase());
      tag.title = faceName(f);
      row.append(tag);
      const opts = el("div", "who-opts");
      opts.setAttribute("role", "radiogroup");
      opts.setAttribute("aria-label", `Who ${faceName(f)} replaces`);
      const cur = S.assign[f.id];
      const scanned = cur && cur !== "all" && cur !== "largest" && cur !== "photo" && S.people && S.people.find((p) => p.id === cur);
      const choices = [
        ["largest", idx ? `The #${idx + 1} biggest face` : "The biggest face", idx ? "" : "in each frame, no searching"],
        ...(multi ? [] : [["all", "Everyone", "every face in the video"]]),
        ["photo", "A specific person", "drop a photo of them"],
      ];
      if (scanned) choices.push([cur, "Picked from the video", `on screen ${Math.round(scanned.share * 100)}%`, scanned.thumb]);
      choices.forEach(([mode, label, sub, img]) => {
        const b = el("button", "who-opt");
        b.type = "button";
        b.setAttribute("role", "radio");
        b.setAttribute("aria-checked", String(cur === mode));
        if (img) { const i = el("img"); i.src = img; i.alt = ""; b.append(i); }
        b.append(el("span", "", label));
        if (sub) b.append(el("small", "", sub));
        b.onclick = () => setWho(f.id, mode);
        opts.append(b);
      });
      row.append(opts);
      if (cur === "photo") {
        const t = S.targets[f.id] || { photos: [] };
        const wrap = el("div", "who-target");
        const drop = el("div", `key-surface${t.photos.length ? " has-faces" : ""}`);
        drop.tabIndex = 0;
        drop.setAttribute("role", "button");
        drop.setAttribute("aria-label", `Photo of the person ${faceName(f)} replaces`);
        const copy = el("span", "key-copy");
        copy.append(el("strong", "", t.photos.length ? "Add another photo" : "Drop their photo"));
        drop.append(copy);
        drop.onclick = () => pickTarget(f.id);
        drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickTarget(f.id); } };
        const list = el("ul", "face-list");
        t.photos.forEach((p, i) => {
          const li = el("li", `face-item ${p.state}`);
          const im = el("img"); im.src = p.thumb || p.url; im.alt = `Person to replace, photo ${i + 1}`;
          li.append(im);
          if (p.state !== "busy") li.append(el("span", "face-state", p.state === "ok" ? "\u2713" : "!"));
          const rm = el("button", "remove", "\u00d7"); rm.type = "button"; rm.setAttribute("aria-label", "Remove this photo");
          rm.onclick = () => { URL.revokeObjectURL(p.url); t.photos.splice(i, 1); renderWho(); syncTarget(f.id); };
          li.append(rm);
          if (p.error) li.title = p.error;
          list.append(li);
        });
        const err = t.photos.find((p) => p.error);
        const side = el("div");
        side.append(list, el("p", "hint", err ? err.error : "A screenshot of their face from the video is ideal. Only that person gets swapped."));
        wrap.append(drop, side);
        row.append(wrap);
      }
      box.append(row);
    });
  }

  async function scan() {
    if (!S.meta) return;
    const btn = $("scanBtn");
    busy(btn, true);
    S.scanning = true;
    update();
    try {
      const r = await task("POST", `/api/jobs/${S.jobId}/scan`, { start: S.inT, end: S.outT });
      S.people = r.people;
      // person ids are renumbered by every scan, so earlier scan picks fall back to the default
      for (const k of Object.keys(S.assign)) if (!["all", "largest", "photo"].includes(S.assign[k])) S.assign[k] = "largest";
      if (!r.people.length) toast("No faces found in this section. Try a section where faces are clearly visible.", true);
      S.activeFace = S.faces[0].id;
      renderCast();
      renderWho();
      renderFaces();
    } catch (e) {
      toast(e.message, true);
    }
    busy(btn, false);
    S.scanning = false;
    btn.textContent = "Scan again";
    update();
  }

  const ownerOf = (pid) => Object.keys(S.assign).find((k) => S.assign[k] === pid);

  function assignPerson(o) {
    const active = S.activeFace;
    const owner = ownerOf(o.id);
    if (owner === active) {
      S.assign[active] = "largest";  // clicking your own pick again clears it
    } else {
      if (owner) S.assign[owner] = "largest";
      S.assign[active] = o.id;
      const next = S.faces.find((f) => f.id !== active && !["all", "photo"].includes(S.assign[f.id]) && !(S.people || []).some((p) => p.id === S.assign[f.id]));
      if (next) S.activeFace = next.id;
      // jump to where this person shows up so the preview lands on them
      if (o.person && o.person.times.length) {
        const t = o.person.times.find((x) => x >= S.inT && x <= S.outT) ?? o.person.times[0];
        seek(t);
      }
    }
    renderCast();
    renderWho();
    renderFaces();
    update();
  }

  function renderAssignBar() {
    const bar = $("assignBar");
    bar.hidden = S.faces.length < 2 || !S.people || !S.people.length;
    const chips = $("assignChips");
    chips.textContent = "";
    if (bar.hidden) return;
    S.faces.forEach((f) => {
      const c = el("button", "assign-chip");
      c.type = "button";
      c.setAttribute("role", "radio");
      c.setAttribute("aria-checked", String(S.activeFace === f.id));
      c.style.setProperty("--role", roleVar(f.id));
      c.append(el("span", "role-tag", f.id.toUpperCase()), el("span", "", faceName(f)),
        el("small", "", (S.people || []).some((p) => p.id === S.assign[f.id]) ? "✓" : "pick a person"));
      c.onclick = () => { S.activeFace = f.id; renderAssignBar(); };
      chips.append(c);
    });
  }

  function renderCast() {
    const cast = $("cast");
    cast.textContent = "";
    renderAssignBar();
    if (S.config) renderFormat();
    if (!S.people) {
      renderAppearances();
      return;
    }
    if (!S.people.length) {
      cast.append(el("li", "cast-empty", "Nobody found in this section."));
      renderAppearances();
      return;
    }
    const multi = S.faces.length > 1;
    const opts = S.people.map((p) => ({
      id: p.id, img: p.thumb, label: `On screen ${Math.round(p.share * 100)}%`, person: p,
    }));
    opts.forEach((o) => {
      const li = el("li");
      const who = ownerOf(o.id);
      const b = el("button", `cast-member${o.id === "all" ? " cast-everyone" : ""}${who ? " assigned" : ""}`);
      b.type = "button";
      if (who) b.style.setProperty("--role", roleVar(who));
      b.setAttribute("aria-pressed", String(!!who));
      const what = o.id === "all" ? "Every face" : `Person, ${o.label.toLowerCase()}`;
      b.setAttribute("aria-label", who ? `${what}, replaced by ${faceName(who)}` : what);
      const face = el("span", "cast-face");
      if (o.img) { const img = el("img"); img.src = o.img; img.alt = ""; face.append(img); } else face.textContent = "All";
      b.append(face, el("span", "cast-label", o.label));
      if (who && multi) b.append(el("span", "cast-badge", who.toUpperCase()));
      b.onclick = () => assignPerson(o);
      li.append(b);
      cast.append(li);
    });
    renderAppearances();
  }

  // green (and blue, amber...) ticks above the timeline: where each chosen person appears
  function renderAppearances() {
    const track = $("appearTrack");
    track.textContent = "";
    if (!S.people || !S.duration) return;
    S.faces.forEach((f, row) => {
      const p = S.people.find((x) => x.id === S.assign[f.id]);
      if (!p) return;
      p.times.forEach((t) => {
        const i = el("i");
        i.style.left = `${(t / S.duration) * 100}%`;
        i.style.setProperty("--role", roleVar(f.id));
        i.dataset.row = String(row);
        track.append(i);
      });
    });
  }

  // ------------------------------------------------------------------ 4. quality
  function renderTiers() {
    const box = $("tiers");
    box.textContent = "";
    S.config.tiers.forEach((t) => {
      const b = el("button", "tier");
      b.type = "button";
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(S.tier === t.key));
      const meter = (label, n) => {
        const m = el("span", "meter");
        m.append(el("span", "meter-label", label));
        for (let i = 1; i <= 3; i++) m.append(el("i", i <= n ? "on" : ""));
        return m;
      };
      const meters = el("span", "meters");
      meters.append(meter("Speed", t.speed), meter("Detail", t.quality));
      meters.setAttribute("aria-label", `Speed ${t.speed} of 3, detail ${t.quality} of 3`);
      b.append(el("span", "tier-name", t.label), meters, el("span", "tier-blurb", t.blurb));
      b.onclick = () => {
        S.tier = t.key;
        localStorage.setItem("lr.tier", t.key);
        renderTiers();
      };
      box.append(b);
    });
  }

  // ------------------------------------------------------------------ format (reframe)
  const ASPECT_LABEL = { original: "Original", "9:16": "9:16", "4:5": "4:5", "1:1": "1:1", "16:9": "16:9" };
  const ASPECT_USE = { original: "as filmed", "9:16": "Reels, TikTok, Stories", "4:5": "feed", "1:1": "square", "16:9": "landscape" };
  function srcSize() {
    if (S.meta) return [S.meta.width, S.meta.height];
    if (video.videoWidth) return [video.videoWidth, video.videoHeight];
    return [16, 9];
  }
  // same maths as the server: cut, never scale, even numbers
  function cropSize(aspect) {
    const [W, H] = srcSize();
    const a = { "9:16": 9 / 16, "4:5": 4 / 5, "1:1": 1, "16:9": 16 / 9 }[aspect];
    if (!a) return [W, H];
    let cw, ch;
    if (W / H > a) { ch = H; cw = H * a; } else { cw = W; ch = W / a; }
    cw = Math.floor(cw / 2) * 2; ch = Math.floor(ch / 2) * 2;
    return [Math.min(cw, W), Math.min(ch, H)];
  }
  const reframed = () => S.frame.aspect !== "original" && (() => { const [W, H] = srcSize(); const [w, h] = cropSize(S.frame.aspect); return w !== W || h !== H; })();
  function renderFormat() {
    const box = $("formatChips");
    box.textContent = "";
    const aspects = (S.config && S.config.aspects) || Object.keys(ASPECT_LABEL);
    aspects.forEach((a) => {
      const b = el("button", "chip");
      b.type = "button";
      b.dataset.aspect = a;
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(S.frame.aspect === a));
      b.title = ASPECT_USE[a] || "";
      b.append(el("i"), el("span", "", ASPECT_LABEL[a] || a));
      b.onclick = () => { S.frame.aspect = a; S.lastCrop = null; renderFormat(); };
      box.append(b);
    });
    const on = reframed();
    const [w, h] = cropSize(S.frame.aspect);
    $("formatSize").textContent = !S.videoFile ? (S.frame.aspect === "original" ? "" : `Exports at ${ASPECT_LABEL[S.frame.aspect]}`)
      : on ? `Exports ${w} × ${h}, cut from the ${srcSize().join(" × ")} original` : `Exports ${srcSize().join(" × ")} as filmed`;
    $("formatMode").hidden = !on;
    $("modeFollow").setAttribute("aria-checked", String(S.frame.mode === "follow"));
    $("modeFixed").setAttribute("aria-checked", String(S.frame.mode === "fixed"));
    const hint = $("formatHint");
    hint.hidden = !on;
    const first = S.faces[0];
    const a0 = first && S.assign[first.id];
    const who = a0 === "all" || a0 === "largest" ? "the biggest face" : first && S.faces.length > 1 ? `${faceName(first)}'s person` : "the person you picked";
    hint.textContent = S.frame.mode === "follow"
      ? `The crop glides to keep ${who} in frame. Preview a frame to see it.`
      : "Drag the box on the video to place the crop. It stays there for the whole section.";
    $("vrToggle").disabled = on;
    $("vrToggle").closest("label").title = on ? "VR footage can't be reframed. Set the format to Original." : "";
    layoutCrop();
  }
  $("modeFollow").onclick = () => { S.frame.mode = "follow"; renderFormat(); };
  $("modeFixed").onclick = () => { S.frame.mode = "fixed"; renderFormat(); };

  // where the video actually sits inside the stage (object-fit: contain)
  function videoRect() {
    const st = $("stage").getBoundingClientRect();
    const [W, H] = srcSize();
    const k = Math.min(st.width / W, st.height / H);
    const w = W * k, h = H * k;
    return { left: (st.width - w) / 2, top: (st.height - h) / 2, width: w, height: h, stage: st };
  }
  function layoutCrop() {
    const ov = $("cropOverlay");
    const show = S.mode === "source" && !!S.videoFile && reframed();
    ov.hidden = !show;
    if (!show) return;
    const r = videoRect();
    const [W, H] = srcSize();
    const [cw, ch] = cropSize(S.frame.aspect);
    let cx = S.frame.cx, cy = S.frame.cy;
    if (S.frame.mode === "follow" && S.lastCrop) { cx = S.lastCrop.x + S.lastCrop.w / 2; cy = S.lastCrop.y + S.lastCrop.h / 2; }
    const fw = cw / W, fh = ch / H;
    const x = Math.min(Math.max(cx - fw / 2, 0), 1 - fw), y = Math.min(Math.max(cy - fh / 2, 0), 1 - fh);
    const bx = r.left + x * r.width, by = r.top + y * r.height, bw = fw * r.width, bh = fh * r.height;
    const box = $("cropBox");
    Object.assign(box.style, { left: `${bx}px`, top: `${by}px`, width: `${bw}px`, height: `${bh}px` });
    box.classList.toggle("draggable", S.frame.mode === "fixed");
    box.setAttribute("aria-valuenow", String(Math.round(cx * 100)));
    box.setAttribute("aria-disabled", String(S.frame.mode !== "fixed"));
    $("cropLabel").textContent = S.frame.mode === "fixed" ? `${ASPECT_LABEL[S.frame.aspect]} · drag to place` : `${ASPECT_LABEL[S.frame.aspect]} · follows the person`;
    const dim = (sel, st) => Object.assign(ov.querySelector(sel).style, st);
    dim(".crop-dim.n", { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${by - r.top}px` });
    dim(".crop-dim.s", { left: `${r.left}px`, top: `${by + bh}px`, width: `${r.width}px`, height: `${r.top + r.height - by - bh}px` });
    dim(".crop-dim.w", { left: `${r.left}px`, top: `${by}px`, width: `${bx - r.left}px`, height: `${bh}px` });
    dim(".crop-dim.e", { left: `${bx + bw}px`, top: `${by}px`, width: `${r.left + r.width - bx - bw}px`, height: `${bh}px` });
  }
  let cropDrag = null;
  $("cropBox").addEventListener("pointerdown", (e) => {
    if (S.frame.mode !== "fixed") return;
    e.preventDefault();
    $("cropBox").setPointerCapture(e.pointerId);
    cropDrag = { x: e.clientX, y: e.clientY, cx: S.frame.cx, cy: S.frame.cy };
  });
  $("cropBox").addEventListener("pointermove", (e) => {
    if (!cropDrag) return;
    const r = videoRect();
    S.frame.cx = Math.min(Math.max(cropDrag.cx + (e.clientX - cropDrag.x) / r.width, 0), 1);
    S.frame.cy = Math.min(Math.max(cropDrag.cy + (e.clientY - cropDrag.y) / r.height, 0), 1);
    layoutCrop();
  });
  for (const ev of ["pointerup", "pointercancel"]) $("cropBox").addEventListener(ev, () => (cropDrag = null));
  $("cropBox").addEventListener("keydown", (e) => {
    if (S.frame.mode !== "fixed") return;
    const d = e.shiftKey ? 0.1 : 0.02;
    const map = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] };
    if (!map[e.key]) return;
    e.preventDefault();
    e.stopPropagation();
    S.frame.cx = Math.min(Math.max(S.frame.cx + map[e.key][0], 0), 1);
    S.frame.cy = Math.min(Math.max(S.frame.cy + map[e.key][1], 0), 1);
    layoutCrop();
  });
  window.addEventListener("resize", layoutCrop);

  // ------------------------------------------------------------------ gating
  function blocker(forRender) {
    if (!S.videoFile) return "Add the ad video.";
    if (S.uploading) return `Uploading video ${Math.round(S.uploadPct * 100)}%`;
    if (!S.meta) return "The video didn't upload. Drop it again.";
    const multi = S.faces.length > 1;
    for (const f of S.faces) {
      if (f.busy) return `Checking the photos for ${faceName(f)}`;
      if (!f.usable) return multi ? `Add a photo for ${faceName(f)}, or remove it.` : "Add a photo of your face.";
    }
    for (const f of S.faces) {
      const a = S.assign[f.id];
      if (!a) return multi ? `Pick who ${faceName(f)} replaces.` : "Pick who you're replacing.";
      if (a === "photo") {
        const t = S.targets[f.id];
        if (!t || !t.photos.length) return `Drop a photo of the person ${faceName(f)} replaces, or choose "The biggest face".`;
        if (t.busy) return "Checking the photo of the person to replace";
        if (!t.usable) return `No face was found in the photo for ${faceName(f)}. Try a clearer one.`;
      }
    }
    if (forRender && !$("consent").checked) return "Tick the permission box to render.";
    return "";
  }
  const castBody = () => S.faces.map((f) => ({ face: f.id, person: S.assign[f.id] }));
  function update() {
    const preBlock = blocker(false), renBlock = blocker(true);
    const rendering = activeTakes().length > 0;
    $("previewBtn").disabled = !!preBlock || S.previewing;
    $("renderBtn").disabled = !!renBlock;
    $("renderBtn").textContent = rendering ? "Render another take" : "Render section";
    $("scanBtn").disabled = !S.meta || S.scanning;
    $("transport").setAttribute("aria-disabled", String(!S.duration));
    $("whyDisabled").textContent = rendering ? "" : renBlock;
    const steps = document.querySelectorAll("h2 .step");
    steps[0].classList.toggle("done", S.faces.every((f) => f.usable > 0));
    steps[1].classList.toggle("done", S.faces.every((f) => S.assign[f.id] && (S.assign[f.id] !== "photo" || (S.targets[f.id] && S.targets[f.id].usable))));
    steps[2].classList.toggle("done", !!S.takes.find((t) => t.state === "done"));
  }

  // ------------------------------------------------------------------ preview
  async function previewFrame() {
    if (blocker(false)) return toast(blocker(false), true);
    const btn = $("previewBtn");
    if (S.mode === "source" && S.playable) video.pause();
    const t = Math.min(Math.max(curT(), 0), S.duration);
    S.previewing = true;
    busy(btn, true);
    update();
    try {
      const r = await task("POST", `/api/jobs/${S.jobId}/preview`,
        { t, cast: castBody(), tier: S.tier, vr: $("vrToggle").checked, frame: S.frame });
      S.lastCrop = r.crop || null;
      $("splitBefore").src = r.before;
      $("splitAfter").src = r.after;
      showStage("preview", `Preview at ${tc(r.t)}, ${tierLabel(S.tier)}, ${(r.ms / 1000).toFixed(1)} s`);
      setSplit(50);
      const everyFace = S.assign[S.faces[0].id] === "all";
      const missing = everyFace ? [] : S.faces.filter((f) => !r.found.includes(f.id) && S.assign[f.id] !== "largest");
      if (!r.faces) toast("No faces in this frame. Move the playhead to a frame with a face.", true, 6000);
      else if (missing.length) {
        toast(S.faces.length === 1
          ? "The person you picked isn't in this frame. Move the playhead to a frame where they're visible."
          : `The person for ${missing.map(faceName).join(" and ")} isn't in this frame. Move the playhead to a frame where they're visible.`,
        true, 6000);
      }
    } catch (e) {
      toast(e.message, true);
    }
    S.previewing = false;
    busy(btn, false);
    update();
  }
  const tierLabel = (k) => ((S.config && S.config.tiers.find((t) => t.key === k)) || { label: k }).label;

  function setSplit(pct) {
    pct = Math.min(100, Math.max(0, pct));
    $("splitAfter").style.clipPath = `inset(0 0 0 ${pct}%)`;
    $("splitHandle").style.left = `${pct}%`;
    $("splitHandle").setAttribute("aria-valuenow", String(Math.round(pct)));
  }
  const stagePct = (e) => { const r = $("stage").getBoundingClientRect(); return ((e.clientX - r.left) / r.width) * 100; };
  dragger($("splitHandle"), (e) => setSplit(stagePct(e)));
  $("split").addEventListener("pointerdown", (e) => { if (e.target === $("split") || e.target.tagName === "IMG") setSplit(stagePct(e)); });
  $("splitHandle").addEventListener("keydown", (e) => {
    const cur = parseFloat($("splitHandle").style.left) || 50;
    if (e.key === "ArrowLeft") { e.preventDefault(); e.stopPropagation(); setSplit(cur - 5); }
    if (e.key === "ArrowRight") { e.preventDefault(); e.stopPropagation(); setSplit(cur + 5); }
  });

  // ------------------------------------------------------------------ stage modes
  function showStage(mode, badge) {
    S.mode = mode;
    layoutCrop();
    $("split").hidden = mode !== "preview";
    $("stageBadge").hidden = !badge;
    $("stageBadge").textContent = badge || "";
    $("stageClose").hidden = mode === "source";
    if (mode !== "result") setCompare(false);
  }
  function showSource() {
    if (S.mode === "result") {
      video.removeAttribute("controls");
      if (!S.videoUrl) { video.removeAttribute("src"); video.load(); video.hidden = true; $("videoDrop").hidden = false; showStage("source"); return; }
      video.src = S.videoUrl;
      video.currentTime = S.t;
      if (!S.playable) { video.hidden = true; $("posterFallback").hidden = false; }
    }
    showStage("source");
  }
  function showResult(take) {
    S.take = take;
    if (!S.videoUrl) { $("videoDrop").hidden = true; }
    $("posterFallback").hidden = true;
    video.hidden = false;
    video.pause();
    video.src = `/api/renders/${take.rid}/video`;
    video.setAttribute("controls", "");
    showStage("result", `Take ${take.n}, ${tierLabel(take.tier)}${take.aspect && take.aspect !== "original" ? `, ${take.aspect}` : ""}`);
    const cropped = take.aspect && take.aspect !== "original";
    $("compareBtn").disabled = !S.playable || cropped;
    $("compareBtn").title = cropped ? "This take is cropped to a different shape, so it can't be laid over the original."
      : S.playable ? "" : "Your browser can't play the original file, so it can't be compared here.";
    renderTakes();
    showPanelFor(take);
  }

  // before/after over the playing render
  function setCompare(on) {
    S.comparing = on && S.playable;
    const cv = $("compareVideo");
    cv.hidden = !S.comparing;
    $("compareHandle").hidden = !S.comparing;
    $("compareBtn").setAttribute("aria-pressed", String(S.comparing));
    if (S.comparing) {
      if (cv.src !== S.videoUrl) cv.src = S.videoUrl;
      setComparePct(S.comparePct);
      syncCompare(true);
    } else cv.pause();
  }
  function setComparePct(p) {
    S.comparePct = Math.min(100, Math.max(0, p));
    $("compareVideo").style.clipPath = `inset(0 ${100 - S.comparePct}% 0 0)`;
    $("compareHandle").style.left = `${S.comparePct}%`;
    $("compareHandle").setAttribute("aria-valuenow", String(Math.round(S.comparePct)));
  }
  function syncCompare(force) {
    const cv = $("compareVideo");
    if (!S.take) return;
    const want = S.take.start + video.currentTime;
    if (force || Math.abs(cv.currentTime - want) > 0.08) cv.currentTime = want;
    if (video.paused && !cv.paused) cv.pause();
    if (!video.paused && cv.paused) cv.play().catch(() => {});
  }
  dragger($("compareHandle"), (e) => setComparePct(stagePct(e)));
  $("compareHandle").addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); e.stopPropagation(); setComparePct(S.comparePct - 5); }
    if (e.key === "ArrowRight") { e.preventDefault(); e.stopPropagation(); setComparePct(S.comparePct + 5); }
  });

  // ------------------------------------------------------------------ render
  async function startRender() {
    const block = blocker(true);
    if (block) return toast(block, true);
    const btn = $("renderBtn");
    busy(btn, true);
    try {
      const cast = castBody();
      const body = { start: S.inT, end: S.outT, cast, tier: S.tier, vr: $("vrToggle").checked, consent: true, frame: S.frame };
      const r = await api("POST", `/api/jobs/${S.jobId}/render`, body);
      // how much of the section each chosen person is on screen, to judge the face-found numbers later
      const share = Object.fromEntries(cast.map((c) => [c.face, ((S.people || []).find((p) => p.id === c.person) || { share: 1 }).share]));
      const take = { n: S.takes.length + 1, rid: r.render_id, tier: S.tier, start: S.inT, end: S.outT, state: "queued", lanes: r.lanes, faces: cast.length, share, aspect: S.frame.aspect, outSize: r.out_size,
        video: (S.videoFile && S.videoFile.name) || "video", jobId: S.jobId };
      S.takes.push(take);
      S.take = take;
      renderTakes();
      showPanelFor(take);
      $("renderPanel").scrollIntoView({ block: "nearest", behavior: "smooth" });
      poll(take);
    } catch (e) {
      toast(e.message, true);
    }
    busy(btn, false);
    update();
  }

  const isActive = (t) => t && ["queued", "rendering", "finalizing"].includes(t.state);
  const activeTakes = () => S.takes.filter(isActive);
  let liveAt = 0;
  async function poll(take) {
    clearTimeout(take.timer);
    take.timer = null;
    if (take.polling || !S.takes.includes(take)) return;  // one poll loop per take; none for removed takes
    take.polling = true;
    try {
      const st = await api("GET", `/api/renders/${take.rid}`);
      take.state = st.state;
      take.status = st;
      if (S.take === take) {
        renderPanel(take);
        if (Date.now() - liveAt > 1800 && st.state === "rendering") { liveAt = Date.now(); refreshLive(take); }
      }
      renderTakes();
      if (st.state === "done") {
        update();
        // show it only if you're not already looking at another finished take
        if (S.take === take || S.mode !== "result") showResult(take);
        toast(`Take ${take.n} is ready.`);
        take.polling = false;
        return;
      }
      if (st.state === "error" || st.state === "cancelled") { update(); take.polling = false; return; }
    } catch (e) {
      if (S.take === take) $("renderMsg").textContent = `Lost contact with the server, retrying. (${e.message})`;
    }
    take.polling = false;
    take.timer = setTimeout(() => poll(take), 1000);
  }
  async function refreshLive(take) {
    try {
      const r = await fetch(`/api/renders/${take.rid}/live.jpg`, { cache: "no-store" });
      if (r.status !== 200) return;
      const url = URL.createObjectURL(await r.blob());
      const img = $("liveFrame");
      if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
      img.dataset.url = url;
      img.src = url;
      img.hidden = false;
    } catch { /* next poll will try again */ }
  }

  function showPanelFor(take) {
    $("renderPanel").hidden = false;
    $("liveFrame").hidden = true;
    renderPanel(take);
  }
  function renderPanel(take) {
    const st = take.status || { state: take.state, lanes: [], frames_done: 0, frames_total: 0, progress: 0 };
    const where = S.config && S.config.backend === "modal" ? "GPUs" : "this computer";
    const lanes = st.lanes || [];
    const anyActive = lanes.some((l) => l.active);
    const titles = {
      queued: "Queued",
      rendering: anyActive
        ? (S.config.backend === "modal" ? `Rendering on ${lanes.length} GPU${lanes.length > 1 ? "s" : ""}` : "Rendering on this computer")
        : (S.config.backend === "modal" ? "Warming up GPUs" : "Loading face models"),
      finalizing: "Stitching and adding audio",
      done: `Take ${take.n} is ready`,
      error: "Render stopped",
      cancelled: "Stopped",
    };
    $("renderTitle").textContent = titles[st.state] || st.state;
    $("tally").className = `tally ${st.state === "done" ? "done" : ["error", "cancelled"].includes(st.state) ? "" : "live"}`;
    $("renderPct").textContent = st.state === "done" ? "" : `${Math.floor((st.progress || 0) * 100)}%`;

    const box = $("lanes");
    if (box.childElementCount !== lanes.length || box.dataset.rid !== take.rid) {
      box.textContent = "";
      box.dataset.rid = take.rid;
      lanes.forEach((_, i) => {
        const row = el("div", "lane");
        const bar = el("span", "lane-bar");
        bar.append(el("span"));
        row.append(el("span", "", where === "GPUs" ? `GPU ${i + 1}` : `Part ${i + 1}`), bar);
        box.append(row);
      });
    }
    lanes.forEach((l, i) => {
      const row = box.children[i];
      row.classList.toggle("waiting", !l.active && st.state === "rendering");
      row.querySelector(".lane-bar span").style.width = `${(l.done / Math.max(1, l.total)) * 100}%`;
    });
    $("stFrames").textContent = st.frames_total ? `${st.frames_done} / ${st.frames_total}` : "-";
    $("stElapsed").textContent = secs(st.elapsed);
    $("stEta").textContent = st.state === "done" ? "0s" : secs(st.eta);
    const rc = (st.role_coverage || []).filter((c) => c.coverage != null);
    $("stCover").textContent = !rc.length ? "-" : rc.length === 1 ? `${Math.round(rc[0].coverage * 100)}% of frames`
      : rc.map((c) => `${c.face.toUpperCase()} ${Math.round(c.coverage * 100)}%`).join(", ");
    const msg = $("renderMsg");
    msg.classList.toggle("err", st.state === "error");
    if (st.state === "error") msg.textContent = st.error || "Something went wrong. Try rendering again.";
    else if (st.state === "done") {
      const mb = st.size ? `${(st.size / 1048576).toFixed(1)} MB` : "";
      msg.textContent = `${(take.end - take.start).toFixed(1)} s rendered in ${secs(st.elapsed)}${mb ? `, ${mb}` : ""}.`;
      // compare with how much the person is on screen: being off screen part of the time is fine
      const low = rc.filter((c) => c.coverage < 0.7 * ((take.share || {})[c.face] ?? 1) - 0.05);
      if (low.length) {
        msg.textContent += take.faces > 1
          ? ` ${low.map((c) => faceName(c.face)).join(" and ")} missed some frames where the person is on screen. Try Premium for small or turned faces.`
          : " Your face missed some frames where the person is on screen. Try Premium for small or turned faces.";
      }
    } else msg.textContent = st.state === "rendering" && !anyActive && S.config.backend === "modal"
      ? "GPUs take up to a minute to start when they've been idle." : "";
    const done = st.state === "done";
    $("resultActions").hidden = !done;
    if (done) $("downloadBtn").href = `/api/renders/${take.rid}/video?download=1`;
    if (done) $("liveFrame").hidden = true;
  }

  // Takes are remembered in this browser, so a reload or a server restart doesn't lose them.
  function saveTakes() {
    try {
      localStorage.setItem("lr.takes", JSON.stringify(S.takes.map((t) => ({
        n: t.n, rid: t.rid, tier: t.tier, start: t.start, end: t.end, state: t.state, lanes: t.lanes, faces: t.faces,
        share: t.share, aspect: t.aspect, outSize: t.outSize, video: t.video, jobId: t.jobId,
      }))));
    } catch { /* storage unavailable */ }
  }
  async function restoreTakes() {
    let saved = [];
    try { saved = JSON.parse(localStorage.getItem("lr.takes") || "[]"); } catch { return; }
    if (!saved.length) return;
    const kept = [];
    for (const t of saved) {
      try {
        const st = await api("GET", `/api/renders/${t.rid}`);
        t.state = st.state;
        t.status = st;
        kept.push(t);
      } catch { /* expired on the server: drop it */ }
    }
    S.takes = kept;
    saveTakes();
    renderTakes();
    update();
    kept.filter(isActive).forEach((t) => poll(t));
    if (kept.length) toast(`${kept.length} earlier take${kept.length > 1 ? "s" : ""} restored.`);
  }
  function renderTakes() {
    saveTakes();
    $("takesWrap").hidden = S.takes.length < 1;
    const ol = $("takes");
    ol.textContent = "";
    const manyVideos = new Set(S.takes.map((t) => t.video)).size > 1;
    S.takes.forEach((t) => {
      const li = el("li");
      const b = el("button", `take-${t.state}`);
      b.type = "button";
      if (S.take === t) b.setAttribute("aria-current", "true");
      const pct = t.status ? Math.floor((t.status.progress || 0) * 100) : 0;
      const state = t.state === "done" ? `Ready, ${(t.end - t.start).toFixed(1)} s` : t.state === "error" ? "Failed" : t.state === "cancelled" ? "Stopped"
        : t.state === "queued" ? "Queued" : t.state === "finalizing" ? "Finishing" : `Rendering ${pct}%`;
      const label = `Take ${t.n}, ${tierLabel(t.tier)}${t.faces > 1 ? `, ${t.faces} faces` : ""}${t.aspect && t.aspect !== "original" ? `, ${t.aspect}` : ""}${manyVideos ? `, ${t.video}` : ""}`;
      b.append(el("span", "", label), el("span", "", state));
      if (isActive(t)) { const bar = el("i", "take-bar"); bar.style.width = `${pct}%`; b.append(bar); }
      b.onclick = () => {
        if (t.state === "done") showResult(t);
        else { S.take = t; showPanelFor(t); renderTakes(); if (isActive(t)) poll(t); }
      };
      const x = el("button", "take-x", "\u00d7");
      x.type = "button";
      x.title = isActive(t) ? (t.state === "queued" ? "Remove from the queue" : "Stop this render") : "Remove from the list";
      x.setAttribute("aria-label", `${x.title}: take ${t.n}`);
      x.onclick = (ev) => { ev.stopPropagation(); removeTake(t); };
      li.append(b, x);
      ol.append(li);
    });
  }

  async function removeTake(t) {
    const wasActive = isActive(t);
    if (wasActive) {
      const wasQueued = t.state === "queued";
      try { await api("DELETE", `/api/renders/${t.rid}`); } catch (e) { return toast(e.message, true); }
      t.state = "cancelled";
      toast(wasQueued ? `Take ${t.n} removed from the queue.` : `Take ${t.n} stopped.`);
    }
    clearTimeout(t.timer);
    S.takes = S.takes.filter((x) => x !== t);
    if (S.take === t) {
      S.take = null;
      $("renderPanel").hidden = true;
      if (S.mode === "result") showSource();
    }
    renderTakes();
    update();
  }

  // ------------------------------------------------------------------ warm GPUs early
  let warmed = 0;
  function warm() {
    if (Date.now() - warmed < 120000) return;
    warmed = Date.now();
    api("POST", "/api/warm").catch(() => {});
  }

  // ------------------------------------------------------------------ inputs + drag/drop
  function bindDrop(zone) {
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); zone.querySelector("input").click(); }
    });
  }
  bindDrop($("videoDrop"));
  $("faceInput").addEventListener("change", (e) => { addPhotos(e.target.files, S.pickFor || S.activeFace); e.target.value = ""; });
  $("videoInput").addEventListener("change", (e) => { if (e.target.files[0]) setVideo(e.target.files[0]); e.target.value = ""; });
  $("addSetBtn").onclick = () => addFaceSet(true);
  $("targetInput").addEventListener("change", (e) => { addTargetPhotos(e.target.files, S.pickTargetFor || S.faces[0].id); e.target.value = ""; });
  $("camCancel").onclick = closeCamera;
  $("camShoot").onclick = shoot;

  // Drop anywhere: videos go to the stage, images to your face.
  let dragDepth = 0;
  const zones = () => [...document.querySelectorAll(".key-surface")];
  document.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; zones().forEach((z) => z.classList.add("drag")); });
  document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; zones().forEach((z) => z.classList.remove("drag")); } });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    zones().forEach((z) => z.classList.remove("drag"));
    const files = [...(e.dataTransfer && e.dataTransfer.files || [])];
    const vids = files.filter(isVideo), imgs = files.filter(isImage);
    if (vids.length) setVideo(vids[0]);
    if (imgs.length) {
      const whoRow = e.target.closest && e.target.closest(".who-row");   // dropped on "a specific person"?
      const set = e.target.closest && e.target.closest(".face-set");     // dropped on a specific face?
      if (whoRow) addTargetPhotos(imgs, S.faces[[...whoRow.parentNode.children].indexOf(whoRow)].id);
      else addPhotos(imgs, set ? set.dataset.face : S.activeFace);
    }
    if (!vids.length && !imgs.length && files.length) toast("Drop a video (MP4, MOV) or a photo (JPG, PNG).", true);
  });

  $("playBtn").onclick = togglePlay;
  $("stepBack").onclick = () => step(-1);
  $("stepFwd").onclick = () => step(1);
  $("markIn").onclick = () => setIn(curT(), true);
  $("markOut").onclick = () => setOut(curT(), true);
  $("wholeBtn").onclick = wholeVideo;
  bindTimeField($("inField"), setIn);
  bindTimeField($("outField"), setOut);
  $("scanBtn").onclick = scan;
  $("previewBtn").onclick = previewFrame;
  $("renderBtn").onclick = startRender;
  $("consent").onchange = update;
  $("vrToggle").onchange = () => { if ($("vrToggle").checked) { S.frame.aspect = "original"; renderFormat(); } };
  $("stageClose").onclick = showSource;
  $("compareBtn").onclick = () => setCompare(!S.comparing);

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select") || !$("camera").hidden || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape" && S.mode !== "source") return showSource();
    if (!S.duration) return;
    // Space would also press a focused button; arrows belong to a focused slider. Otherwise they drive the video.
    const onButton = e.target.closest("button, a, [role=button], [role=slider]");
    const onSlider = e.target.closest("[role=slider]");
    if (e.key === " " && !onButton) { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowLeft" && !onSlider) { e.preventDefault(); step(e.shiftKey ? -(S.fps || 30) : -1); }
    else if (e.key === "ArrowRight" && !onSlider) { e.preventDefault(); step(e.shiftKey ? (S.fps || 30) : 1); }
    else if (e.key === "i" || e.key === "I") setIn(curT(), true);
    else if (e.key === "o" || e.key === "O") setOut(curT(), true);
    else if (e.key === "p" || e.key === "P") previewFrame();
  });
  window.addEventListener("beforeunload", (e) => {
    if (S.uploading || activeTakes().length) e.preventDefault();
  });

  layoutTimeline();
  addFaceSet(false);
  loadConfig().then(restoreTakes);
  requestAnimationFrame(tick);
})();
