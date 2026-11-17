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
    photos: [], faceSeq: 0, facesBusy: false, usable: 0,
    inT: 0, outT: 0, t: 0,
    people: null, person: null, scanning: false,
    tier: localStorage.getItem("lr.tier") || "fast",
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
    if (!S.config.tiers.some((t) => t.key === S.tier)) S.tier = S.config.tiers[0].key;
    renderTiers();
    update();
  }

  // ------------------------------------------------------------------ 1. your face
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

  async function addPhotos(files) {
    const max = (S.config && S.config.max_photos) || 5;
    const imgs = [...files].filter(isImage);
    if (!imgs.length) return toast("Those aren't images. Use a JPG or PNG photo of your face.", true);
    const room = max - S.photos.length;
    if (room <= 0) return toast(`You can use up to ${max} photos. Remove one to add another.`, true);
    if (imgs.length > room) toast(`Only the first ${room} photo${room > 1 ? "s" : ""} were added (limit ${max}).`);
    for (const f of imgs.slice(0, room)) {
      const blob = await downscale(f);
      S.photos.push({ blob, url: URL.createObjectURL(blob), state: "busy", thumb: null, error: null });
    }
    renderFaces();
    syncFaces();
    warm();
  }

  async function syncFaces() {
    const seq = ++S.faceSeq;
    if (!S.photos.length) { S.usable = 0; S.facesBusy = false; renderFaces(); update(); return; }
    S.facesBusy = true;
    S.photos.forEach((p) => (p.state = "busy"));
    renderFaces();
    update();
    try {
      const id = await ensureJob();
      const form = new FormData();
      S.photos.forEach((p, i) => form.append("files", p.blob, `${i}.jpg`));
      const r = await task("POST", `/api/jobs/${id}/faces`, form);
      if (seq !== S.faceSeq) return; // a newer change superseded this one
      r.photos.forEach((res, i) => {
        const p = S.photos[i];
        if (!p) return;
        p.state = res.ok ? "ok" : "bad";
        p.thumb = res.thumb || null;
        p.error = res.error || null;
      });
      S.usable = r.usable;
      if (!r.usable) toast("We couldn't find a usable face. Try a clear, front-facing photo.", true);
    } catch (e) {
      if (seq !== S.faceSeq) return;
      S.photos.forEach((p) => (p.state = "bad"));
      S.usable = 0;
      toast(e.message, true);
    }
    S.facesBusy = false;
    renderFaces();
    update();
  }

  function renderFaces() {
    const list = $("faceList");
    list.textContent = "";
    S.photos.forEach((p, i) => {
      const li = el("li", `face-item ${p.state}`);
      const img = el("img");
      img.src = p.thumb || p.url;
      img.alt = p.state === "ok" ? `Your face, photo ${i + 1}` : `Photo ${i + 1}`;
      li.append(img);
      if (p.state !== "busy") li.append(el("span", "face-state", p.state === "ok" ? "✓" : "!"));
      const rm = el("button", "remove", "×");
      rm.type = "button";
      rm.setAttribute("aria-label", `Remove photo ${i + 1}`);
      rm.onclick = (ev) => {
        ev.preventDefault();
        URL.revokeObjectURL(p.url);
        S.photos.splice(i, 1);
        renderFaces();
        syncFaces();
      };
      li.append(rm);
      if (p.error) li.title = p.error;
      list.append(li);
    });
    const errs = S.photos.filter((p) => p.error);
    if (errs.length) list.append(el("li", "face-note", errs.map((p) => p.error)[0]));
    $("faceDrop").classList.toggle("has-faces", S.photos.length > 0);
    $("faceDrop").querySelector("strong").textContent = S.photos.length ? "Add a photo" : "Drop a photo of your face";
    $("addFaceBtn").hidden = !S.photos.length || S.photos.length >= ((S.config && S.config.max_photos) || 5);
  }

  // camera
  let camStream = null;
  async function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return toast("This browser can't use the camera here. Upload a photo instead.", true);
    }
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } }, audio: false });
    } catch {
      return toast("Camera access was blocked. Allow it in your browser settings or upload a photo.", true);
    }
    $("camVideo").srcObject = camStream;
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
    addPhotos([new File([blob], "camera.jpg", { type: "image/jpeg" })]);
  }

  // ------------------------------------------------------------------ 2. video
  const video = $("video");

  async function setVideo(file) {
    if (!isVideo(file)) return toast("That isn't a video file. Use an MP4 or MOV.", true);
    if (S.uploading) return toast("Wait for the current upload to finish.", true);
    if (S.videoUrl) URL.revokeObjectURL(S.videoUrl);
    stopPolling();
    Object.assign(S, {
      videoFile: file, videoUrl: URL.createObjectURL(file), playable: false, meta: null,
      people: null, person: null, takes: [], take: null, duration: 0, inT: 0, outT: 0, t: 0,
      filmstripDone: false,
    });
    $("filmstrip").getContext("2d").clearRect(0, 0, $("filmstrip").width, $("filmstrip").height);
    $("scanBtn").textContent = "Find people";
    showSource();
    renderCast();
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
      const form = new FormData();
      form.append("file", file, file.name || "video.mp4");
      S.meta = await upload(`/api/jobs/${id}/video`, form, setUploadPct);
      S.fps = S.meta.fps || 30;
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
    const keepSel = S.outT > 0 && S.outT <= d;
    S.duration = d;
    if (!keepSel) { S.inT = 0; S.outT = d; }
    layoutTimeline();
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
  const minSel = () => Math.max(2 / (S.fps || 30), 0.1);
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
    $("selReadout").innerHTML = S.duration
      ? `Rendering <b>${tc(S.inT)}</b> to <b>${tc(S.outT)}</b>, ${len.toFixed(2)} s of ${tc(S.duration)}. Drag the green handles or press I and O at the playhead.`
      : "Drop a video to choose the section you want.";
  }
  function xToT(clientX) {
    const r = tl.getBoundingClientRect();
    return Math.min(Math.max(0, (clientX - r.left) / r.width), 1) * S.duration;
  }
  function setIn(t) { S.inT = Math.min(Math.max(0, t), S.outT - minSel()); layoutTimeline(); }
  function setOut(t) { S.outT = Math.max(Math.min(S.duration, t), S.inT + minSel()); layoutTimeline(); }

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

  // ------------------------------------------------------------------ 3. people
  async function scan() {
    if (!S.meta) return;
    const btn = $("scanBtn");
    busy(btn, true);
    S.scanning = true;
    update();
    try {
      const r = await task("POST", `/api/jobs/${S.jobId}/scan`, { start: S.inT, end: S.outT });
      S.people = r.people;
      if (!r.people.length) {
        S.person = null;
        toast("No faces found in this section. Try a section where faces are clearly visible.", true);
      } else {
        S.person = r.people.length === 1 ? r.people[0].id : null;
      }
      renderCast();
    } catch (e) {
      toast(e.message, true);
    }
    busy(btn, false);
    S.scanning = false;
    btn.textContent = "Scan again";
    update();
  }

  function renderCast() {
    const cast = $("cast");
    cast.textContent = "";
    if (!S.people) {
      cast.append(el("li", "cast-empty", "Find the people in your section, then pick the one you're replacing."));
      renderAppearances();
      return;
    }
    if (!S.people.length) {
      cast.append(el("li", "cast-empty", "Nobody found in this section."));
      renderAppearances();
      return;
    }
    const opts = S.people.map((p) => ({
      id: p.id, img: p.thumb, label: `On screen ${Math.round(p.share * 100)}%`, person: p,
    }));
    opts.push({ id: "all", img: null, label: "Every face" });
    opts.forEach((o) => {
      const li = el("li");
      const b = el("button", `cast-member${o.id === "all" ? " cast-everyone" : ""}`);
      b.type = "button";
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(S.person === o.id));
      b.setAttribute("aria-label", o.id === "all" ? "Replace every face" : `Person, ${o.label.toLowerCase()}`);
      const face = el("span", "cast-face");
      if (o.img) { const img = el("img"); img.src = o.img; img.alt = ""; face.append(img); } else face.textContent = "All";
      b.append(face, el("span", "cast-label", o.label));
      b.onclick = () => {
        S.person = o.id;
        renderCast();
        update();
        // jump to where this person shows up so the preview lands on them
        if (o.person && o.person.times.length) {
          const t = o.person.times.find((x) => x >= S.inT && x <= S.outT) ?? o.person.times[0];
          seek(t);
        }
      };
      li.append(b);
      cast.append(li);
    });
    renderAppearances();
  }
  function renderAppearances() {
    const track = $("appearTrack");
    track.textContent = "";
    const p = S.people && S.people.find((x) => x.id === S.person);
    if (!p || !S.duration) return;
    p.times.forEach((t) => {
      const i = el("i");
      i.style.left = `${(t / S.duration) * 100}%`;
      track.append(i);
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

  // ------------------------------------------------------------------ gating
  function blocker(forRender) {
    if (!S.videoFile) return "Add the ad video.";
    if (S.uploading) return `Uploading video ${Math.round(S.uploadPct * 100)}%`;
    if (!S.meta) return "The video didn't upload. Drop it again.";
    if (S.facesBusy) return "Checking your photo";
    if (!S.usable) return "Add a photo of your face.";
    if (!S.person) return S.people ? "Pick who you're replacing." : "Find people in the section, then pick who you're replacing.";
    if (forRender && !$("consent").checked) return "Tick the permission box to render.";
    return "";
  }
  function update() {
    const preBlock = blocker(false), renBlock = blocker(true);
    const rendering = S.take && ["queued", "rendering", "finalizing"].includes(S.take.state);
    $("previewBtn").disabled = !!preBlock || S.previewing;
    $("renderBtn").disabled = !!renBlock || rendering;
    $("scanBtn").disabled = !S.meta || S.scanning;
    $("transport").setAttribute("aria-disabled", String(!S.duration));
    $("whyDisabled").textContent = rendering ? "" : renBlock;
    const steps = document.querySelectorAll("h2 .step");
    steps[0].classList.toggle("done", S.usable > 0);
    steps[1].classList.toggle("done", !!S.person);
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
        { t, person: S.person, tier: S.tier, vr: $("vrToggle").checked });
      $("splitBefore").src = r.before;
      $("splitAfter").src = r.after;
      showStage("preview", `Preview at ${tc(r.t)}, ${tierLabel(S.tier)}, ${(r.ms / 1000).toFixed(1)} s`);
      setSplit(50);
      if (!r.swapped) {
        toast(r.faces ? "The person you picked isn't in this frame. Move the playhead to a frame where they're visible."
          : "No faces in this frame. Move the playhead to a frame with a face.", true, 6000);
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
    $("split").hidden = mode !== "preview";
    $("stageBadge").hidden = !badge;
    $("stageBadge").textContent = badge || "";
    $("stageClose").hidden = mode === "source";
    if (mode !== "result") setCompare(false);
  }
  function showSource() {
    if (S.mode === "result") {
      video.removeAttribute("controls");
      video.src = S.videoUrl;
      video.currentTime = S.t;
      if (!S.playable) { video.hidden = true; $("posterFallback").hidden = false; }
    }
    showStage("source");
  }
  function showResult(take) {
    S.take = take;
    $("posterFallback").hidden = true;
    video.hidden = false;
    video.pause();
    video.src = `/api/renders/${take.rid}/video`;
    video.setAttribute("controls", "");
    showStage("result", `Take ${take.n}, ${tierLabel(take.tier)}`);
    $("compareBtn").disabled = !S.playable;
    $("compareBtn").title = S.playable ? "" : "Your browser can't play the original file, so it can't be compared here.";
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
      const body = { start: S.inT, end: S.outT, person: S.person, tier: S.tier, vr: $("vrToggle").checked, consent: true };
      const r = await api("POST", `/api/jobs/${S.jobId}/render`, body);
      const take = { n: S.takes.length + 1, rid: r.render_id, tier: S.tier, start: S.inT, end: S.outT, state: "queued", lanes: r.lanes };
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

  function stopPolling() { clearTimeout(S.pollTimer); S.pollTimer = null; }
  let liveAt = 0;
  async function poll(take) {
    stopPolling();
    try {
      const st = await api("GET", `/api/renders/${take.rid}`);
      take.state = st.state;
      take.status = st;
      if (S.take === take) {
        renderPanel(take);
        if (Date.now() - liveAt > 1800 && st.state === "rendering") { liveAt = Date.now(); refreshLive(take); }
      }
      if (st.state === "done") {
        renderTakes();
        update();
        if (S.take === take) showResult(take);
        toast(`Take ${take.n} is ready.`);
        return;
      }
      if (st.state === "error") { renderTakes(); update(); return; }
    } catch (e) {
      if (S.take === take) $("renderMsg").textContent = `Lost contact with the server, retrying. (${e.message})`;
    }
    S.pollTimer = setTimeout(() => poll(take), 1000);
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
    };
    $("renderTitle").textContent = titles[st.state] || st.state;
    $("tally").className = `tally ${st.state === "done" ? "done" : st.state === "error" ? "" : "live"}`;
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
    $("stCover").textContent = st.face_coverage == null ? "-" : `${Math.round(st.face_coverage * 100)}% of frames`;
    const msg = $("renderMsg");
    msg.classList.toggle("err", st.state === "error");
    if (st.state === "error") msg.textContent = st.error || "Something went wrong. Try rendering again.";
    else if (st.state === "done") {
      const mb = st.size ? `${(st.size / 1048576).toFixed(1)} MB` : "";
      msg.textContent = `${(take.end - take.start).toFixed(1)} s rendered in ${secs(st.elapsed)}${mb ? `, ${mb}` : ""}.`;
      if (st.face_coverage != null && st.face_coverage < 0.6) {
        msg.textContent += " Your face was only found in part of the section. Check the person you picked or try Premium for small or turned faces.";
      }
    } else msg.textContent = st.state === "rendering" && !anyActive && S.config.backend === "modal"
      ? "GPUs take up to a minute to start when they've been idle." : "";
    const done = st.state === "done";
    $("resultActions").hidden = !done;
    if (done) $("downloadBtn").href = `/api/renders/${take.rid}/video?download=1`;
    if (done) $("liveFrame").hidden = true;
  }

  function renderTakes() {
    $("takesWrap").hidden = S.takes.length < 2;
    const ol = $("takes");
    ol.textContent = "";
    S.takes.forEach((t) => {
      const li = el("li");
      const b = el("button");
      b.type = "button";
      if (S.take === t) b.setAttribute("aria-current", "true");
      const state = t.state === "done" ? `${(t.end - t.start).toFixed(1)} s` : t.state === "error" ? "Failed" : "Rendering";
      b.append(el("span", "", `Take ${t.n}, ${tierLabel(t.tier)}`), el("span", "", state));
      b.onclick = () => {
        if (t.state === "done") showResult(t);
        else { S.take = t; showPanelFor(t); renderTakes(); if (t.state !== "error") poll(t); }
      };
      li.append(b);
      ol.append(li);
    });
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
  bindDrop($("faceDrop"));
  bindDrop($("videoDrop"));
  $("faceInput").addEventListener("change", (e) => { addPhotos(e.target.files); e.target.value = ""; });
  $("videoInput").addEventListener("change", (e) => { if (e.target.files[0]) setVideo(e.target.files[0]); e.target.value = ""; });
  $("addFaceBtn").onclick = () => $("faceInput").click();
  $("cameraBtn").onclick = openCamera;
  $("camCancel").onclick = closeCamera;
  $("camShoot").onclick = shoot;

  // Drop anywhere: videos go to the stage, images to your face.
  let dragDepth = 0;
  const zones = () => [$("faceDrop"), $("videoDrop")];
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
    if (imgs.length) addPhotos(imgs);
    if (!vids.length && !imgs.length && files.length) toast("Drop a video (MP4, MOV) or a photo (JPG, PNG).", true);
  });

  $("playBtn").onclick = togglePlay;
  $("stepBack").onclick = () => step(-1);
  $("stepFwd").onclick = () => step(1);
  $("markIn").onclick = () => setIn(curT());
  $("markOut").onclick = () => setOut(curT());
  $("scanBtn").onclick = scan;
  $("previewBtn").onclick = previewFrame;
  $("renderBtn").onclick = startRender;
  $("consent").onchange = update;
  $("stageClose").onclick = showSource;
  $("compareBtn").onclick = () => setCompare(!S.comparing);

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select") || !$("camera").hidden || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape" && S.mode !== "source") return showSource();
    if (!S.duration) return;
    const onButton = e.target.closest("button, [role=slider]");
    if (e.key === " " && !onButton) { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowLeft" && !onButton) { e.preventDefault(); step(e.shiftKey ? -(S.fps || 30) : -1); }
    else if (e.key === "ArrowRight" && !onButton) { e.preventDefault(); step(e.shiftKey ? (S.fps || 30) : 1); }
    else if (e.key === "i" || e.key === "I") setIn(curT());
    else if (e.key === "o" || e.key === "O") setOut(curT());
    else if (e.key === "p" || e.key === "P") previewFrame();
  });
  window.addEventListener("beforeunload", (e) => {
    if (S.uploading || (S.take && ["queued", "rendering", "finalizing"].includes(S.take.state))) e.preventDefault();
  });

  layoutTimeline();
  renderFaces();
  loadConfig();
  requestAnimationFrame(tick);
})();
