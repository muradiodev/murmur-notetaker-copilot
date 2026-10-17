const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ i18n
// Strings live in i18n/<code>.js. Add a language: create the file and list it here.
const LANGUAGES = { en: "English" };
let LANG_CODE = "en";

function t(key, vars = {}) {
  const pack = window.I18N[LANG_CODE] || {};
  let s = key in pack ? pack[key] : (key in window.I18N.en ? window.I18N.en[key] : key);
  if (typeof s === "string") s = s.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`));
  return s;
}

function applyI18n(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  for (const attr of ["placeholder", "title", "aria-label"]) {
    const data = "i18n" + attr.replace(/(^|-)(\w)/g, (_, __, c) => c.toUpperCase());
    root.querySelectorAll(`[data-i18n-${attr}]`).forEach((el) => el.setAttribute(attr, t(el.dataset[data])));
  }
  document.documentElement.lang = LANG_CODE;
}

function loadLanguage(code) {
  return new Promise((resolve) => {
    if (code === "en" || window.I18N[code]) { LANG_CODE = window.I18N[code] ? code : "en"; return resolve(); }
    const s = document.createElement("script");
    s.src = `i18n/${code}.js`;
    s.onload = () => { LANG_CODE = window.I18N[code] ? code : "en"; resolve(); };
    s.onerror = () => { LANG_CODE = "en"; resolve(); };
    document.head.appendChild(s);
  });
}

// ------------------------------------------------------------------ constants & state

// Muted, distinguishable hues used only for speaker names and swatches.
const SPEAKER_COLORS = ["#4263eb", "#d9480f", "#2b8a3e", "#ae3ec9", "#0c8599", "#c2255c", "#5c940d", "#e67700"];
const LANG = { az: "AZ", de: "DE", en: "EN" };
const SENT = {
  positive: { key: "sent.positive", icon: "ph-smiley", color: "var(--pos)" },
  neutral: { key: "sent.neutral", icon: "ph-smiley-meh", color: "var(--neu)" },
  negative: { key: "sent.negative", icon: "ph-smiley-sad", color: "var(--neg)" },
};

const state = {
  meetings: [],
  current: null,
  recordingId: null,
  tab: "transcript",
  query: "",
  tQuery: "",
  sentFilter: null,
  summaryRunning: new Set(),
  progress: {},
  step: {},
  pendingDelete: false,
  engine: { state: "loading" },
};

// ------------------------------------------------------------------ helpers

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function toast(msg, err = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast" + (err ? " err" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.add("hidden"), 4500);
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (a, total) => (total ? Math.round((a / total) * 100) : 0);

function clock(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function minutes(sec) {
  const m = Math.round((sec || 0) / 60);
  return m < 1 ? t("time.seconds", { n: Math.round(sec || 0) }) : t("time.minutes", { n: m });
}

const hhmm = (d) => `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;

function fmtDate(iso) {
  const d = new Date(iso);
  return `${d.getDate()} ${t("months")[d.getMonth()]}, ${hhmm(d)}`;
}

function dayGroup(iso) {
  const d = new Date(iso), now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(now) - day(d)) / 86400000);
  if (diff === 0) return t("date.today");
  if (diff === 1) return t("date.yesterday");
  if (diff < 7) return t("date.thisWeek");
  return `${t("months")[d.getMonth()]} ${d.getFullYear()}`;
}

function speakerColor(key) {
  if (key === "mic" || key === "system") return "var(--text-3)";
  const n = parseInt(String(key).replace(/\D/g, ""), 10) || 1;
  return SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.length];
}

const speakerName = (m, key) => (m.speakers && m.speakers[key]) || key;

function speakerKeys(m) {
  const num = (k) => parseInt(k.replace(/\D/g, ""), 10) || 0;
  return Object.keys(m.speakers || {}).sort((a, b) => num(a) - num(b) || a.localeCompare(b));
}

function highlight(text, q) {
  const safe = esc(text);
  if (!q) return safe;
  const re = new RegExp(esc(q).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  return safe.replace(re, (m) => `<mark>${m}</mark>`);
}

// Minimal Markdown for the summary: headings, lists, task items, bold.
// The first "# title" line is skipped because the title is already the page header.
function markdown(md) {
  const lines = esc(md).split(/\r?\n/);
  let html = "", inList = false, firstPara = true;
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  const close = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if (/^#\s/.test(line)) { close(); continue; }
    if ((m = line.match(/^#{2,3}\s+(.*)/))) { close(); html += `<h2>${inline(m[1])}</h2>`; }
    else if ((m = line.match(/^\s*[-*]\s+\[( |x)\]\s+(.*)/i))) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li class="task"><i class="ph ${m[1].trim() ? "ph-check-square" : "ph-square"}"></i><span>${inline(m[2])}</span></li>`;
    }
    else if ((m = line.match(/^\s*[-*]\s+(.*)/))) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inline(m[1])}</li>`; }
    else if (line.trim()) { close(); html += `<p${firstPara ? ' class="lead"' : ""}>${inline(line)}</p>`; firstPara = false; }
    else close();
  }
  close();
  return html;
}

const sentimentOf = (seg) => (state.current && state.current.sentiments && state.current.sentiments[seg.id]) || "neutral";
const sortedSegs = (m) => [...m.segments].sort((a, b) => a.start - b.start);

// Consecutive segments of one speaker form a turn: "Speaker 1  00:12 - 00:45".
// maxGap: a pause longer than this starts a new block even for the same speaker (live view).
function turnsOf(segs, maxGap = Infinity) {
  const turns = [];
  for (const seg of segs) {
    const last = turns[turns.length - 1];
    if (last && last.speaker === seg.speaker && seg.start - last.end <= maxGap) { last.segs.push(seg); last.end = Math.max(last.end, seg.end); }
    else turns.push({ speaker: seg.speaker, start: seg.start, end: seg.end, segs: [seg] });
  }
  return turns;
}

// Each word is its own span with its own timestamp, so clicking any word seeks to it.
function sentenceHtml(s, query, filter) {
  const hit = filter && sentimentOf(s) === filter ? ` hit-${filter}` : "";
  if (s.edited) {
    const tip = esc(t("transcript.edited", { raw: s.raw_text || "" }));
    return `<span class="sent edited${hit}" data-id="${s.id}" data-start="${s.start}" data-end="${s.end}" data-seek="${s.start}" title="${tip}">${highlight(s.text, query)}</span>`;
  }
  if (s.corrected) {
    const tip = esc(t("transcript.corrected", { raw: s.raw_text || "" }));
    return `<span class="sent corrected${hit}" data-id="${s.id}" data-start="${s.start}" data-end="${s.end}" data-seek="${s.start}" title="${tip}">${highlight(s.text, query)}</span>`;
  }
  if (s.words && s.words.length) {
    const words = s.words.map(([a, b, w]) => `<span class="w" data-seek="${a}" data-ws="${a}" data-we="${b}">${highlight(w, query)}</span>`).join(" ");
    return `<span class="sent${hit}" data-id="${s.id}" data-start="${s.start}" data-end="${s.end}">${words}</span>`;
  }
  return `<span class="sent${hit}" data-id="${s.id}" data-start="${s.start}" data-end="${s.end}" data-seek="${s.start}">${highlight(s.text, query)}</span>`;
}

function turnHtml(m, turn, { query = "", filter = null, editable = false, live = false } = {}) {
  const langs = [...new Set(turn.segs.map((s) => LANG[s.lang] || s.lang).filter(Boolean))].join(" ");
  const name = esc(speakerName(m, turn.speaker));
  const who = editable
    ? `<button class="who" data-rename="${esc(turn.speaker)}" title="${esc(t("transcript.renameHint"))}">${name}</button>`
    : `<span class="who">${name}</span>`;
  const edit = editable
    ? `<button class="edit-turn" data-edit-turn title="${esc(t("transcript.renameHint"))}" aria-label="${esc(t("transcript.renameHint"))}"><i class="ph ph-pencil-simple"></i></button>` : "";
  return `<article class="turn" style="--spk:${speakerColor(turn.speaker)}">
    <div>
      <div class="turn-head">
        ${who}
        ${live ? `<span class="range mono static">${clock(turn.start)} - ${clock(turn.end)}</span>`
               : `<button class="range mono" data-seek="${turn.start}" title="${esc(t("transcript.playFrom"))}">${clock(turn.start)} - ${clock(turn.end)}</button>`}
        ${langs ? `<span class="lang">${langs}</span>` : ""}
        ${edit}
      </div>
      <p class="text">${turn.segs.map((s) => sentenceHtml(s, query, filter)).join(" ")}</p>
    </div>
  </article>`;
}

// ------------------------------------------------------------------ copy

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    toast(t("copy.copied"));
  } catch (err) { toast(t("copy.failed", { message: err.message }), true); }
}

// "Speaker 1 (00:12 - 00:45)" followed by what they said, one block per turn.
function transcriptAsText(m) {
  return turnsOf(sortedSegs(m)).map((turn) =>
    `${speakerName(m, turn.speaker)} (${clock(turn.start)} - ${clock(turn.end)})\n${turn.segs.map((s) => s.text).join(" ")}`).join("\n\n");
}

$("copyTranscriptBtn").addEventListener("click", () => state.current && copyText(transcriptAsText(state.current)));
$("copyLiveBtn").addEventListener("click", () => state.current && copyText(transcriptAsText(state.current)));

// ------------------------------------------------------------------ routing
// #/  home   ·   #/meeting/<id>   ·   #/settings

function showView(name) {
  for (const v of ["homeView", "liveView", "meetingView", "settingsView", "archiveView", "cpSessionsView", "cpLibView"]) $(v).classList.toggle("hidden", v !== name);
  $("settingsBtn").classList.toggle("active", name === "settingsView" || name === "archiveView");
  renderNowPlaying();
}

function goMeeting(id) {
  const hash = `#/meeting/${id}`;
  if (location.hash === hash) openMeeting(id);
  else location.hash = hash;
}

async function route() {
  const h = location.hash;
  closeMenu();
  if (h === "#/settings") return showSettings();
  if (h === "#/archive") return showArchive();
  if (h.startsWith("#/copilot")) return cpRoute(h);
  const m = h.match(/^#\/meeting\/(.+)$/);
  if (m) {
    try { await openMeeting(decodeURIComponent(m[1])); }
    catch { location.hash = "#/"; }
    return;
  }
  state.current = null;
  setMode("notes");
  renderList();
  showView("homeView");
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ library

async function loadMeetings() {
  state.meetings = await api("/api/meetings");
  renderList();
  if (!$("archiveView").classList.contains("hidden")) renderArchive();
  if (!$("settingsView").classList.contains("hidden")) renderArchiveCount();
}

function itemHtml(m) {
  const active = state.current && state.current.id === m.id ? " active" : "";
  const status = m.status === "recording" ? `<span class="rec-dot" aria-label="${esc(t("list.recording"))}"></span>` : "";
  const sub = m.status === "processing" ? t("list.processing") : `${hhmm(new Date(m.created_at))}, ${minutes(m.duration)}`;
  const busy = m.status !== "ready";
  const confirm = state.confirmDelete === m.id;
  const archiveKey = m.archived ? "list.unarchive" : "list.archive";
  const actions = busy ? "" : `<span class="item-actions">
      <button class="icon-btn" data-archive="${m.id}" title="${esc(t(archiveKey))}" aria-label="${esc(t(archiveKey))}">
        <i class="ph ${m.archived ? "ph-arrow-u-up-left" : "ph-archive"}"></i></button>
      <button class="icon-btn danger${confirm ? " confirm" : ""}" data-delete="${m.id}" title="${esc(t(confirm ? "list.deleteConfirm" : "list.delete"))}" aria-label="${esc(t("list.delete"))}">
        <i class="ph ${confirm ? "ph-check" : "ph-trash"}"></i></button>
    </span>`;
  return `<a class="item${active}${confirm ? " confirming" : ""}" href="#/meeting/${encodeURIComponent(m.id)}">
      <div class="item-title">${status}<span>${esc(m.title)}</span></div>
      <div class="item-sub">${esc(confirm ? t("list.deleteConfirm") : sub)}</div>
      ${actions}
    </a>`;
}

function renderList() {
  const q = state.query.toLowerCase();
  const match = (m) => !q || m.title.toLowerCase().includes(q);
  const notes = state.meetings.filter((m) => m.mode !== "copilot");
  const live = notes.filter((m) => !m.archived && match(m));
  const archived = notes.filter((m) => m.archived && match(m));
  cpRenderRecent();
  cpRenderNewButton();
  let html = "", group = null;
  for (const m of live) {
    const g = dayGroup(m.created_at);
    if (g !== group) { html += `<div class="group-label">${esc(g)}</div>`; group = g; }
    html += itemHtml(m);
  }
  if (archived.length) {
    const open = state.showArchived || !!q;
    html += `<button class="group-toggle" data-toggle-archived><i class="ph ${open ? "ph-caret-down" : "ph-caret-right"}"></i>${esc(t("list.archived", { n: archived.length }))}</button>`;
    if (open) html += archived.map(itemHtml).join("");
  }
  $("meetingList").innerHTML = html || `<p class="library-empty">${esc(t(q ? "list.noMatch" : "list.empty"))}</p>`;
}

// Hover actions in the sidebar: archive / unarchive, and delete with a second click to confirm.
$("meetingList").addEventListener("click", async (e) => {
  const toggle = e.target.closest("[data-toggle-archived]");
  if (toggle) { state.showArchived = !state.showArchived; return renderList(); }
  const arch = e.target.closest("[data-archive]");
  const del = e.target.closest("[data-delete]");
  if (!arch && !del) { state.confirmDelete = null; return; }
  e.preventDefault();
  e.stopPropagation();
  if (arch) {
    const id = arch.dataset.archive;
    const m = state.meetings.find((x) => x.id === id);
    try {
      await api(`/api/meetings/${id}`, { method: "PATCH", body: { archived: !m.archived } });
      toast(t(m.archived ? "list.unarchived" : "list.archivedToast"));
      await loadMeetings();
    } catch (err) { toast(err.message, true); }
    return;
  }
  const id = del.dataset.delete;
  if (state.confirmDelete !== id) {
    state.confirmDelete = id;
    renderList();
    clearTimeout(state.confirmTimer);
    state.confirmTimer = setTimeout(() => { state.confirmDelete = null; renderList(); }, 4000);
    return;
  }
  state.confirmDelete = null;
  try {
    await api(`/api/meetings/${id}`, { method: "DELETE" });
    if (playingId() === id) $("npClose").click();
    if (state.current && state.current.id === id) { state.current = null; location.hash = "#/"; }
    await loadMeetings();
    toast(t("toast.deleted"));
  } catch (err) { toast(err.message, true); }
});

$("search").addEventListener("input", (e) => { state.query = e.target.value; renderList(); });

// ------------------------------------------------------------------ meeting

async function openMeeting(id, { keepTab = false } = {}) {
  const m = await api(`/api/meetings/${id}`);
  const switched = !state.current || state.current.id !== id;
  state.current = m;
  closeMenu();
  setMode(m.mode === "copilot" ? "copilot" : "notes");
  if (m.status === "recording") {
    showView("liveView");
    $("liveTitle").value = m.title;
    renderShareButton();
    cpLiveLayout(m);
    renderLive();
  } else {
    showView("meetingView");
    if (switched) {
      state.tQuery = ""; $("tSearch").value = ""; state.sentFilter = null;
      if (!keepTab) state.tab = "transcript";
    }
    renderMeeting();
    setupPlayer(m);
    setupStage(m);
  }
  renderList();
}

function renderMeeting() {
  const m = state.current;
  $("meetingTitle").value = m.title;
  const people = speakerKeys(m).filter((k) => m.segments.some((s) => s.speaker === k)).length;
  const langs = [...new Set(m.segments.map((s) => LANG[s.lang] || s.lang).filter(Boolean))].join(", ");
  $("meetingMeta").innerHTML = [fmtDate(m.created_at), minutes(m.duration), people ? t("meeting.speakersCount", { n: people }) : "", langs]
    .filter(Boolean).map((x) => `<span>${esc(x)}</span>`).join("");
  $("exportTxt").href = `/api/meetings/${m.id}/export/txt`;
  $("exportSrt").href = `/api/meetings/${m.id}/export/srt`;
  $("exportMd").href = `/api/meetings/${m.id}/export/md`;
  $("exportMd").classList.toggle("hidden", !m.summary);
  $("retranscribeBtn").disabled = m.status !== "ready" || !m.has_audio;
  renderProgress();
  renderTabs();
  renderOverview();
  renderTranscript();
  renderSentiment();
  renderSpeakers();
  cpRenderAnswersPanel();
}

function renderProgress() {
  const m = state.current;
  const box = $("progress");
  const busy = m.status === "processing";
  box.classList.toggle("hidden", !busy && !m.error);
  if (!busy) {
    box.classList.remove("indeterminate");
    $("progressBar").style.width = "0";
    $("progressText").textContent = m.error || "";
    return;
  }
  const p = state.progress[m.id];
  const step = state.step[m.id];
  const diarizing = step === "diarizing" || step === "echo";
  box.classList.toggle("indeterminate", diarizing || p === undefined);
  $("progressBar").style.width = `${p || 0}%`;
  $("progressText").textContent = diarizing ? t("progress.diarizing")
    : m.source ? t("progress.importing", { p: p || 0 })
    : step === "echo" ? t("progress.echo")
    : step === "refining" ? t("progress.refining", { p: p || 0 })
    : step === "transcribing" ? t("progress.transcribing", { p: p || 0 })
    : t("progress.finishing");
}

function renderTabs() {
  document.querySelectorAll("#tabs [data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === state.tab)));
  document.querySelectorAll(".meeting-side .panel").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== state.tab));
}

function setTab(tab) {
  state.tab = tab;
  renderTabs();
  document.querySelector(".meeting-side .panels").scrollTop = 0;
}

$("tabs").addEventListener("click", (e) => {
  const b = e.target.closest("[data-tab]");
  if (b) setTab(b.dataset.tab);
});

// ---- overview

function emptyState(titleKey, bodyKey, action = "") {
  return `<div class="empty"><h3>${esc(t(titleKey))}</h3><p>${esc(t(bodyKey))}</p>${action}</div>`;
}

function renderOverview() {
  const m = state.current;
  const box = $("overviewPanel");
  const createBtn = (key) => `<button class="btn btn-primary" data-action="summary"><i class="ph ph-sparkle"></i>${esc(t(key))}</button>`;
  if (state.summaryRunning.has(m.id)) {
    box.innerHTML = `<div class="skeleton">${"<i></i>".repeat(9)}</div><p class="note">${esc(t("overview.running"))}</p>`;
  } else if (m.status !== "ready") {
    box.innerHTML = emptyState("overview.processingTitle", "overview.processingBody");
  } else if (!m.segments.length) {
    box.innerHTML = emptyState("overview.emptyTranscriptTitle", "overview.emptyTranscriptBody");
  } else if (!m.summary) {
    box.innerHTML = emptyState("overview.noSummaryTitle", "overview.noSummaryBody", createBtn("overview.create"));
  } else {
    const current = m.summary_lang || "";
    const auto = meetingLanguage(m);
    const langs = [["az", "settings.langAz"], ["de", "settings.langDe"], ["en", "settings.langEn"]];
    const options = langs.map(([code, key]) =>
      `<option value="${code}" ${code === current ? "selected" : ""}>${esc(t(key))}${code === auto ? ` (${esc(t("overview.meetingLang"))})` : ""}</option>`).join("");
    box.innerHTML = `<div class="overview-bar">
        <select id="summaryLang" aria-label="${esc(t("overview.language"))}" title="${esc(t("overview.language"))}">${options}</select>
        <span class="spacer"></span>
        <button class="btn btn-ghost btn-sm" data-action="copy-summary" title="${esc(t("copy.summaryHint"))}"><i class="ph ph-copy"></i>${esc(t("copy.copy"))}</button>
        <button class="btn btn-ghost btn-sm" data-action="summary-force"><i class="ph ph-arrow-counter-clockwise"></i>${esc(t("overview.regenerate"))}</button>
      </div>
      <div class="doc">${markdown(m.summary)}</div>`;
  }
}

// The language most of the meeting was spoken in, as the summary default.
function meetingLanguage(m) {
  const words = {};
  m.segments.forEach((s) => { if (LANG[s.lang]) words[s.lang] = (words[s.lang] || 0) + s.text.split(/\s+/).length; });
  return Object.keys(words).sort((a, b) => words[b] - words[a])[0] || "en";
}

async function startSummary({ language = null, force = false } = {}) {
  const id = state.current.id;
  try {
    const r = await api(`/api/meetings/${id}/summary`, { method: "POST", body: { language, force } });
    if (r.cached) return openMeeting(id, { keepTab: true });  // that language was already written
    state.summaryRunning.add(id);
    renderOverview();
  } catch (err) { toast(err.message, true); }
}

$("overviewPanel").addEventListener("change", (e) => {
  if (e.target.id === "summaryLang") startSummary({ language: e.target.value });
});

// ---- transcript

function renderTranscript() {
  const m = state.current;
  const q = state.tQuery.toLowerCase();
  const f = state.sentFilter;
  const turns = turnsOf(sortedSegs(m)).filter((turn) =>
    (!q || turn.segs.some((s) => s.text.toLowerCase().includes(q))) &&
    (!f || turn.segs.some((s) => sentimentOf(s) === f)));
  const box = $("transcript");
  box.classList.toggle("seekable", !!m.has_audio);
  const emptyKey = q || f ? "transcript.noMatch" : m.status === "processing" ? "transcript.preparing" : "transcript.empty";
  const editable = m.status === "ready";
  box.innerHTML = turns.map((turn) => turnHtml(m, turn, { query: state.tQuery, filter: f, editable })).join("") ||
    `<p class="muted">${esc(t(emptyKey))}</p>`;
  const chip = $("filterChip");
  chip.classList.toggle("hidden", !f);
  if (f) chip.querySelector("span").textContent = t("transcript.filter", { label: t(SENT[f].key).toLowerCase() });
}

$("tSearch").addEventListener("input", (e) => { state.tQuery = e.target.value; renderTranscript(); });
$("filterChip").addEventListener("click", () => { state.sentFilter = null; renderTranscript(); renderSentiment(); });

$("transcript").addEventListener("click", (e) => {
  const rename = e.target.closest("[data-rename]");
  if (rename) return startRename(rename);
  const editBtn = e.target.closest("[data-edit-turn]");
  if (editBtn) return startRename(editBtn.closest(".turn").querySelector("[data-rename]"));
  if (e.target.closest(".turn.editing")) return;  // typing, not seeking
  const target = e.target.closest("[data-seek]");
  if (!target || !state.current.has_audio) return;
  playCurrentAt(parseFloat(target.dataset.seek));
});

// Rename a speaker straight from the transcript; applies to every line of that speaker.
function startRename(btn) {
  const key = btn.dataset.rename;
  const input = document.createElement("input");
  input.className = "who-input";
  input.value = speakerName(state.current, key);
  btn.replaceWith(input);
  input.focus();
  input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const value = input.value.trim();
    if (save && value && value !== speakerName(state.current, key)) {
      try {
        const m = await api(`/api/meetings/${state.current.id}`, { method: "PATCH", body: { speakers: { [key]: value } } });
        Object.assign(state.current, { speakers: m.speakers, segments: m.segments });
        toast(t("spk.saved"));
      } catch (err) { toast(err.message, true); }
    }
    renderMeeting();
  };
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); finish(true); }
    if (ev.key === "Escape") { ev.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

// Edit mode for one turn: every sentence becomes editable and saves on blur or Enter.
function toggleEditTurn(turnEl) {
  const editing = !turnEl.classList.contains("editing");
  turnEl.classList.toggle("editing", editing);
  const btn = turnEl.querySelector("[data-edit-turn]");
  btn.innerHTML = editing ? esc(t("transcript.done")) : '<i class="ph ph-pencil-simple"></i>';
  turnEl.querySelector(".edit-hint")?.remove();
  turnEl.querySelectorAll(".sent").forEach((el) => {
    el.contentEditable = editing ? "plaintext-only" : "false";
    if (editing) el.dataset.original = el.textContent;
  });
  if (editing) {
    turnEl.querySelector(".text").insertAdjacentHTML("afterend", `<p class="edit-hint">${esc(t("transcript.editHint"))}</p>`);
    turnEl.querySelector(".sent")?.focus();
  } else {
    renderTranscript();
  }
}

async function saveSentence(el) {
  const text = el.textContent.replace(/\s+/g, " ").trim();
  if (text === (el.dataset.original || "").replace(/\s+/g, " ").trim()) return;
  try {
    const res = await api(`/api/meetings/${state.current.id}/segments/${el.dataset.id}`, { method: "PATCH", body: { text } });
    const segs = state.current.segments;
    const i = segs.findIndex((x) => x.id === el.dataset.id);
    if (res.segment) segs[i] = res.segment; else segs.splice(i, 1);
    el.dataset.original = text;
    toast(t("transcript.saved"));
  } catch (err) { toast(err.message, true); }
}

$("transcript").addEventListener("keydown", (e) => {
  const el = e.target.closest(".turn.editing .sent");
  if (!el) return;
  if (e.key === "Enter") { e.preventDefault(); el.blur(); }
  if (e.key === "Escape") { el.textContent = el.dataset.original || ""; el.blur(); }
});
$("transcript").addEventListener("focusout", (e) => {
  const el = e.target.closest(".turn.editing .sent");
  if (el) saveSentence(el);
});

// ---- sentiment

function renderSentiment() {
  const m = state.current;
  const box = $("sentimentPanel");
  if (!m.sentiments) {
    const action = m.status === "ready" && m.segments.length
      ? `<button class="btn btn-primary" data-action="summary"><i class="ph ph-sparkle"></i>${esc(t("sent.create"))}</button>` : "";
    box.innerHTML = emptyState("sent.noneTitle", "sent.noneBody", action);
    return;
  }
  const count = (segs) => {
    const c = { positive: 0, neutral: 0, negative: 0 };
    segs.forEach((s) => { c[sentimentOf(s)] += 1; });
    return c;
  };
  const split = (c, n) => `<div class="split">${["positive", "neutral", "negative"]
    .filter((k) => c[k]).map((k) => `<i class="${k}" style="width:${pct(c[k], n)}%"></i>`).join("")}</div>`;
  const all = count(m.segments), n = m.segments.length;
  const cards = ["positive", "neutral", "negative"].map((k) => `
    <button class="sent-card${state.sentFilter === k ? " on" : ""}" style="--k:${SENT[k].color}" data-sent="${k}">
      <span class="label"><i class="ph ${SENT[k].icon}"></i>${esc(t(SENT[k].key))}</span>
      <span class="value">${pct(all[k], n)}%</span>
      <span class="count">${esc(t("sent.sentences", { n: all[k] }))}</span>
    </button>`).join("");
  const rows = speakerKeys(m).map((k) => {
    const segs = m.segments.filter((s) => s.speaker === k);
    if (!segs.length) return "";
    const c = count(segs);
    return `<div class="sent-speaker" style="--spk:${speakerColor(k)}">
      <span class="name"><span class="swatch"></span>${esc(speakerName(m, k))}</span>
      ${split(c, segs.length)}
      <span class="nums">${pct(c.positive, segs.length)}% / ${pct(c.negative, segs.length)}%</span>
    </div>`;
  }).join("");
  box.innerHTML = `<div class="sent-summary">${cards}</div>
    <div class="sent-overall">${split(all, n)}</div>
    <p class="note">${esc(t("sent.hint"))}</p>
    <h3 class="section-title">${esc(t("sent.bySpeaker"))}</h3>${rows}`;
}

$("sentimentPanel").addEventListener("click", (e) => {
  const card = e.target.closest("[data-sent]");
  if (!card) return;
  state.sentFilter = card.dataset.sent;
  renderSentiment();
  renderTranscript();
  setTab("transcript");
});

// ---- speakers

function renderSpeakerCountOptions() {
  const opts = [`<option value="0">${esc(t("spk.auto"))}</option>`, `<option value="1">${esc(t("spk.person"))}</option>`];
  for (const n of [2, 3, 4, 5, 6, 8]) opts.push(`<option value="${n}">${esc(t("spk.people", { n }))}</option>`);
  $("numSpeakers").innerHTML = opts.join("");
}

function renderSpeakers() {
  const m = state.current;
  const stats = {};
  m.segments.forEach((s) => {
    const st = (stats[s.speaker] ||= { time: 0, words: 0, turns: 0 });
    st.time += Math.max(0, s.end - s.start);
    st.words += s.text.split(/\s+/).filter(Boolean).length;
  });
  turnsOf(sortedSegs(m)).forEach((turn) => { if (stats[turn.speaker]) stats[turn.speaker].turns += 1; });
  const total = Object.values(stats).reduce((a, s) => a + s.time, 0);
  const keys = speakerKeys(m).filter((k) => stats[k]).sort((a, b) => stats[b].time - stats[a].time);
  $("speakersList").innerHTML = keys.map((k) => {
    const st = stats[k], p = pct(st.time, total);
    const wpm = st.time > 10 ? Math.round(st.words / (st.time / 60)) : 0;
    const meta = t("spk.meta", { time: clock(st.time), turns: st.turns, words: st.words }) + (wpm ? t("spk.wpm", { n: wpm }) : "");
    return `<div class="spk-row" style="--spk:${speakerColor(k)}">
      <div class="spk-name"><span class="swatch"></span>
        <input data-key="${k}" value="${esc(speakerName(m, k))}" aria-label="${esc(t("spk.nameLabel"))}"></div>
      <div class="spk-pct">${p}%</div>
      <div class="talk"><i style="width:${Math.max(p, 1)}%"></i></div>
      <div class="spk-meta">${esc(meta)}</div>
    </div>`;
  }).join("") || emptyState("spk.noneTitle", "spk.noneBody");
  $("diarizeBtn").disabled = m.status !== "ready";
}

$("speakersList").addEventListener("change", async (e) => {
  const key = e.target.dataset.key;
  if (!key) return;
  try {
    const m = await api(`/api/meetings/${state.current.id}`, { method: "PATCH", body: { speakers: { [key]: e.target.value } } });
    Object.assign(state.current, { speakers: m.speakers, segments: m.segments });
    renderMeeting();
    toast(t("spk.saved"));
  } catch (err) { toast(err.message, true); }
});
$("speakersList").addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.blur(); });

$("diarizeBtn").addEventListener("click", async () => {
  const btn = $("diarizeBtn");
  btn.disabled = true;
  try {
    await api(`/api/meetings/${state.current.id}/diarize`, { method: "POST", body: { num_speakers: +$("numSpeakers").value } });
    await openMeeting(state.current.id, { keepTab: true });
    toast(t("spk.redone"));
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
});

// ---- title, menu, actions

async function saveTitle(value) {
  try {
    const m = await api(`/api/meetings/${state.current.id}`, { method: "PATCH", body: { title: value } });
    state.current.title = m.title;
    loadMeetings();
  } catch (err) { toast(err.message, true); }
}
for (const id of ["meetingTitle", "liveTitle"]) {
  $(id).addEventListener("change", (e) => saveTitle(e.target.value));
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.blur(); });
}

function closeMenu() {
  $("menu").classList.add("hidden");
  state.pendingDelete = false;
  $("deleteBtn").querySelector("span").textContent = t("menu.delete");
}
$("menuBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  const open = $("menu").classList.contains("hidden");
  closeMenu();
  if (open) $("menu").classList.remove("hidden");
});
document.addEventListener("click", (e) => { if (!e.target.closest(".menu-wrap")) closeMenu(); });

$("retranscribeBtn").addEventListener("click", async () => {
  closeMenu();
  try {
    await api(`/api/meetings/${state.current.id}/retranscribe`, { method: "POST" });
    state.progress[state.current.id] = 0;
    state.step[state.current.id] = "transcribing";
    toast(t("toast.retranscribing"));
    await openMeeting(state.current.id, { keepTab: true });
  } catch (err) { toast(err.message, true); }
});

// Two-step delete inside the menu instead of a blocking browser dialog.
$("deleteBtn").addEventListener("click", async (e) => {
  e.stopPropagation();
  if (!state.pendingDelete) {
    state.pendingDelete = true;
    $("deleteBtn").querySelector("span").textContent = t("menu.deleteConfirm");
    return;
  }
  try {
    await api(`/api/meetings/${state.current.id}`, { method: "DELETE" });
    const deletedId = state.current.id;
    state.current = null;
    closeMenu();
    if (playingId() === deletedId) $("npClose").click();
    await loadMeetings();
    location.hash = "#/";
    toast(t("toast.deleted"));
  } catch (err) { toast(err.message, true); }
});

// ------------------------------------------------------------------ player
// One global <audio>: playback keeps going while you browse other meetings or Settings.
// The bottom bar controls the meeting on screen; the top "now playing" bar appears when the
// audio belongs to a meeting you are not looking at.

const SPEEDS = [1, 1.5, 2];
const audioEl = $("audio");
const playingId = () => audioEl.dataset.meeting || null;
const viewingPlaying = () => state.current && state.current.id === playingId() && !$("meetingView").classList.contains("hidden");

function loadSpeed() {
  try { return SPEEDS.includes(+localStorage.getItem("mh.speed")) ? +localStorage.getItem("mh.speed") : 1; } catch { return 1; }
}

function setSpeed(rate) {
  audioEl.playbackRate = rate;
  $("screenVideo").playbackRate = rate;
  try { localStorage.setItem("mh.speed", String(rate)); } catch {}
  for (const id of ["speedBtn", "npSpeed"]) {
    $(id).textContent = `${rate}x`;
    $(id).classList.toggle("fast", rate !== 1);
  }
}
const nextSpeed = () => setSpeed(SPEEDS[(SPEEDS.indexOf(audioEl.playbackRate) + 1) % SPEEDS.length] || 1);
$("speedBtn").addEventListener("click", nextSpeed);
$("npSpeed").addEventListener("click", nextSpeed);

// Point the audio at a meeting (keeps the speed). Resolves when the duration is known.
function loadAudio(m) {
  if (playingId() === m.id && audioEl.src) return Promise.resolve();
  audioEl.src = `/api/meetings/${m.id}/audio`;
  audioEl.dataset.meeting = m.id;
  audioEl.dataset.title = m.title;
  audioEl.playbackRate = loadSpeed();
  return new Promise((resolve) => audioEl.addEventListener("loadedmetadata", resolve, { once: true }));
}

async function playCurrentAt(seconds = null) {
  const m = state.current;
  if (!m || !m.has_audio) return;
  await loadAudio(m);
  if (seconds !== null) audioEl.currentTime = seconds;
  audioEl.play();
}

function setupPlayer(m) {
  $("player").classList.toggle("hidden", !m.has_audio);
  if (!m.has_audio) return;
  // Only take over the audio when nothing else is playing.
  if (!playingId() || (audioEl.paused && playingId() !== m.id)) loadAudio(m);
  renderPlayer();
}

function renderPlayer() {
  const m = state.current;
  const mine = m && playingId() === m.id;
  const now = mine ? audioEl.currentTime : 0;
  const dur = mine && audioEl.duration ? audioEl.duration : (m ? m.duration : 0);
  $("curTime").textContent = clock(now);
  $("durTime").textContent = clock(dur);
  $("seek").value = dur ? Math.round((now / dur) * 1000) : 0;
  $("playBtn").innerHTML = mine && !audioEl.paused ? '<i class="ph ph-pause"></i>' : '<i class="ph ph-play"></i>';
  renderNowPlaying();
}

function renderNowPlaying() {
  const id = playingId();
  const active = id && (!audioEl.paused || audioEl.currentTime > 0);
  const show = !!active && !viewingPlaying();
  $("nowPlaying").classList.toggle("hidden", !show);
  if (!show) return;
  const meeting = state.meetings.find((x) => x.id === id);
  $("npTitle").textContent = meeting ? meeting.title : audioEl.dataset.title || "";
  $("npTitle").href = `#/meeting/${encodeURIComponent(id)}`;
  $("npTime").textContent = `${clock(audioEl.currentTime)} / ${clock(audioEl.duration)}`;
  $("npPlay").innerHTML = audioEl.paused ? '<i class="ph ph-play"></i>' : '<i class="ph ph-pause"></i>';
}

$("playBtn").addEventListener("click", () => {
  if (state.current && playingId() === state.current.id) audioEl.paused ? audioEl.play() : audioEl.pause();
  else playCurrentAt(0);
});
$("npPlay").addEventListener("click", () => (audioEl.paused ? audioEl.play() : audioEl.pause()));
$("npClose").addEventListener("click", () => {
  audioEl.pause();
  audioEl.removeAttribute("src");
  delete audioEl.dataset.meeting;
  audioEl.load();
  renderPlayer();
  if (state.current && state.current.has_audio && !$("meetingView").classList.contains("hidden")) loadAudio(state.current);
});
$("seek").addEventListener("input", async (e) => {
  const m = state.current;
  if (!m) return;
  await loadAudio(m);
  if (audioEl.duration) audioEl.currentTime = (e.target.value / 1000) * audioEl.duration;
  renderPlayer();
});
for (const ev of ["play", "pause", "loadedmetadata", "ended", "emptied"]) audioEl.addEventListener(ev, renderPlayer);
setSpeed(loadSpeed());

// Highlight the sentence being played, and inside it the exact word.
audioEl.addEventListener("timeupdate", () => {
  renderPlayer();
  if (!viewingPlaying()) {
    if (state.word) { state.word.classList.remove("now"); state.word = null; }
    return;
  }
  const now = audioEl.currentTime;
  let sentence = null;
  document.querySelectorAll("#transcript .sent").forEach((el) => {
    const on = now >= +el.dataset.start && now < +el.dataset.end + 0.25;
    el.classList.toggle("playing", on);
    if (on) sentence = el;
  });
  let word = null;
  if (sentence) {
    for (const w of sentence.querySelectorAll(".w")) {
      if (now >= +w.dataset.ws && now < +w.dataset.we + 0.15) { word = w; break; }
    }
  }
  if (state.word !== word) {
    if (state.word) state.word.classList.remove("now");
    if (word) word.classList.add("now");
    state.word = word;
  }
  if (sentence && !audioEl.paused && state.tab === "transcript" && !sentence.closest(".editing") && sentence !== state.lastSentence) {
    sentence.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
  state.lastSentence = sentence;
});

// ------------------------------------------------------------------ screen recording
// The browser records the chosen screen/window/tab and uploads 2-second chunks. The server stamps
// the meeting second the clip started at, so sharing can start or stop at any moment while the
// audio keeps recording, and playback lines the video up with the audio.

const screenRec = { stream: null, recorder: null, clip: null, uploads: Promise.resolve() };

function renderShareButton() {
  const on = !!screenRec.recorder;
  $("shareBtn").innerHTML = on
    ? `<i class="ph ph-monitor-play"></i><span>${esc(t("live.stopShare"))}</span>`
    : `<i class="ph ph-monitor"></i><span>${esc(t("live.shareScreen"))}</span>`;
  $("screenPreview").classList.toggle("hidden", !on || (state.current && state.current.mode === "copilot"));
  cpRenderScreen();
}

async function startScreen() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia || !window.MediaRecorder) {
    return toast(t("live.shareUnsupported"), true);
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 15, max: 30 }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,  // audio already comes from the microphone and system loopback
    });
  } catch (err) {
    if (err.name !== "NotAllowedError") toast(t("live.shareFailed", { message: err.message }), true);
    return;
  }
  try {
    const clip = await api("/api/record/screen/start", { method: "POST" });
    const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
    const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 1500000 });
    recorder.ondataavailable = (e) => {
      if (!e.data.size) return;
      // Chained so chunks arrive in order.
      screenRec.uploads = screenRec.uploads
        .then(() => fetch(`/api/record/screen/${clip.n}/chunk`, { method: "POST", body: e.data }))
        .catch(() => {});
    };
    stream.getVideoTracks()[0].addEventListener("ended", () => stopScreen());  // the browser's own "Stop sharing"
    Object.assign(screenRec, { stream, recorder, clip });
    recorder.start(2000);
    $("previewVideo").srcObject = stream;
    $("cpScreenVideo").srcObject = stream;
  } catch (err) {
    stream.getTracks().forEach((tr) => tr.stop());
    toast(t("live.shareFailed", { message: err.message }), true);
  }
  renderShareButton();
}

async function stopScreen() {
  const { recorder, stream, clip } = screenRec;
  if (!recorder) return;
  screenRec.recorder = null;
  await new Promise((resolve) => {
    if (recorder.state === "inactive") return resolve();
    recorder.addEventListener("stop", resolve, { once: true });
    recorder.stop();
  });
  stream.getTracks().forEach((tr) => tr.stop());
  $("previewVideo").srcObject = null;
  $("cpScreenVideo").srcObject = null;
  cpRenderScreen();
  await screenRec.uploads;  // the last chunk must reach the server before the clip is closed
  await api(`/api/record/screen/${clip.n}/stop`, { method: "POST" }).catch(() => {});
  renderShareButton();
}

$("shareBtn").addEventListener("click", () => (screenRec.recorder ? stopScreen() : startScreen()));

// Playback: the audio is the clock, the screen video follows it.
function setupStage(m) {
  const clips = (m.screens || []).filter((c) => c.duration);
  $("stage").classList.toggle("hidden", !clips.length);
  const v = $("screenVideo");
  v.pause();
  v.removeAttribute("src");
  delete v.dataset.key;
  syncScreen();
}

function syncScreen() {
  const m = state.current;
  if (!m || !(m.screens || []).length) return;
  const a = $("audio"), v = $("screenVideo"), now = a.currentTime || 0;
  if (playingId() !== m.id) { if (!v.paused) v.pause(); return; }
  if (v.playbackRate !== a.playbackRate) v.playbackRate = a.playbackRate;
  const clip = m.screens.find((c) => c.duration && now >= c.offset && now < c.offset + c.duration);
  $("stageEmpty").classList.toggle("hidden", !!clip);
  v.classList.toggle("hidden", !clip);
  if (!clip) { if (!v.paused) v.pause(); return; }
  const key = `${m.id}/${clip.n}`;
  const target = now - clip.offset;
  if (v.dataset.key !== key) {
    v.dataset.key = key;
    v.src = `/api/meetings/${m.id}/screen/${clip.n}`;
    v.addEventListener("loadedmetadata", () => { v.currentTime = Math.max(0, (a.currentTime || 0) - clip.offset); }, { once: true });
  } else if (v.readyState >= 1 && Math.abs(v.currentTime - target) > 0.35) {
    v.currentTime = target;
  }
  if (!a.paused && v.paused) v.play().catch(() => {});
  if (a.paused && !v.paused) v.pause();
}

for (const ev of ["play", "pause", "seeked", "timeupdate"]) $("audio").addEventListener(ev, syncScreen);

// ------------------------------------------------------------------ mute while recording

state.muted = { mic: false, system: false };

function renderMute() {
  const spec = [["muteMic", "mic", "ph-microphone", "ph-microphone-slash", "live.mic", "live.muteMic", "live.unmuteMic", "micMeter"],
                ["muteSys", "system", "ph-speaker-high", "ph-speaker-slash", "live.system", "live.muteSys", "live.unmuteSys", "sysMeter"]];
  for (const [id, ch, on, off, label, muteKey, unmuteKey, meter] of spec) {
    const m = state.muted[ch];
    const b = $(id);
    b.setAttribute("aria-pressed", String(m));
    b.title = t(m ? unmuteKey : muteKey);
    b.innerHTML = `<i class="ph ${m ? off : on}"></i><span>${esc(t(label))}${m ? ` (${esc(t("live.muted"))})` : ""}</span>`;
    $(meter).classList.toggle("muted", m);
  }
}

for (const id of ["muteMic", "muteSys"]) {
  $(id).addEventListener("click", async () => {
    const ch = $(id).dataset.channel;
    try {
      const r = await api("/api/record/mute", { method: "POST", body: { channel: ch, muted: !state.muted[ch] } });
      state.muted = r.muted;
      renderMute();
    } catch (err) { toast(err.message, true); }
  });
}

// ------------------------------------------------------------------ recording

// Interim captions: what is being said right now (fast, may change), per channel.
state.interim = {};

function renderLive() {
  const m = state.current;
  if (m.mode === "copilot") return cpRenderLive(m);
  const feed = $("liveFeed");
  const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 120;
  const turns = turnsOf(sortedSegs(m), 4);   // live: a pause of more than 4 s starts a new line
  const interim = Object.entries(state.interim)
    .map(([ch, it]) => `<article class="turn interim" style="--spk:var(--text-3)"><div>
        <div class="turn-head"><span class="who">${esc(it.label || "")}</span><span class="range mono">${clock(it.start)}</span></div>
        <p class="text">${esc(it.committed || "")} <span class="tentative">${esc(it.tentative ?? it.text)}</span></p></div></article>`).join("");
  feed.innerHTML = (turns.map((turn) => turnHtml(m, turn, { live: true })).join("") + interim) || `<p class="waiting">${esc(t("live.waiting"))}</p>`;
  if (nearBottom) feed.scrollTop = feed.scrollHeight;
}

async function startRecording() {
  const btn = $("recordBtn");
  btn.disabled = true;
  try {
    const m = await api("/api/record/start", { method: "POST", body: {} });
    state.recordingId = m.id;
    state.muted = { mic: false, system: false };
    renderMute();
    renderRecordButton();
    await loadMeetings();
    goMeeting(m.id);
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
}

async function stopRecording() {
  $("stopBtn").disabled = true;
  state.interim = {};
  try {
    await stopScreen();
    const m = await api("/api/record/stop", { method: "POST" });
    state.recordingId = null;
    renderRecordButton();
    await loadMeetings();
    state.current = null;
    goMeeting(m.id);
  } catch (err) { toast(err.message, true); }
  $("stopBtn").disabled = false;
}

function renderRecordButton() {
  cpRenderNewButton();
  $("recordBtn").innerHTML = state.recordingId
    ? `<span class="rec-dot"></span><span>${esc(t("nav.recordingNow"))}</span>`
    : `<i class="ph ph-record"></i><span>${esc(t("nav.newRecording"))}</span>`;
}

$("recordBtn").addEventListener("click", () => (state.recordingId ? goMeeting(state.recordingId) : startRecording()));
$("stopBtn").addEventListener("click", stopRecording);

// ------------------------------------------------------------------ import

async function importFile(file) {
  const btn = $("importBtn");
  btn.disabled = true;
  toast(t("toast.uploading", { name: file.name }));
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/import", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const m = await res.json();
    await loadMeetings();
    state.current = null;
    goMeeting(m.id);
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
}

const pickFile = () => $("importFile").click();
$("importBtn").addEventListener("click", pickFile);
$("importFile").addEventListener("change", (e) => {
  if (e.target.files[0]) importFile(e.target.files[0]);
  e.target.value = "";
});
const sidebar = document.querySelector(".sidebar");
sidebar.addEventListener("dragover", (e) => { e.preventDefault(); sidebar.classList.add("drag"); });
sidebar.addEventListener("dragleave", () => sidebar.classList.remove("drag"));
sidebar.addEventListener("drop", (e) => {
  e.preventDefault();
  sidebar.classList.remove("drag");
  if (e.dataTransfer.files[0]) importFile(e.dataTransfer.files[0]);
});

// Buttons rendered inside panels and the home view
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-action]");
  if (!b) return;
  if (b.dataset.action === "record") startRecording();
  if (b.dataset.action === "new-session") cpOpenWizard();
  if (b.dataset.action === "import") pickFile();
  if (b.dataset.action === "summary") startSummary();
  if (b.dataset.action === "summary-force") startSummary({ language: state.current.summary_lang || null, force: true });
  if (b.dataset.action === "copy-summary") copyText(state.current.summary || "");
});

// ------------------------------------------------------------------ settings page

function renderArchive() {
  const items = state.meetings.filter((m) => m.archived);
  $("archiveList").innerHTML = items.map((m) => {
    const confirm = state.confirmDelete === m.id;
    return `<div class="archive-row">
      <div class="info"><div class="title">${esc(m.title)}</div>
        <div class="sub">${esc(confirm ? t("list.deleteConfirm") : `${fmtDate(m.created_at)}, ${minutes(m.duration)}`)}</div></div>
      <a class="btn btn-ghost btn-sm" href="#/meeting/${encodeURIComponent(m.id)}">${esc(t("settings.archiveOpen"))}</a>
      <button type="button" class="icon-btn" data-archive="${m.id}" title="${esc(t("list.unarchive"))}" aria-label="${esc(t("list.unarchive"))}"><i class="ph ph-arrow-u-up-left"></i></button>
      <button type="button" class="icon-btn${confirm ? " confirm" : ""}" data-delete="${m.id}" title="${esc(t("list.delete"))}" aria-label="${esc(t("list.delete"))}"><i class="ph ${confirm ? "ph-check" : "ph-trash"}"></i></button>
    </div>`;
  }).join("") || `<p class="help">${esc(t("settings.archiveEmpty"))}</p>`;
}

// Same archive / delete behaviour as the sidebar, from the Settings page.
$("archiveList").addEventListener("click", async (e) => {
  const arch = e.target.closest("[data-archive]"), del = e.target.closest("[data-delete]");
  if (!arch && !del) return;
  e.preventDefault();
  if (arch) {
    try { await api(`/api/meetings/${arch.dataset.archive}`, { method: "PATCH", body: { archived: false } }); toast(t("list.unarchived")); }
    catch (err) { toast(err.message, true); }
  } else if (state.confirmDelete !== del.dataset.delete) {
    state.confirmDelete = del.dataset.delete;
    return renderArchive();
  } else {
    state.confirmDelete = null;
    try { await api(`/api/meetings/${del.dataset.delete}`, { method: "DELETE" }); toast(t("toast.deleted")); }
    catch (err) { toast(err.message, true); }
  }
  await loadMeetings();
  renderArchive();
});

function renderArchiveCount() {
  const n = state.meetings.filter((m) => m.archived).length;
  $("archiveCount").textContent = !n ? t("settings.archiveNone") : n === 1 ? t("settings.archiveCountOne") : t("settings.archiveCount", { n });
}

async function showArchive() {
  state.current = null;
  await loadMeetings();
  showView("archiveView");
  renderArchive();
}

async function showSettings() {
  state.current = null;
  renderList();
  showView("settingsView");
  saveState("");
  renderArchiveCount();
  try {
    const [s, d] = await Promise.all([api("/api/settings"), api("/api/devices")]);
    const opts = (list, sel) => `<option value="">${esc(t("settings.systemDefault"))}</option>` +
      list.map((x) => `<option value="${esc(x.id)}" ${x.id === sel ? "selected" : ""}>${esc(x.name)}</option>`).join("");
    $("setMic").innerHTML = opts(d.microphones, s.mic_device);
    $("setSpeaker").innerHTML = opts(d.speakers, s.speaker_device);
    $("setEngine").value = s.engine;
    $("setModel").value = s.whisper_model;
    $("setDevice").value = s.device;
    $("setLang").value = s.language_mode;
    $("setSummaryLang").value = s.summary_language;
    $("setAuto").checked = s.auto_summary;
    $("setRefine").checked = s.refine_after_recording !== false;
    $("setVocabulary").value = s.vocabulary || "";
    settingsCache = s;
    $("setKey").value = s.anthropic_api_key || "";
    $("setWorkspace").value = s.anthropic_workspace_id || "";
    $("setOpenaiKey").value = s.openai_api_key || "";
    $("setNvidiaKey").value = s.nvidia_api_key || "";
    renderProviders(s);
    renderModelList();
    $("setUiLang").innerHTML = Object.entries(LANGUAGES)
      .map(([code, name]) => `<option value="${code}" ${code === (s.ui_language || "en") ? "selected" : ""}>${esc(name)}</option>`).join("");
  } catch (err) { toast(err.message, true); }
}

// Settings save themselves: every change is sent at once, no Save button.
const SETTING_FIELDS = {
  setMic: "mic_device", setSpeaker: "speaker_device", setEngine: "engine", setModel: "whisper_model",
  setDevice: "device", setLang: "language_mode", setSummaryLang: "summary_language",
  setAuto: "auto_summary", setUiLang: "ui_language", setProvider: "summary_provider",
  setRefine: "refine_after_recording", setVocabulary: "vocabulary",
  setKey: "anthropic_api_key", setWorkspace: "anthropic_workspace_id",
  setOpenaiKey: "openai_api_key", setNvidiaKey: "nvidia_api_key",
};
const KEY_FIELDS = ["anthropic_api_key", "anthropic_workspace_id", "openai_api_key", "nvidia_api_key"];
const MODEL_PROVIDERS = ["openai", "nvidia"];  // Claude uses a fixed model

function saveState(kind) {
  const el = $("saveState");
  el.className = "save-state" + (kind === "ok" ? " ok" : "");
  el.innerHTML = kind === "saving" ? esc(t("settings.saving"))
    : kind === "ok" ? `<i class="ph ph-check"></i>${esc(t("settings.saved"))}` : esc(t("settings.autosave"));
}

// Only providers that have a key can be chosen; with no keys at all, say so.
function renderProviders(s) {
  const available = s.available_providers || [];
  $("noKeys").classList.toggle("hidden", available.length > 0);
  const sel = $("setProvider");
  sel.disabled = !available.length;
  for (const opt of sel.options) {
    const has = available.includes(opt.value);
    opt.disabled = !has;
    opt.textContent = t(`settings.provider${opt.value[0].toUpperCase()}${opt.value.slice(1)}`) + (has ? "" : t("settings.noKeySuffix"));
  }
  let provider = s.summary_provider;
  if (available.length && !available.includes(provider)) {
    provider = available[0];
    saveSetting("summary_provider", provider);
  }
  sel.value = available.length ? provider : "";
  renderModelField(s, available.length ? provider : null);
}

// Model list for OpenAI-compatible providers comes from the user's own account.
async function renderModelField(s, provider) {
  const field = $("modelField"), sel = $("setModelPick"), help = $("modelHelp");
  field.classList.toggle("hidden", !MODEL_PROVIDERS.includes(provider));
  if (!MODEL_PROVIDERS.includes(provider)) return;
  const selected = s[`${provider}_model`] || "";
  sel.dataset.provider = provider;
  sel.disabled = true;
  sel.innerHTML = selected ? `<option>${esc(selected)}</option>` : "";
  help.textContent = t("settings.modelLoading");
  try {
    const models = await api(`/api/models/${provider}`);
    if (sel.dataset.provider !== provider) return;  // the user switched provider meanwhile
    const list = selected && !models.includes(selected) ? [selected, ...models] : models;
    sel.innerHTML = (selected ? "" : `<option value="">${esc(t("settings.modelPick"))}</option>`) +
      list.map((id) => `<option value="${esc(id)}" ${id === selected ? "selected" : ""}>${esc(id)}</option>`).join("");
    sel.disabled = false;
    help.textContent = "";
  } catch (err) {
    help.textContent = t("settings.modelFailed", { message: err.message });
  }
}

let settingsCache = {};

// ---- recognition models: status + download (with live progress over the websocket) ----
const GB = 1024 * 1024 * 1024;
function gb(bytes) { return (bytes / GB).toFixed(1).replace(/\.0$/, "") + " GB"; }

function modelRow(m) {
  let status, action = "";
  if (m.state === "downloading") {
    status = t("settings.modelDownloading", { percent: m.percent });
    action = `<div class="model-bar"><span style="width:${m.percent}%"></span></div>`;
  } else if (m.present || m.state === "done") {
    status = `<i class="ph ph-check-circle"></i> ${t("settings.modelInstalled")} · ${gb(m.size)}`;
  } else if (m.state === "error") {
    status = `<span class="err">${esc(t("settings.modelDownloadFailed", { message: m.message }))}</span>`;
    action = `<button type="button" class="btn btn-ghost" data-download="${esc(m.key)}"><i class="ph ph-arrow-clockwise"></i>${esc(t("settings.modelRetry"))}</button>`;
  } else {
    status = `${t("settings.modelNotInstalled")} · ~${gb(m.expected)}`;
    action = `<button type="button" class="btn btn-ghost" data-download="${esc(m.key)}"><i class="ph ph-download-simple"></i>${esc(t("settings.modelDownload"))}</button>`;
  }
  return `<div class="model-row" data-key="${esc(m.key)}">
    <div class="model-info"><span class="model-name">${esc(m.label)}</span><span class="model-status">${status}</span></div>
    <div class="model-action">${action}</div>
  </div>`;
}

function paintModels() {
  const box = $("modelList");
  if (box) box.innerHTML = (state.models || []).map(modelRow).join("");
}

async function renderModelList() {
  try {
    state.models = await api("/api/recognition-models");
    paintModels();
  } catch (err) {
    $("modelList").innerHTML = `<p class="help err">${esc(err.message)}</p>`;
  }
}

$("modelList").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-download]");
  if (!btn) return;
  const key = btn.dataset.download;
  btn.disabled = true;
  const m = (state.models || []).find((x) => x.key === key);
  if (m) { m.state = "downloading"; m.percent = 0; paintModels(); }
  try {
    await api(`/api/recognition-models/${key}/download`, { method: "POST" });
  } catch (err) {
    toast(err.message, true);
    renderModelList();
  }
});

async function saveSetting(field, value) {
  saveState("saving");
  try {
    const s = await api("/api/settings", { method: "POST", body: { [field]: value } });
    settingsCache = s;
    if (field === "ui_language" && value !== LANG_CODE) {
      await loadLanguage(value);
      applyI18n();
      renderRecordButton();
      renderShareButton();
      renderMute();
      renderSpeakerCountOptions();
      renderEngine();
      renderList();
    }
    if (KEY_FIELDS.includes(field) || field === "summary_provider" || field === "ui_language") renderProviders(s);
    saveState("ok");
  } catch (err) {
    saveState("");
    toast(err.message, true);
    showSettings();  // put the field back to the stored value
  }
}

$("settingsForm").addEventListener("submit", (e) => e.preventDefault());
$("testModelBtn").addEventListener("click", async () => {
  const sel = $("setModelPick"), provider = sel.dataset.provider, model = sel.value;
  if (!provider || !model) return;
  const btn = $("testModelBtn"), help = $("modelHelp");
  btn.disabled = true;
  help.textContent = t("settings.testing", { model });
  try {
    const r = await api(`/api/models/${provider}/test`, { method: "POST", body: { model } });
    help.textContent = t("settings.testOk", { model, s: r.seconds });
  } catch (err) {
    help.textContent = t("settings.testFail", { model, message: err.message });
  }
  btn.disabled = false;
});
$("settingsForm").addEventListener("change", (e) => {
  if (e.target.id === "setModelPick") {
    const provider = e.target.dataset.provider;
    if (provider) saveSetting(`${provider}_model`, e.target.value);
    return;
  }
  const field = SETTING_FIELDS[e.target.id];
  if (!field) return;
  if (KEY_FIELDS.includes(field)) return saveSetting(field, e.target.value.trim());
  saveSetting(field, e.target.type === "checkbox" ? e.target.checked : e.target.value);
});

// ------------------------------------------------------------------ websocket

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1500);
}

function renderEngine() {
  const e = state.engine;
  const el = $("engineStatus");
  const device = e.device && e.device.startsWith("cuda") ? "GPU" : "CPU";
  el.textContent = {
    loading: t("engine.loading"),
    ready: t("engine.ready", { device }) + (e.queue ? t("engine.queue", { n: e.queue }) : ""),
    error: t("engine.error", { error: e.error }),
    idle: t("engine.idle"),
  }[e.state] || "";
  el.className = "engine" + (e.state === "error" ? " err" : "");
}

function handle(msg) {
  const cur = state.current;
  const isCur = cur && cur.id === msg.meeting_id;
  switch (msg.type) {
    case "engine":
      state.engine = msg;
      renderEngine();
      break;
    case "model_download": {
      const m = (state.models || []).find((x) => x.key === msg.key);
      if (m) {
        m.state = msg.state; m.percent = msg.percent; m.message = msg.message || "";
        if (msg.state === "done") { m.present = true; renderModelList(); }  // refresh size on disk
        else paintModels();
      }
      break;
    }
    case "muted":
      state.muted = msg.muted;
      renderMute();
      break;
    case "levels":
      if (msg.muted && (msg.muted.mic !== state.muted.mic || msg.muted.system !== state.muted.system)) { state.muted = msg.muted; renderMute(); }
      $("liveTimer").textContent = clock(msg.elapsed);
      $("micLevel").style.width = `${msg.mic * 100}%`;
      $("sysLevel").style.width = `${msg.system * 100}%`;
      break;
    case "interim":
      if (isCur && cur.status === "recording") {
        state.interim[msg.channel] = { start: msg.start, text: msg.text, committed: msg.committed, tentative: msg.tentative, label: msg.label };
        renderLive();
      }
      break;
    case "speakers":
      if (isCur) {
        cur.speakers = msg.speakers;
        if (cur.status === "recording") renderLive(); else renderMeeting();
      }
      break;
    case "relabel":
      if (isCur) {
        for (const seg of cur.segments) if (msg.assign[seg.id]) seg.speaker = msg.assign[seg.id];
        cur.speakers = msg.speakers;
        if (cur.status === "recording") renderLive(); else renderMeeting();
      }
      break;
    case "segment_removed":
      if (isCur) {
        cur.segments = cur.segments.filter((x) => x.id !== msg.id);
        if (cur.status === "recording") renderLive(); else renderTranscript();
      }
      break;
    case "segment":
      if (isCur) {
        cur.segments.push(msg.segment);
        const it = state.interim[msg.segment.channel];
        if (it && msg.segment.start >= it.start - 0.5) delete state.interim[msg.segment.channel];  // final text replaces it
        if (cur.status === "recording") renderLive();
        else { renderTranscript(); renderProgress(); }
      }
      break;
    case "progress":
      state.progress[msg.meeting_id] = msg.percent;
      if (isCur) renderProgress();
      break;
    case "device_switched":
      toast(t("toast.deviceSwitched", { source: t(msg.channel === "mic" ? "live.mic" : "live.system") }));
      break;
    case "capture_error":
      toast(t("toast.captureFailed", { source: t(msg.channel === "mic" ? "live.mic" : "live.system"), message: msg.message }), true);
      break;
    case "meeting":
      if (msg.step) state.step[msg.meeting_id] = msg.step;
      loadMeetings();
      if (isCur) {
        if (msg.status === "ready") openMeeting(cur.id, { keepTab: true });
        else renderProgress();
      }
      break;
    case "answer":
    case "answer_delta":
    case "answer_title":
    case "answer_removed":
    case "answer_question":
    case "answer_done":
      cpHandle(msg);
      break;
    case "summary":
      if (msg.state === "running") state.summaryRunning.add(msg.meeting_id);
      else state.summaryRunning.delete(msg.meeting_id);
      if (msg.state === "error") toast(t("toast.summaryFailed", { message: msg.message }), true);
      if (msg.state === "done") loadMeetings();
      if (isCur) {
        if (msg.state === "done") openMeeting(cur.id, { keepTab: true });
        else renderOverview();
      }
      break;
  }
}

// ------------------------------------------------------------------ resizable panels
// Left sidebar and the right transcript panel can be dragged wider or narrower. Widths are kept
// per browser (localStorage) and restored on reload. Double-click a handle to reset it.

const PANELS = {
  sidebar: { cssVar: "--sidebar-w", key: "mh.sidebarWidth", min: 200, max: 480, fallback: 272 },
  side: { cssVar: "--side-w", key: "mh.sideWidth", min: 320, maxRatio: 0.7, fallback: null },
};

function panelMax(p) {
  return p.max || Math.round(window.innerWidth * p.maxRatio);
}

function applyPanelWidth(name, px) {
  const p = PANELS[name];
  if (px == null) {
    document.documentElement.style.removeProperty(p.cssVar);
    return;
  }
  const w = Math.round(Math.min(panelMax(p), Math.max(p.min, px)));
  document.documentElement.style.setProperty(p.cssVar, `${w}px`);
  return w;
}

function savePanelWidth(name, px) {
  try {
    if (px == null) localStorage.removeItem(PANELS[name].key);
    else localStorage.setItem(PANELS[name].key, String(px));
  } catch {}
}

function restorePanelWidths() {
  for (const name of Object.keys(PANELS)) {
    let v = null;
    try { v = localStorage.getItem(PANELS[name].key); } catch {}
    if (v && !isNaN(+v)) applyPanelWidth(name, +v);
  }
}

document.querySelectorAll("[data-resize]").forEach((handle) => {
  const name = handle.dataset.resize;
  const startDrag = (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    document.body.classList.add("resizing");
    const move = (ev) => {
      // sidebar grows to the right; the transcript panel grows to the left
      const px = name === "sidebar" ? ev.clientX - document.querySelector(".sidebar").getBoundingClientRect().left
                                    : document.querySelector("#meetingView").getBoundingClientRect().right - ev.clientX;
      handle.dataset.width = applyPanelWidth(name, px);
    };
    const up = () => {
      handle.classList.remove("dragging");
      document.body.classList.remove("resizing");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      if (handle.dataset.width) savePanelWidth(name, +handle.dataset.width);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  };
  handle.addEventListener("pointerdown", startDrag);
  handle.addEventListener("dblclick", () => { applyPanelWidth(name, null); savePanelWidth(name, null); });
  // keyboard: arrow keys resize in 16 px steps
  handle.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const el = name === "sidebar" ? document.querySelector(".sidebar") : document.querySelector(".meeting-side");
    const grow = (e.key === "ArrowRight") === (name === "sidebar") ? 16 : -16;
    const w = applyPanelWidth(name, el.getBoundingClientRect().width + grow);
    savePanelWidth(name, w);
  });
});
window.addEventListener("resize", restorePanelWidths);
restorePanelWidths();

// ------------------------------------------------------------------ boot

(async function boot() {
  const settings = await api("/api/settings").catch(() => ({}));
  await loadLanguage(settings.ui_language || "en");
  applyI18n();
  renderRecordButton();
  renderSpeakerCountOptions();
  renderEngine();
  connect();
  const st = await api("/api/status");
  state.recordingId = st.recording ? st.meeting_id : null;
  state.muted = st.muted || { mic: false, system: false };
  renderMute();
  renderRecordButton();
  await loadMeetings();
  let savedMode = "notes";
  try { savedMode = localStorage.getItem("mh.mode") || "notes"; } catch {}
  setMode(savedMode);
  if (!location.hash || location.hash === "#") {
    const notes = state.meetings.filter((m) => m.mode !== "copilot");
    const first = st.recording ? st.meeting_id : savedMode === "notes" && notes[0] && notes[0].id;
    if (first) history.replaceState(null, "", `#/meeting/${first}`);
    else if (savedMode === "copilot") history.replaceState(null, "", "#/copilot");
  }
  route();
})();
