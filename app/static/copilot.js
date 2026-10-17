// Murmur Copilot: interview mode. Sessions, resumes, documents, the create-session wizard and the live
// answer panel. Uses the helpers from app.js ($, t, api, esc, state, toast, ...).

const CP_LIB = {
  resumes: { title: "cp.resumes", sub: "cp.resumesSub", add: "cp.addResume", addSub: "cp.addResumeSub", fresh: "cp.newResume", url: false },
  documents: { title: "cp.documents", sub: "cp.documentsSub", add: "cp.addDocument", addSub: "cp.addDocumentSub", fresh: "cp.newDocument", url: true },
};
const PROVIDER_NAMES = { claude: "Claude", openai: "OpenAI", nvidia: "NVIDIA" };

state.mode = "notes";
state.cp = { filter: "all", query: "", libKind: "resumes", libQuery: "", items: [], confirm: null, editing: null };

// ------------------------------------------------------------------ mode switch (top bar)

function setMode(mode) {
  state.mode = mode;
  const cp = mode === "copilot";
  document.body.classList.toggle("mode-copilot", cp);
  $("brandName").textContent = cp ? "Murmur Copilot" : "Murmur Notetaker";
  document.querySelector(".wordmark").setAttribute("href", cp ? "#/copilot" : "#/");
  document.querySelector(".wordmark").setAttribute("aria-label", cp ? "Murmur Copilot" : "Murmur Notetaker");
  document.title = cp ? "Murmur Copilot" : "Murmur Notetaker";
  document.querySelectorAll(".mode-switch [data-mode]").forEach((b) => {
    const on = b.dataset.mode === mode;
    b.setAttribute("aria-selected", String(on));
    b.title = t(b.dataset.mode === "copilot" ? "mode.toCopilotHint" : "mode.toNotesHint");
  });
  $("modeHint").textContent = t(cp ? "mode.copilotHint" : "mode.notesHint");
  try { localStorage.setItem("mh.mode", mode); } catch {}
}

document.querySelector(".mode-switch").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (!b || b.dataset.mode === state.mode) return;
  location.hash = b.dataset.mode === "copilot" ? "#/copilot" : "#/";
});

function cpRenderNewButton() {
  const running = state.recordingId && state.meetings.some((m) => m.id === state.recordingId && m.mode === "copilot");
  $("newSessionBtn").innerHTML = running
    ? `<span class="rec-dot"></span><span>${esc(t("cp.sessionRunning"))}</span>`
    : `<i class="ph ph-plus"></i><span>${esc(t("cp.newSession"))}</span>`;
}

$("newSessionBtn").addEventListener("click", () => (state.recordingId ? goMeeting(state.recordingId) : cpOpenWizard()));

// ------------------------------------------------------------------ routing: #/copilot, #/copilot/resumes, #/copilot/documents

async function cpRoute(h) {
  setMode("copilot");
  state.current = null;
  const kind = h === "#/copilot/resumes" ? "resumes" : h === "#/copilot/documents" ? "documents" : null;
  document.querySelectorAll("[data-cp-nav]").forEach((a) => a.classList.toggle("active", a.dataset.cpNav === (kind || "sessions")));
  renderList();
  if (kind) return cpShowLib(kind);
  showView("cpSessionsView");
  cpRenderSessions();
}

const cpSessions = () => state.meetings.filter((m) => m.mode === "copilot");

function cpRenderRecent() {
  const items = cpSessions().filter((m) => !m.archived).slice(0, 30);
  $("cpRecent").innerHTML = items.length
    ? `<div class="group-label">${esc(t("cp.recent"))}</div>` + items.map((m) => {
        const active = state.current && state.current.id === m.id ? " active" : "";
        const dot = m.status === "recording" ? `<span class="rec-dot"></span>` : "";
        const sub = m.status === "processing" ? t("list.processing") : `${fmtDate(m.created_at)}, ${minutes(m.duration)}`;
        return `<a class="item${active}" href="#/meeting/${encodeURIComponent(m.id)}"><div class="item-title">${dot}<span>${esc(m.title)}</span></div><div class="item-sub">${esc(sub)}</div></a>`;
      }).join("")
    : "";
  document.querySelectorAll("[data-cp-nav]").forEach((a) => {
    if (state.current && state.current.mode === "copilot") a.classList.remove("active");
  });
}

// ------------------------------------------------------------------ sessions page

function cpDate(iso) {
  const d = new Date(iso);
  return `${t("months")[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()} · ${hhmm(d)}`;
}

function cpInitial(name) {
  return esc((name || "?").trim().charAt(0).toUpperCase() || "?");
}

function cpRenderSessions() {
  const q = state.cp.query.toLowerCase();
  const all = cpSessions();
  const list = all.filter((m) => {
    const c = m.copilot || {};
    if (state.cp.filter !== "all" && (c.type || "interview") !== state.cp.filter) return false;
    return !q || `${c.company} ${c.title} ${m.title}`.toLowerCase().includes(q);
  });
  document.querySelectorAll("#cpFilter [data-filter]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.filter === state.cp.filter)));
  $("cpCount").textContent = t("cp.countSessions", { n: all.length });
  $("cpSessions").innerHTML = list.map((m) => {
    const c = m.copilot || {};
    const company = c.company || m.title;
    const type = c.type === "regular" ? ["ph-phone", "cp.typeRegular"] : ["ph-briefcase", "cp.typeInterview"];
    const status = m.status === "recording" ? `<span class="live-pill"><span class="rec-dot"></span>${esc(t("list.recording"))}</span>`
      : m.status === "processing" ? `<span class="muted">${esc(t("list.processing"))}</span>` : `<span class="muted">${esc(minutes(m.duration))}</span>`;
    const confirm = state.cp.confirm === m.id;
    return `<a class="cp-card" href="#/meeting/${encodeURIComponent(m.id)}">
      <div class="cp-card-top"><span class="cp-date">${esc(cpDate(m.created_at))}</span>
        ${m.status === "ready" ? `<button class="icon-btn icon-sm${confirm ? " confirm" : ""}" data-cp-del="${m.id}" title="${esc(t(confirm ? "cp.deleteConfirm" : "cp.delete"))}"><i class="ph ${confirm ? "ph-check" : "ph-trash"}"></i></button>` : ""}</div>
      <div class="cp-company"><span class="avatar">${cpInitial(company)}</span><span class="name">${esc(company)}</span></div>
      <div class="cp-role">${esc(c.title || "")}</div>
      <div class="cp-chips"><span class="chip-sm"><i class="ph ${type[0]}"></i>${esc(t(type[1]))}</span>
        ${m.answers ? `<span class="chip-sm"><i class="ph ph-sparkle"></i>${esc(t("cp.answersCount", { n: m.answers }))}</span>` : ""}</div>
      <div class="cp-card-foot">${status}<span class="btn btn-ghost btn-sm">${esc(t("cp.open"))}</span></div>
    </a>`;
  }).join("") || `<p class="cp-empty">${esc(t(all.length ? "cp.noMatch" : "cp.noSessions"))}</p>`;
}

$("cpFilter").addEventListener("click", (e) => {
  const b = e.target.closest("[data-filter]");
  if (b) { state.cp.filter = b.dataset.filter; cpRenderSessions(); }
});
$("cpSearch").addEventListener("input", (e) => { state.cp.query = e.target.value; cpRenderSessions(); });
$("cpSessions").addEventListener("click", async (e) => {
  const del = e.target.closest("[data-cp-del]");
  if (!del) return;
  e.preventDefault();
  const id = del.dataset.cpDel;
  if (state.cp.confirm !== id) {
    state.cp.confirm = id;
    clearTimeout(state.cp.confirmTimer);
    state.cp.confirmTimer = setTimeout(() => { state.cp.confirm = null; cpRenderSessions(); }, 4000);
    return cpRenderSessions();
  }
  state.cp.confirm = null;
  try { await api(`/api/meetings/${id}`, { method: "DELETE" }); toast(t("cp.deleted")); await loadMeetings(); cpRenderSessions(); }
  catch (err) { toast(err.message, true); }
});

// ------------------------------------------------------------------ resumes / documents

async function cpShowLib(kind) {
  state.cp.libKind = kind;
  state.cp.libQuery = "";
  $("cpLibSearch").value = "";
  const spec = CP_LIB[kind];
  $("cpLibTitle").textContent = t(spec.title);
  $("cpLibSub").textContent = t(spec.sub);
  $("cpAddLabel").textContent = t(spec.add);
  showView("cpLibView");
  $("cpLib").innerHTML = `<div class="skeleton">${"<i></i>".repeat(4)}</div>`;
  await cpLoadLib();
}

async function cpLoadLib() {
  try { state.cp.items = await api(`/api/copilot/${state.cp.libKind}`); }
  catch (err) { state.cp.items = []; toast(err.message, true); }
  cpRenderLib();
}

function cpRenderLib() {
  const q = state.cp.libQuery.toLowerCase();
  const items = state.cp.items.filter((it) => !q || `${it.name} ${it.preview}`.toLowerCase().includes(q));
  $("cpLibCount").textContent = t("cp.countItems", { n: state.cp.items.length });
  const src = { uploaded: ["ph-upload-simple", "cp.srcUploaded"], manual: ["ph-pencil-simple", "cp.srcManual"], scraped: ["ph-globe", "cp.srcScraped"] };
  $("cpLib").innerHTML = items.map((it) => {
    const s = src[it.source] || src.manual;
    const confirm = state.cp.confirm === it.id;
    return `<article class="lib-card" data-id="${it.id}">
      <button class="lib-thumb" data-lib-edit="${it.id}" title="${esc(t("cp.edit"))}">
        <div class="paper"><b>${esc(it.name)}</b><p>${esc(it.preview)}</p></div>
        <span class="src-pill"><i class="ph ${s[0]}"></i>${esc(t(s[1]))}</span>
      </button>
      <div class="lib-info">
        <div><div class="lib-name">${esc(it.name)}</div>
          <div class="lib-sub">${esc(fmtDate(it.created_at))} · ${esc(t("cp.chars", { n: it.chars.toLocaleString() }))}</div></div>
        <button class="icon-btn icon-sm" data-lib-edit="${it.id}" title="${esc(t("cp.edit"))}"><i class="ph ph-pencil-simple"></i></button>
        <button class="icon-btn icon-sm${confirm ? " confirm" : ""}" data-lib-del="${it.id}" title="${esc(t(confirm ? "cp.deleteConfirm" : "cp.delete"))}"><i class="ph ${confirm ? "ph-check" : "ph-trash"}"></i></button>
      </div>
    </article>`;
  }).join("") || `<p class="cp-empty">${esc(t(q ? "cp.noMatch" : "cp.noItems"))}</p>`;
}

$("cpLibSearch").addEventListener("input", (e) => { state.cp.libQuery = e.target.value; cpRenderLib(); });

$("cpLib").addEventListener("click", async (e) => {
  const edit = e.target.closest("[data-lib-edit]");
  const del = e.target.closest("[data-lib-del]");
  if (edit) return cpEdit(edit.dataset.libEdit);
  if (!del) return;
  const id = del.dataset.libDel;
  if (state.cp.confirm !== id) {
    state.cp.confirm = id;
    clearTimeout(state.cp.confirmTimer);
    state.cp.confirmTimer = setTimeout(() => { state.cp.confirm = null; cpRenderLib(); }, 4000);
    return cpRenderLib();
  }
  state.cp.confirm = null;
  try { await api(`/api/copilot/${state.cp.libKind}/${id}`, { method: "DELETE" }); toast(t("cp.deleted")); await cpLoadLib(); }
  catch (err) { toast(err.message, true); }
});

// dialogs close on the x / Cancel buttons and on a click outside
document.querySelectorAll("dialog.modal").forEach((d) => {
  d.addEventListener("click", (e) => {
    if (e.target.closest("[data-close]") || e.target === d) d.close();
  });
});

$("cpAddBtn").addEventListener("click", () => {
  const spec = CP_LIB[state.cp.libKind];
  $("addTitle").textContent = t(spec.add);
  $("addSub").textContent = t(spec.addSub);
  $("addUrlBtn").classList.toggle("hidden", !spec.url);
  $("addDialog").showModal();
});

$("addDialog").addEventListener("click", (e) => {
  const b = e.target.closest("[data-add]");
  if (!b) return;
  $("addDialog").close();
  if (b.dataset.add === "upload") $("cpLibFile").click();
  if (b.dataset.add === "manual") cpEdit(null);
  if (b.dataset.add === "url") cpEdit(null, { url: true });
});

$("cpLibFile").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  e.target.value = "";
  for (const file of files) {
    toast(t("cp.reading", { name: file.name }));
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`/api/copilot/${state.cp.libKind}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const it = await res.json();
      toast(t("cp.added", { name: it.name }));
    } catch (err) { toast(`${file.name}: ${err.message}`, true); }
  }
  await cpLoadLib();
});

async function cpEdit(id, { url = false } = {}) {
  const kind = state.cp.libKind;
  state.cp.editing = { id, url };
  $("editUrlField").classList.toggle("hidden", !url);
  $("editTextField").classList.toggle("hidden", url);
  $("editTitle").textContent = url ? t("cp.fromUrl") : id ? t("cp.edit") : t(CP_LIB[kind].fresh);
  $("editUrl").value = "";
  $("editName").value = "";
  $("editText").value = "";
  if (id) {
    try {
      const it = await api(`/api/copilot/${kind}/${id}`);
      $("editName").value = it.name;
      $("editText").value = it.text;
    } catch (err) { return toast(err.message, true); }
  }
  $("editSave").textContent = t("cp.save");
  $("editDialog").showModal();
  (url ? $("editUrl") : id ? $("editText") : $("editName")).focus();
}

$("editForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const kind = state.cp.libKind, ed = state.cp.editing;
  const btn = $("editSave");
  btn.disabled = true;
  try {
    if (ed.url) await api(`/api/copilot/${kind}/url`, { method: "POST", body: { url: $("editUrl").value, name: $("editName").value } });
    else if (ed.id) await api(`/api/copilot/${kind}/${ed.id}`, { method: "PATCH", body: { name: $("editName").value, text: $("editText").value } });
    else await api(`/api/copilot/${kind}`, { method: "POST", body: { name: $("editName").value, text: $("editText").value } });
    $("editDialog").close();
    toast(t("cp.saved"));
    await cpLoadLib();
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
});

// ------------------------------------------------------------------ create-session wizard

const wiz = { step: 1, type: "interview", prefs: {}, docs: "all", defaults: null, docList: [] };

function segPick(id, value) {
  document.querySelectorAll(`#${id} [data-v], #${id} [data-type]`).forEach((b) =>
    b.setAttribute("aria-pressed", String((b.dataset.v || b.dataset.type) === value)));
}

async function cpOpenWizard() {
  if (state.recordingId) return goMeeting(state.recordingId);
  try {
    const [d, resumes, docs] = await Promise.all([api("/api/copilot/defaults"), api("/api/copilot/resumes"), api("/api/copilot/documents")]);
    wiz.defaults = d;
    wiz.step = 1;
    wiz.type = "interview";
    wiz.prefs = { ...d.prefs };
    wiz.docList = docs;
    wiz.docs = d.document_ids === "all" || !Array.isArray(d.document_ids) ? "all" : d.document_ids.filter((x) => docs.some((it) => it.id === x));
    $("sJobUrl").value = "";
    $("sCompany").value = "";
    $("sRole").value = "";
    $("sJd").value = "";
    const resumeId = resumes.some((r) => r.id === d.resume_id) ? d.resume_id : (resumes[0] && resumes[0].id) || "";
    $("sResume").innerHTML = `<option value="">${esc(t("cp.noResume"))}</option>` +
      resumes.map((r) => `<option value="${r.id}" ${r.id === resumeId ? "selected" : ""}>${esc(r.name)}</option>`).join("");
    cpRenderDocs();
    $("sLang").value = d.language || "auto";
    $("sInstructions").value = d.instructions || "";
    $("sStar").checked = !!wiz.prefs.star;
    $("sFillers").checked = !!wiz.prefs.fillers;
    $("sAuto").checked = d.auto_generate !== false;
    segPick("sType", wiz.type);
    segPick("sFormat", wiz.prefs.format);
    segPick("sLength", wiz.prefs.length);
    segPick("sTone", wiz.prefs.tone);
    const prov = $("sProvider");
    prov.innerHTML = d.available_providers.length
      ? d.available_providers.map((p) => `<option value="${p}" ${p === d.provider ? "selected" : ""}>${esc(PROVIDER_NAMES[p] || p)}</option>`).join("")
      : `<option value="">${esc(t("cp.noKey"))}</option>`;
    await cpRenderModels(d.model);
    cpWizardStep(1);
    $("sessionDialog").showModal();
    $("sCompany").focus();
  } catch (err) { toast(err.message, true); }
}

function cpRenderDocs() {
  const all = wiz.docs === "all";
  const picked = all ? wiz.docList.map((d) => d.id) : wiz.docs;
  $("sDocsLabel").textContent = !wiz.docList.length ? t("cp.noDocuments") : all ? t("cp.allDocuments")
    : picked.length ? t("cp.someDocuments", { n: picked.length }) : t("cp.noDocuments");
  $("sDocsList").innerHTML = (wiz.docList.length ? `<label class="check"><input type="checkbox" data-doc="all" ${all ? "checked" : ""}><span>${esc(t("cp.allDocuments"))}</span></label>` : "") +
    wiz.docList.map((d) => `<label class="check"><input type="checkbox" data-doc="${d.id}" ${picked.includes(d.id) ? "checked" : ""}><span>${esc(d.name)}</span></label>`).join("");
}

$("sDocsList").addEventListener("change", (e) => {
  const id = e.target.dataset.doc;
  if (!id) return;
  if (id === "all") wiz.docs = e.target.checked ? "all" : [];
  else {
    const cur = wiz.docs === "all" ? wiz.docList.map((d) => d.id) : [...wiz.docs];
    wiz.docs = e.target.checked ? [...new Set([...cur, id])] : cur.filter((x) => x !== id);
    if (wiz.docs.length === wiz.docList.length) wiz.docs = "all";
  }
  cpRenderDocs();
});

async function cpRenderModels(selected) {
  const provider = $("sProvider").value, sel = $("sModel");
  if (!provider) { sel.innerHTML = ""; sel.disabled = true; return; }
  if (provider === "claude") {
    const list = wiz.defaults.claude_models;
    sel.innerHTML = list.map((m) => `<option ${m === selected ? "selected" : ""}>${esc(m)}</option>`).join("");
    sel.disabled = false;
    return;
  }
  sel.disabled = true;
  sel.innerHTML = selected ? `<option>${esc(selected)}</option>` : "";
  try {
    const models = await api(`/api/models/${provider}`);
    if ($("sProvider").value !== provider) return;
    const list = selected && !models.includes(selected) ? [selected, ...models] : models;
    sel.innerHTML = list.map((m) => `<option ${m === selected ? "selected" : ""}>${esc(m)}</option>`).join("");
  } catch {}
  sel.disabled = false;
}

$("sProvider").addEventListener("change", () => {
  const p = $("sProvider").value;
  cpRenderModels(p === wiz.defaults.provider ? wiz.defaults.model : "");
});

function cpWizardStep(n) {
  wiz.step = n;
  document.querySelectorAll("#sessionForm .step").forEach((s) => s.classList.toggle("hidden", +s.dataset.step !== n));
  document.querySelectorAll(".steps li").forEach((li) => {
    li.classList.toggle("on", +li.dataset.step === n);
    li.classList.toggle("done", +li.dataset.step < n);
  });
  $("sBack").textContent = t(n === 1 ? "cp.cancel" : "cp.back");
  $("sNext").textContent = t(n === 1 ? "cp.next" : "cp.start");
  document.querySelector(".wizard-main").scrollTop = 0;
}

$("sessionForm").addEventListener("click", (e) => {
  const b = e.target.closest(".seg [data-v], .seg [data-type]");
  if (!b) return;
  const group = b.closest(".seg").id;
  if (group === "sType") { wiz.type = b.dataset.type; segPick("sType", wiz.type); }
  else {
    const key = { sFormat: "format", sLength: "length", sTone: "tone" }[group];
    wiz.prefs[key] = b.dataset.v;
    segPick(group, b.dataset.v);
  }
});

$("sResetInstr").addEventListener("click", () => { $("sInstructions").value = wiz.defaults.default_instructions; });

$("sBack").addEventListener("click", () => (wiz.step === 1 ? $("sessionDialog").close() : cpWizardStep(1)));

$("sJobImport").addEventListener("click", async () => {
  const url = $("sJobUrl").value.trim();
  if (!url) return $("sJobUrl").focus();
  const btn = $("sJobImport");
  btn.disabled = true;
  toast(t("cp.importing"));
  try {
    const job = await api("/api/copilot/job-from-url", { method: "POST", body: { url } });
    if (job.company) $("sCompany").value = job.company;
    if (job.title) $("sRole").value = job.title;
    if (job.description) $("sJd").value = job.description;
    toast(t("cp.imported"));
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
});

$("sessionForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (wiz.step === 1) {
    if (!$("sCompany").value.trim() && !$("sRole").value.trim()) { toast(t("cp.needCompany"), true); return $("sCompany").focus(); }
    return cpWizardStep(2);
  }
  const btn = $("sNext");
  btn.disabled = true;
  try {
    const m = await api("/api/copilot/start", { method: "POST", body: {
      type: wiz.type, company: $("sCompany").value.trim(), title: $("sRole").value.trim(), job_description: $("sJd").value,
      resume_id: $("sResume").value, document_ids: wiz.docs, language: $("sLang").value,
      provider: $("sProvider").value, model: $("sModel").value,
      prefs: { ...wiz.prefs, star: $("sStar").checked, fillers: $("sFillers").checked },
      instructions: $("sInstructions").value, auto_generate: $("sAuto").checked,
    } });
    $("sessionDialog").close();
    state.recordingId = m.id;
    state.muted = { mic: false, system: false };
    renderMute();
    renderRecordButton();
    await loadMeetings();
    goMeeting(m.id);
  } catch (err) { toast(err.message, true); }
  btn.disabled = false;
});

// ------------------------------------------------------------------ live session: chat transcript + answers

function answerMd(md) {
  const out = [];
  const parts = String(md || "").split(/```/);
  parts.forEach((part, i) => {
    if (i % 2) {  // code block; the first line may name the language
      const nl = part.indexOf("\n");
      const code = nl >= 0 && /^[\w+#.-]*$/.test(part.slice(0, nl).trim()) ? part.slice(nl + 1) : part;
      out.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
      return;
    }
    let list = null;
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
    const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of part.split(/\r?\n/)) {
      const line = raw.trim();
      let m;
      if (!line) { close(); continue; }
      if ((m = line.match(/^#{1,4}\s+(.*)/))) { close(); out.push(`<h4>${inline(m[1])}</h4>`); }
      else if ((m = line.match(/^[-*•]\s+(.*)/))) { if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(m[1])}</li>`); }
      else if ((m = line.match(/^\d+[.)]\s+(.*)/))) { if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(m[1])}</li>`); }
      else { close(); out.push(`<p>${inline(line)}</p>`); }
    }
    close();
  });
  return out.join("");
}

function cpAnswerHtml(a, { live = false } = {}) {
  const lagged = a.lag != null && a.first ? t("cp.afterQuestion", { s: (a.lag + a.first).toFixed(1) }) : a.first ? t("cp.firstWord", { s: a.first }) : "";
  const meta = [a.auto ? t("cp.auto") : "", lagged, a.seconds ? t("cp.seconds", { s: a.seconds }) : "", live ? clock(a.at) : ""].filter(Boolean).join(" · ");
  const body = a.state === "error" ? `<p class="ap-err">${esc(t("cp.answerFailed", { message: a.error || "" }))}</p>`
    : a.text ? answerMd(a.text) : `<p class="ap-writing">${esc(t("cp.writing"))}</p>`;
  return `<article class="ap-item${a.state === "running" ? " running" : ""}" data-ans="${a.id}">
    <div class="ap-q" title="${esc(a.question || "")}"><i class="ph ${a.image ? "ph-camera" : "ph-chat-circle-text"}"></i><span>${a.title ? `<b class="ap-title-q">${esc(a.title)}</b>` : esc(a.question || (a.image ? t("cp.fromScreenshot") : ""))}</span></div>
    ${a.image && state.current ? `<a class="ap-shot" href="/api/meetings/${state.current.id}/shots/${a.image}" target="_blank" rel="noopener"><img src="/api/meetings/${state.current.id}/shots/${a.image}" alt="${esc(t("cp.screenshot"))}" loading="lazy"></a>` : ""}
    <div class="ap-a">${body}</div>
    <div class="ap-meta"><span>${esc(meta)}</span><button class="icon-btn icon-sm" data-copy-ans="${a.id}" title="${esc(t("cp.copyAnswer"))}"><i class="ph ph-copy"></i></button></div>
  </article>`;
}

function cpRenderLive(m) {
  const feed = $("liveFeed");
  const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 140;
  const groups = [];
  for (const s of sortedSegs(m)) {
    const g = groups[groups.length - 1];
    if (g && g.channel === s.channel && s.start - g.end <= 4) { g.text += " " + s.text; g.end = s.end; }
    else groups.push({ channel: s.channel, start: s.start, end: s.end, text: s.text });
  }
  const bubble = (ch, text, start, interim = "") => {
    const me = ch === "mic";
    return `<div class="bubble ${me ? "me" : "them"}${interim ? " interim" : ""}">
      <p>${text}</p><span class="b-meta">${esc(t(me ? "cp.you" : "cp.interviewer"))} · ${clock(start)}</span></div>`;
  };
  let html = groups.map((g) => bubble(g.channel, esc(g.text), g.start)).join("");
  for (const [ch, it] of Object.entries(state.interim)) {
    html += bubble(ch, `${esc(it.committed || "")} <span class="tentative">${esc(it.tentative ?? it.text)}</span>`, it.start, "1");
  }
  feed.innerHTML = html || `<p class="waiting">${esc(t("live.waiting"))}</p>`;
  if (nearBottom) feed.scrollTop = feed.scrollHeight;
}

function cpRenderAnswers() {
  const m = state.current;
  if (!m || m.mode !== "copilot") return;
  const list = $("apList");
  const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 160;
  const answers = m.answers || [];
  list.innerHTML = answers.map((a) => cpAnswerHtml(a, { live: true })).join("") ||
    `<p class="ap-empty"><i class="ph ph-sparkle"></i>${esc(t("cp.waitingAnswers"))}</p>`;
  if (nearBottom) list.scrollTop = list.scrollHeight;
  const c = m.copilot || {};
  $("apModel").textContent = c.model || PROVIDER_NAMES[c.provider] || "";
  $("apAuto").checked = c.auto_generate !== false;
}

function cpLiveLayout(m) {
  const cp = m.mode === "copilot";
  $("liveView").classList.toggle("copilot", cp);
  $("answerPanel").classList.toggle("hidden", !cp);
  $("cpScreen").classList.toggle("hidden", !cp);
  $("shareBtn").classList.toggle("hidden", cp);
  cpRenderScreen();
  if (cp) { setMode("copilot"); cpRenderAnswers(); }
}

$("apAuto").addEventListener("change", async (e) => {
  try {
    const r = await api("/api/copilot/auto", { method: "POST", body: { auto_generate: e.target.checked } });
    if (state.current && state.current.copilot) state.current.copilot.auto_generate = r.auto_generate;
  } catch (err) { toast(err.message, true); e.target.checked = !e.target.checked; }
});

async function cpAsk(question = "", image = "") {
  try { await api("/api/copilot/answer", { method: "POST", body: { question, image } }); }
  catch (err) { toast(err.message, true); }
}

// ---- screen: shared and recorded like in Notes; screenshots go to the AI with the conversation

function cpRenderScreen() {
  const on = !!screenRec.recorder;
  $("cpShareBtn").innerHTML = on ? `<i class="ph ph-stop-circle"></i><span>${esc(t("live.stopShare"))}</span>`
                                 : `<i class="ph ph-monitor-arrow-up"></i><span>${esc(t("cp.shareScreen"))}</span>`;
  $("cpChangeTab").disabled = !on;
  $("cpFullscreen").disabled = !on;
  $("cpScreenEmpty").classList.toggle("hidden", on);
  $("cpScreenRec").classList.toggle("hidden", !on);
  $("cpScreen").classList.toggle("on", on);
}

$("cpShareBtn").addEventListener("click", () => (screenRec.recorder ? stopScreen() : startScreen()));
$("cpChangeTab").addEventListener("click", async () => { await stopScreen(); await startScreen(); });
$("cpFullscreen").addEventListener("click", () => {
  const v = $("cpScreen").querySelector(".cp-screen-view");
  if (document.fullscreenElement) document.exitFullscreen(); else v.requestFullscreen().catch(() => {});
});

// A frame of the shared screen as JPEG (data URL); asks to share first if nothing is shared yet.
async function cpCaptureShot() {
  if (!screenRec.recorder) {
    await startScreen();
    if (!screenRec.recorder) return "";
  }
  const v = $("cpScreenVideo");
  if (!v.videoWidth) await new Promise((r) => { v.addEventListener("loadeddata", r, { once: true }); setTimeout(r, 1500); });
  if (!v.videoWidth) return "";
  const scale = Math.min(1, 1600 / v.videoWidth);
  const c = document.createElement("canvas");
  c.width = Math.round(v.videoWidth * scale);
  c.height = Math.round(v.videoHeight * scale);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  const flash = $("cpScreen").querySelector(".cp-screen-view");
  flash.classList.add("flash");
  setTimeout(() => flash.classList.remove("flash"), 250);
  return c.toDataURL("image/jpeg", 0.85);
}

async function cpScreenshotAnswer(question = "") {
  const btn = $("apShotBtn");
  btn.disabled = true;
  try {
    const image = await cpCaptureShot();
    if (image) await cpAsk(question, image);
  } finally { btn.disabled = false; }
}

state.cp.attach = false;
$("apAttach").addEventListener("click", () => {
  state.cp.attach = !state.cp.attach;
  $("apAttach").setAttribute("aria-pressed", String(state.cp.attach));
});
$("apAnswerBtn").addEventListener("click", () => cpAsk(""));
$("apShotBtn").addEventListener("click", () => cpScreenshotAnswer(""));

$("apAsk").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("apInput").value.trim();
  if (!q && !state.cp.attach) return cpAsk("");
  $("apInput").value = "";
  if (state.cp.attach) {
    state.cp.attach = false;
    $("apAttach").setAttribute("aria-pressed", "false");
    return cpScreenshotAnswer(q);
  }
  cpAsk(q);
});

// Space answers the last question while a copilot session is on screen (not while typing).
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
  const m = state.current;
  if (!m || m.mode !== "copilot" || m.status !== "recording" || $("liveView").classList.contains("hidden")) return;
  if (e.target.closest("input, textarea, select, [contenteditable], dialog")) return;
  e.preventDefault();
  cpAsk("");
});

function cpCopyAnswer(e, answers) {
  const b = e.target.closest("[data-copy-ans]");
  if (!b) return;
  const a = (answers || []).find((x) => x.id === b.dataset.copyAns);
  if (a) copyText(a.text);
}
$("apList").addEventListener("click", (e) => cpCopyAnswer(e, state.current && state.current.answers));
$("answersPanel").addEventListener("click", (e) => cpCopyAnswer(e, state.current && state.current.answers));

// WebSocket: answers stream in piece by piece
function cpHandle(msg) {
  const cur = state.current;
  if (!cur || cur.id !== msg.meeting_id) return;
  cur.answers = cur.answers || [];
  if (msg.type === "answer") {
    cur.answers.push(msg.answer);
    cpRenderAnswers();
  } else if (msg.type === "answer_delta") {
    const a = cur.answers.find((x) => x.id === msg.id);
    if (!a) return;
    a.text = (a.text || "") + msg.delta;
    const el = document.querySelector(`#apList [data-ans="${msg.id}"] .ap-a`);
    if (!el) return cpRenderAnswers();
    const list = $("apList");
    const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 160;
    el.innerHTML = answerMd(a.text);
    if (nearBottom) list.scrollTop = list.scrollHeight;
  } else if (msg.type === "answer_removed") {
    cur.answers = cur.answers.filter((x) => x.id !== msg.id);
    const el = document.querySelector(`#apList [data-ans="${msg.id}"]`);
    if (el) el.remove();
  } else if (msg.type === "answer_question") {
    const a = cur.answers.find((x) => x.id === msg.id);
    if (a) { a.question = msg.question; const q = document.querySelector(`#apList [data-ans="${msg.id}"] .ap-q`); if (q) q.title = msg.question; }
  } else if (msg.type === "answer_title") {
    const a = cur.answers.find((x) => x.id === msg.id);
    if (!a) return;
    a.title = msg.title;
    const el = document.querySelector(`#apList [data-ans="${msg.id}"] .ap-q span`);
    if (el) el.innerHTML = `<b class="ap-title-q">${esc(msg.title)}</b>`;
  } else if (msg.type === "answer_done") {
    const i = cur.answers.findIndex((x) => x.id === msg.answer.id);
    if (i >= 0) cur.answers[i] = msg.answer; else cur.answers.push(msg.answer);
    if (msg.answer.state === "error") toast(t("cp.answerFailed", { message: msg.answer.error }), true);
    if (cur.status === "recording") cpRenderAnswers(); else cpRenderAnswersPanel();
  }
}

// ------------------------------------------------------------------ finished session: AI answers tab

function cpRenderAnswersPanel() {
  const m = state.current;
  const cp = m && m.mode === "copilot";
  $("answersTab").classList.toggle("hidden", !cp);
  if (!cp) { if (state.tab === "answers") setTab("transcript"); return; }
  $("answersPanel").innerHTML = `<div class="ap-list ap-static">${(m.answers || []).map((a) => cpAnswerHtml(a)).join("")}</div>` +
    ((m.answers || []).length ? "" : `<p class="muted">${esc(t("cp.noAnswers"))}</p>`);
}
