"""Compass, basic: one page on the 2.0 API - threads, their timeline (live), forks and threads of other
agents, runs with their events, sessions and traces (the spans of a run, as a tree; a span's attributes, model
input and output readable), the thread's state, and a composer to start a run, send to the active one, answer
its question or append."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compass</title>
<style>
  :root { --bg:#f6f7f9; --panel:#fff; --ink:#1d2330; --muted:#6b7385; --line:#e3e6ec; --accent:#3d5afe;
          --own:#eef1ff; --other:#fff; --sys:#fff8e6; --warn:#b3261e; --ok:#1e7a3c; --mono:ui-monospace,Menlo,monospace; }
  @media (prefers-color-scheme: dark) { :root { --bg:#14161b; --panel:#1c1f26; --ink:#e7e9ee; --muted:#9097a6;
          --line:#2c313b; --accent:#8c9eff; --own:#252b40; --other:#1c1f26; --sys:#2d2a1e; --warn:#f2b8b5; --ok:#7fd49b; } }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--ink); }
  header { display:flex; align-items:center; gap:12px; padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:16px; margin:0; } header .hint { color:var(--muted); font-size:12px; }
  main { display:grid; grid-template-columns:260px 1fr 380px; height:calc(100vh - 45px); }
  section { overflow:auto; border-right:1px solid var(--line); }
  section:last-child { border-right:none; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:14px 14px 6px; }
  .list a { display:block; padding:8px 14px; color:inherit; text-decoration:none; border-bottom:1px solid var(--line); }
  .list a.on { background:var(--own); } .list a:hover { background:var(--own); }
  .small { font-size:12px; color:var(--muted); } .mono { font-family:var(--mono); font-size:12px; }
  .row { display:flex; gap:8px; align-items:center; padding:8px 14px; flex-wrap:wrap; }
  select, input, textarea, button { font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:6px 8px; }
  button { cursor:pointer; } button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button:disabled { opacity:.45; cursor:default; }
  #timeline { padding:8px 16px 120px; }
  .msg { margin:8px 0; padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--other); max-width:85%; }
  .msg.agent { background:var(--own); margin-left:auto; } .msg.system { background:var(--sys); }
  .msg .who { font-size:12px; color:var(--muted); margin-bottom:2px; }
  .msg pre { white-space:pre-wrap; margin:0; font:inherit; }
  .ev { font-size:12px; color:var(--muted); margin:4px 0; font-family:var(--mono); }
  .ev .k { color:var(--accent); }
  .badge { display:inline-block; padding:1px 6px; border-radius:10px; font-size:11px; border:1px solid var(--line); }
  .s-completed { color:var(--ok); } .s-failed, .s-cancelled, .s-rejected { color:var(--warn); }
  .s-input_required, .s-auth_required { color:#b26a00; }
  #composer { position:sticky; bottom:0; background:var(--panel); border-top:1px solid var(--line); padding:10px 16px; }
  #composer textarea { width:100%; height:60px; resize:vertical; }
  .card { margin:8px 14px; padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }
  .card.on { outline:2px solid var(--accent); }
  .kv { font-family:var(--mono); font-size:12px; white-space:pre-wrap; word-break:break-word; }
  a { color:var(--accent); }
  .empty { color:var(--muted); padding:20px; }
  .span { display:flex; gap:6px; align-items:baseline; padding:3px 8px; border-radius:4px; cursor:pointer; font-family:var(--mono); font-size:12px; }
  .span:hover, .span.on { background:var(--own); }
  .span .n { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .span .d { color:var(--muted); }
  .attrs { margin-top:8px; border-top:1px solid var(--line); padding-top:6px; }
  .attrs details { margin:4px 0; } .attrs summary { cursor:pointer; font-family:var(--mono); font-size:12px; color:var(--accent); }
  .attrs pre { white-space:pre-wrap; word-break:break-word; margin:4px 0 0; padding:6px 8px; background:var(--bg); border-radius:6px;
               font-family:var(--mono); font-size:12px; max-height:360px; overflow:auto; }
  @media (max-width: 900px) { main { grid-template-columns:1fr; height:auto; } section { border-right:none; border-bottom:1px solid var(--line); } }
</style>
</head>
<body>
<header><h1>Compass</h1><span class="hint">llamphouse 2.0 · threads, runs, forks, sessions</span>
  <span style="flex:1"></span><span id="status" class="small"></span></header>
<main>
  <section>
    <h2>New thread</h2>
    <div class="row"><select id="agent"></select><button class="primary" id="new">Create</button></div>
    <h2>Threads</h2>
    <div class="list" id="threads"></div>
  </section>
  <section>
    <div id="head" class="row"></div>
    <div id="timeline"><div class="empty">Pick a thread, or create one.</div></div>
    <div id="composer" hidden>
      <textarea id="text" placeholder="A message…"></textarea>
      <div class="row" style="padding:6px 0 0">
        <span class="small">as</span><input id="who" value="you" size="8">
        <span style="flex:1"></span>
        <button id="append" title="On the thread, for the record (only while no run is active)">Append</button>
        <button id="send" title="To the active run's inbox">Send to run</button>
        <button id="answer" title="Answer the active run's question">Answer</button>
        <button class="primary" id="start" title="Start a run with this message (only while no run is active)">Start run</button>
      </div>
      <div id="error" class="small" style="color:var(--warn)"></div>
    </div>
  </section>
  <section>
    <h2>Runs</h2><div id="runs"></div>
    <h2>Run</h2><div id="run"><div class="empty">Pick a run.</div></div>
    <h2>Trace</h2><div id="trace"><div class="empty">Pick a run.</div></div>
    <h2>Threads it made</h2><div id="tree"></div>
    <h2>State</h2><div id="state" class="card kv"></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const key = new URLSearchParams(location.search).get("key");
const H = key ? {Authorization: "Bearer " + key} : {};
async function api(path, opts = {}) {
  const r = await fetch(path, {...opts, headers: {...H, "Content-Type": "application/json", ...(opts.headers || {})}});
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : r.status);
  return body;
}
let threadId = null, runId = null, source = null, runSource = null;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function text(m) {
  if (!m) return "";
  return (m.parts || []).map(p => p.kind === "text" ? p.text
    : p.kind === "data" ? JSON.stringify(p.data, null, 1)
    : p.kind === "artifact" ? `[artifact ${p.name}: ${(p.parts||[]).map(x => x.name || x.kind).join(", ")}]`
    : p.kind === "file" ? `[file ${p.name || p.uri}]`
    : p.kind === "tool_call" ? `[tool call ${p.name}(${JSON.stringify(p.arguments)})]`
    : p.kind === "tool_result" ? `[tool result]` : `[${p.kind}]`).join("\n");
}
function files(m) {
  const all = [], walk = (ps) => (ps || []).forEach(p => { if (p.kind === "file" && p.uri) all.push(p); if (p.parts) walk(p.parts); });
  walk(m.parts);
  return all.map(p => `<a href="/v2/files/${encodeURIComponent(p.uri.split(/[\/:]/).pop())}${key ? "?key=" + encodeURIComponent(key) : ""}" target="_blank">${esc(p.name || p.uri)}</a>`).join(" ");
}
async function loadAgents() {
  const agents = await api("/v2/agents");
  $("agent").innerHTML = agents.map(a => `<option value="${esc(a.name)}">${esc(a.name)}${a.type === "workflow" ? " (workflow)" : ""}</option>`).join("");
}
async function loadThreads() {
  const ts = await api("/v2/threads?top=true");
  $("threads").innerHTML = ts.map(t => `<a href="#${t.id}" class="${t.id === threadId ? "on" : ""}">
    <b>${esc(t.agent)}</b> ${t.active_run ? `<span class="badge s-${t.active_run.status}">${t.active_run.status}</span>` : ""}
    <div class="small mono">${t.id}</div></a>`).join("") || `<div class="empty">No threads yet.</div>`;
}
function eventLine(e) {
  if (e.message) {
    const s = e.message.sender || {}, cls = s.kind === "agent" ? "agent" : s.kind === "system" ? "system" : "";
    const to = e.to ? ` → <a href="#" data-run="${e.to}">${e.to}</a>` : "";
    return `<div class="msg ${cls}"><div class="who">${esc(s.name || s.id)} · ${s.kind}${e.run ? ` · <a href="#" data-run="${e.run}">${e.run}</a>` : ""}${to} · #${e.position}</div><pre>${esc(text(e.message))}</pre>${files(e.message) ? `<div class="small">${files(e.message)}</div>` : ""}</div>`;
  }
  let what = "";
  if (e.kind === "state.changed") what = (e.data.changes || []).map(c => `${c.key}=${JSON.stringify(c.value)} (v${c.version})`).join(", ");
  else if (e.kind.startsWith("run.")) what = e.run ? `<a href="#" data-run="${e.run}">${e.run}</a> ${e.agent || ""}` : "";
  else if (e.data) what = esc(JSON.stringify(e.data)).slice(0, 200);
  return `<div class="ev">#${e.position} <span class="k">${esc(e.kind)}</span> ${what}</div>`;
}
async function openThread(id) {
  threadId = id; runId = null;
  if (source) source.abort();
  const t = await api(`/v2/threads/${id}`);
  $("head").innerHTML = `<b>${esc(t.agent)}</b> <span class="mono small">${t.id}</span>
    ${t.forked_from ? `<span class="small">fork of <a href="#${t.forked_from.thread}">${t.forked_from.thread}</a> @${t.forked_from.position}</span>` : ""}
    ${t.closed ? `<span class="badge">closed</span>` : ""}`;
  $("composer").hidden = false;
  $("timeline").innerHTML = "";
  const mine = source = new AbortController();
  follow(`/v2/threads/${id}/events/stream?after=0`, mine.signal, (e) => {
    $("timeline").insertAdjacentHTML("beforeend", eventLine(e));
    $("timeline").parentElement.scrollTop = 1e9; refreshSoon(); }).catch(() => {});
  await refreshSide(); loadThreads();
}
async function follow(path, signal, each) {
  // SSE over fetch: every event kind, and the Authorization header
  const r = await fetch(path, {headers: H, signal});
  const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
  for (;;) {
    const {value, done} = await reader.read(); if (done) return;
    buf += dec.decode(value, {stream: true});
    let i; while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const data = chunk.split("\n").filter(l => l.startsWith("data: ")).map(l => l.slice(6)).join("\n");
      if (data) each(JSON.parse(data));
    }
  }
}
let soon = null;
function refreshSoon() { clearTimeout(soon); soon = setTimeout(refreshSide, 150); }
let sideBusy = false;
async function refreshSide() {
  if (!threadId || sideBusy) return; sideBusy = true;
  try {
    const t = await api(`/v2/threads/${threadId}`);
    const runs = await api(`/v2/threads/${threadId}/runs`);
    const active = t.active_run;
    $("runs").innerHTML = runs.map(r => `<div class="card ${r.id === runId ? "on" : ""}"><a href="#" data-run="${r.id}" class="mono">${r.id}</a>
      <span class="badge s-${r.status}">${r.status}</span><div class="small">${esc(text(r.input)).slice(0, 80)}</div></div>`).join("") || `<div class="empty">No runs.</div>`;
    $("append").disabled = !!active; $("start").disabled = !!active;
    $("send").disabled = !active; $("answer").disabled = !(active && active.pending);
    const created = t.created.length ? (await Promise.all(t.created.map(c => api(`/v2/threads/${c}`)))) : [];
    $("tree").innerHTML = created.map(c => `<div class="card"><a href="#${c.id}">${c.forked_from ? "fork" : esc(c.agent)}</a>
      <span class="mono small">${c.id}</span> ${c.active_run ? `<span class="badge s-${c.active_run.status}">${c.active_run.status}</span>` : `<span class="small">idle</span>`}</div>`).join("") || `<div class="empty">None.</div>`;
    const st = await api(`/v2/threads/${threadId}/state`);
    $("state").textContent = Object.keys(st).length ? Object.entries(st).map(([k, v]) => `${k} (v${v.version}): ${JSON.stringify(v.value)}`).join("\n") : "empty";
    if (!runId && runs.length) runId = (active || runs[runs.length - 1]).id;
    if (runId) await showRun(runId, false);
  } finally { sideBusy = false; }
}
async function showRun(id, fresh = true) {
  runId = id;
  const r = await api(`/v2/runs/${id}`);
  let session = null; try { session = await api(`/v2/runs/${id}/session`); } catch {}
  const events = await api(`/v2/runs/${id}/events`);
  $("run").innerHTML = `<div class="card"><div><span class="mono">${r.id}</span> <span class="badge s-${r.status}">${r.status}</span>
      ${r.parent ? `<span class="small">child of ${r.parent}</span>` : ""}</div>
    ${r.pending ? `<div class="small" style="margin-top:6px">asks: <b>${esc(r.pending.reason)}</b></div>` : ""}
    ${r.error ? `<div class="small" style="color:var(--warn)">error: ${esc(r.error)}</div>` : ""}
    ${r.output != null ? `<div class="kv">output: ${esc(JSON.stringify(r.output, null, 1))}</div>` : ""}
    ${r.unread.length ? `<div class="small">unread: ${r.unread.map(m => esc(text(m))).join(" · ")}</div>` : ""}
    <div class="small" style="margin-top:6px">events</div>
    <div class="kv">${events.slice(-30).map(e => `${e.seq} ${e.kind}${e.label ? " " + esc(e.label) : ""}${e.delta ? " " + esc(e.delta) : ""}`).join("\n") || "none"}</div>
    ${session ? `<div class="small" style="margin-top:6px">session ${session.id} · v${session.version} · ${session.revisions} revisions${session.revising ? " · revising" : ""}</div>
      <div class="kv">${session.messages.map(m => `${m.own ? "own  " : "other"} ${esc(text(m)).slice(0, 140)}`).join("\n")}</div>
      ${session.log.length ? `<div class="kv small">${session.log.map(l => esc(l.kind)).join(" · ")}</div>` : ""}` : ""}
  </div>`;
  if (fresh) { spanId = null; refreshSide(); }
  await showTrace(id);
}
// the run's trace: its spans as a tree; the one picked, its attributes - model input and output readable
let spanId = null, spans = [];
const opened = new Set();  // attributes unfolded: kept across refreshes
const READABLE = ["gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions", "input.value", "output.value"];
function ms(ns) { const v = (ns || 0) / 1e6; return v >= 1000 ? (v / 1000).toFixed(2) + " s" : v.toFixed(v >= 10 ? 0 : 1) + " ms"; }
function spanStatus(s) { return s.StatusCode === "STATUS_CODE_ERROR" ? `<span class="badge s-failed">error</span>`
  : s.StatusCode === "STATUS_CODE_OK" ? `<span class="badge s-completed">ok</span>` : ""; }
function partText(p) {
  if (typeof p === "string") return p;
  if (p.type === "text" || p.type === "reasoning") return p.content ?? p.text ?? "";
  if (p.type === "tool_call") return `[tool call ${p.name}(${JSON.stringify(p.arguments)})]`;
  if (p.type === "tool_call_response") return `[tool result] ${typeof p.response === "string" ? p.response : JSON.stringify(p.response, null, 2)}`;
  return JSON.stringify(p, null, 2);
}
function readable(value) {
  // a JSON string: messages ({role, parts | content}) as a conversation, anything else pretty-printed
  let v = value;
  if (typeof v === "string") { try { v = JSON.parse(v); } catch { return v; } }
  if (Array.isArray(v) && v.length && v.every(m => m && typeof m === "object" && "role" in m)) {
    return v.map(m => {
      const body = Array.isArray(m.parts) ? m.parts.map(partText).join("\n")
        : typeof m.content === "string" ? m.content : Array.isArray(m.content) ? m.content.map(partText).join("\n")
        : JSON.stringify(m.content ?? m, null, 2);
      return `── ${m.role}${m.name ? " (" + m.name + ")" : ""}${m.finish_reason ? " · " + m.finish_reason : ""}\n${body}`;
    }).join("\n\n");
  }
  return typeof v === "string" ? v : JSON.stringify(v, null, 2);
}
function fold(k, title, body) {
  return `<details data-k="${esc(k)}" ${opened.has(k) ? "open" : ""}><summary>${esc(title)}</summary><pre>${esc(body)}</pre></details>`;
}
async function showTrace(id) {
  let t; try { t = await api(`/v2/runs/${id}/trace`); } catch (e) { $("trace").innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  if (id !== runId) return;
  if (!t.tracing) { $("trace").innerHTML = `<div class="empty">Tracing is off.</div>`; return; }
  spans = t.spans;
  if (!spans.length) { $("trace").innerHTML = `<div class="empty">No spans yet.</div>`; return; }
  const ids = new Set(spans.map(s => s.SpanId)), kids = new Map();
  spans.forEach(s => { const p = ids.has(s.ParentSpanId) ? s.ParentSpanId : ""; kids.set(p, [...(kids.get(p) || []), s]); });
  const rows = [], walk = (parent, depth) => (kids.get(parent) || []).forEach(s => {
    rows.push(`<div class="span ${s.SpanId === spanId ? "on" : ""}" data-span="${esc(s.SpanId)}" style="padding-left:${8 + depth * 14}px">
      <span class="n" title="${esc(s.SpanName)}">${esc(s.SpanName)}</span><span class="d">${ms(s.Duration)}</span>${spanStatus(s)}</div>`);
    walk(s.SpanId, depth + 1); });
  walk("", 0);
  const s = spans.find(x => x.SpanId === spanId);
  let detail = "";
  if (s) {
    const a = s.SpanAttributes || {}, keys = Object.keys(a).sort();
    const big = keys.filter(k => READABLE.includes(k) || (typeof a[k] === "string" && a[k].length > 200));
    const small = keys.filter(k => !big.includes(k));
    const events = (s["Events.Name"] || []).map((n, i) => fold(`${s.SpanId}/event/${i}`, `event: ${n}`, readable((s["Events.Attributes"] || [])[i] || {})));
    detail = `<div class="attrs"><div class="small">${esc(s.SpanName)} · ${ms(s.Duration)} · ${esc(s.Timestamp)}${s.StatusMessage ? " · " + esc(s.StatusMessage) : ""}</div>
      <div class="kv">${small.map(k => `${esc(k)}: ${esc(typeof a[k] === "string" ? a[k] : JSON.stringify(a[k]))}`).join("\n")}</div>
      ${big.map(k => fold(`${s.SpanId}/${k}`, k, readable(a[k]))).join("")}${events.join("")}</div>`;
  }
  $("trace").innerHTML = `<div class="card">${rows.join("")}${detail || `<div class="small" style="margin-top:6px">Pick a span for its attributes.</div>`}</div>`;
}
document.addEventListener("toggle", (ev) => { const k = ev.target.dataset && ev.target.dataset.k;
  if (k) ev.target.open ? opened.add(k) : opened.delete(k); }, true);
document.addEventListener("click", (ev) => {
  const sp = ev.target.closest("[data-span]");
  if (sp) { spanId = sp.dataset.span; if (runId) showTrace(runId); return; }
  const a = ev.target.closest("[data-run]"); if (!a) return; ev.preventDefault(); showRun(a.dataset.run);
});
window.addEventListener("hashchange", () => { const id = location.hash.slice(1); if (id) openThread(id); });
function body() { return {text: $("text").value, sender: {kind: "person", id: $("who").value || "you"}}; }
async function act(fn) {
  $("error").textContent = "";
  try { await fn(); $("text").value = ""; await refreshSide(); } catch (e) { $("error").textContent = e.message; }
}
$("new").onclick = () => act(async () => { const t = await api("/v2/threads", {method: "POST", body: JSON.stringify({agent: $("agent").value})}); location.hash = t.id; });
$("start").onclick = () => act(() => api(`/v2/threads/${threadId}/runs`, {method: "POST", body: JSON.stringify(body())}));
$("append").onclick = () => act(() => api(`/v2/threads/${threadId}/messages`, {method: "POST", body: JSON.stringify(body())}));
$("send").onclick = () => act(async () => { const t = await api(`/v2/threads/${threadId}`);
  await api(`/v2/runs/${t.active_run.id}/messages`, {method: "POST", body: JSON.stringify(body())}); });
$("answer").onclick = () => act(async () => { const t = await api(`/v2/threads/${threadId}`);
  await api(`/v2/runs/${t.active_run.id}/answer`, {method: "POST", body: JSON.stringify(body())}); });
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
  ($("start").disabled ? ($("answer").disabled ? $("send") : $("answer")) : $("start")).click(); } });
(async () => { $("status").textContent = "connected"; await loadAgents(); await loadThreads();
  setInterval(() => { loadThreads(); if (threadId) refreshSide(); }, 4000);
  if (location.hash.length > 1) openThread(location.hash.slice(1)); })()
  .catch(e => $("status").textContent = "error: " + e.message);
</script>
</body>
</html>
"""
