"use strict";
// All DOM is built with createElement + textContent (via h()); no innerHTML, so
// transcript/log content is inserted as text, never parsed as HTML (XSS-safe).
const app = document.getElementById("app");
const crumb = document.getElementById("crumb");
document.getElementById("brand").onclick = () => { location.hash = ""; };

function h(tag, cls, ...kids) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  for (const k of kids) e.append(k && k.nodeType ? k : document.createTextNode(k ?? ""));
  return e;
}
const clear = (el) => el.replaceChildren();
const fmt = (n) => (n == null ? "—" : Intl.NumberFormat().format(n));
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(0) + "%");

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

const _cache = {};               // game name -> detail payload (shared across tabs)
async function gameDetail(name) {
  if (!_cache[name]) _cache[name] = await api("/api/game/" + encodeURIComponent(name));
  return _cache[name];
}

// ---------------------------------------------------------------- dashboard
async function renderDashboard() {
  crumb.textContent = "";
  const data = await api("/api/games");
  crumb.textContent = `${data.experiment} · ${data.games.length} game(s)`;
  if (data.games.length === 1) { location.hash = "game/" + encodeURIComponent(data.games[0].name); return; }
  const grid = h("div", "grid");
  for (const g of data.games) {
    const m = g.metrics;
    const card = h("div", "card click", h("h3", null, g.name));
    card.append(h("div", "muted",
      `${g.state.problem_name || "?"} · ${m.n_turns} turns · ${m.n_tool_calls} tools · ${m.n_submissions} submits`));
    card.append(h("div", null, statusBadge(m), " ",
      h("span", "badge", `levels ${m.best_levels ?? "—"}`), " ",
      h("span", "badge", `out ${fmt(m.tokens.output)} tok`)));
    card.onclick = () => { location.hash = "game/" + encodeURIComponent(g.name); };
    grid.append(card);
  }
  clear(app); app.append(grid);
}

function statusOf(m) {
  return m.last_error_category ? m.last_error_category : (m.exit_requested ? "exited" : "stopped");
}
function statusBadge(m) {
  const cls = m.last_error_category ? "b-bad" : (m.exit_requested ? "b-good" : "b-warn");
  return h("span", "badge " + cls, statusOf(m));
}

// ---------------------------------------------------------------- per-game shell
const TABS = [["", "Overview"], ["conversation", "Conversation"], ["artifacts", "Artifacts"], ["logs", "Logs"]];

function shell(name, active, body) {
  crumb.textContent = name;
  const nav = h("div", "tabs");
  for (const [slug, label] of TABS) {
    const href = "#game/" + encodeURIComponent(name) + (slug ? "/" + slug : "");
    const a = h("a", "tab" + (slug === active ? " on" : ""), label);
    a.href = href;
    nav.append(a);
  }
  clear(app); app.append(nav, body);
}

// ---------------------------------------------------------------- overview tab
async function renderOverview(name) {
  const d = await gameDetail(name);
  const m = d.metrics;
  const wrap = h("div");
  const kpis = h("div", "kpis");
  kpis.append(
    kpi("Status", statusOf(m)),
    kpi("Turns", m.n_turns),
    kpi("Tool calls", m.n_tool_calls),
    kpi("Submissions", m.n_submissions),
    kpi("Output tokens", fmt(m.tokens.output), `cache ${fmt(m.tokens.cache_read)}`),
    kpi("Levels", `${m.best_levels ?? "—"}`, `final ${m.final_levels ?? "—"}`),
    kpi("Success", pct(m.success_rate)),
    kpi("Thinking", fmt(m.thinking_chars) + " ch"));
  wrap.append(kpis, barChart("Tool calls", m.tool_histogram));
  if (m.submission_trajectory.length) wrap.append(trajectory(m.submission_trajectory));
  shell(name, "", wrap);
}

function kpi(label, value, sub) {
  const k = h("div", "kpi", h("div", "v", String(value)));
  k.append(h("div", "l", label + (sub ? ` · ${sub}` : "")));
  return k;
}
function barChart(title, obj) {
  const wrap = h("div"); wrap.append(h("h2", null, title));
  const max = Math.max(1, ...Object.values(obj));
  for (const [n, c] of Object.entries(obj)) {
    const row = h("div", "barrow", h("div", null, n));
    const bar = h("div", "bar"); bar.style.width = `${(c / max) * 100}%`;
    row.append(bar, h("div", "n", String(c)));
    wrap.append(row);
  }
  if (!Object.keys(obj).length) wrap.append(h("div", "muted", "none"));
  return wrap;
}
function trajectory(traj) {
  const wrap = h("div"); wrap.append(h("h2", null, "Score per submission"));
  const t = h("table");
  t.append(rowEl("th", ["#", "success", "levels", "mean_steps", "error"]));
  for (const s of traj)
    t.append(rowEl("td", [s.submission, pct(s.success_rate), s.levels ?? "—", s.mean_steps ?? "—", s.error || ""]));
  wrap.append(t); return wrap;
}
function rowEl(cell, vals) {
  const tr = h("tr"); for (const v of vals) tr.append(h(cell, null, String(v))); return tr;
}

// ---------------------------------------------------------------- conversation tab
async function renderConversation(name) {
  const d = await gameDetail(name);
  const wrap = h("div");
  if (!d.turns.length) wrap.append(h("div", "muted", "no transcript"));
  d.turns.forEach((t, i) => {
    const turn = h("div", "turn");
    const u = t.usage || {};
    const head = h("div", "head", `turn ${i + 1}`);
    if (u.output_tokens != null) head.append(h("span", null, `· out ${fmt(u.output_tokens)} tok`));
    if (t.error) head.append(h("span", "badge b-bad", t.error.category));
    const body = h("div", "body");
    for (const it of t.items || []) {     // chronological order
      if (it.kind === "thinking") {
        const det = h("details", "think"); det.append(h("summary", null, "💭 thinking"), h("pre", null, it.text));
        body.append(det);
      } else if (it.kind === "text") {
        body.append(h("pre", "text", it.text));
      } else if (it.kind === "tool" && it.tool) {
        body.append(toolBlock(it.tool));
      }
    }
    if (t.error) body.append(h("pre", "text", t.error.message));
    turn.append(head, body);
    wrap.append(turn);
  });
  shell(name, "conversation", wrap);
}
function toolBlock(tool) {
  const isSubmit = /submit|exit/i.test(tool.name);
  const box = h("div", "tool" + (isSubmit ? " submit" : ""));
  const inp = JSON.stringify(tool.input);
  box.append(h("div", "t", h("b", null, tool.name), " ",
    h("span", "muted", inp.length > 240 ? inp.slice(0, 240) + "…" : inp)));
  if (tool.result != null) {
    const res = h("div", "res" + (tool.is_error ? " err" : ""));
    res.append(h("pre", null, String(tool.result).slice(0, 4000)));
    box.append(res);
  }
  return box;
}

// ---------------------------------------------------------------- artifacts tab
async function renderArtifacts(name) {
  const d = await api("/api/game/" + encodeURIComponent(name) + "/artifacts");
  const wrap = h("div", "split");
  const list = h("div", "filelist");
  const view = h("div", "fileview", h("div", "muted", "select a file"));
  list.append(h("div", "h", "Workdir files"));
  for (const f of d.files) {
    const item = h("div", "fileitem", f.relpath);
    item.onclick = () => {
      [...list.querySelectorAll(".fileitem")].forEach((x) => x.classList.remove("on"));
      item.classList.add("on");
      clear(view);
      view.append(h("h3", null, f.relpath),
        f.too_large ? h("div", "muted", "(too large to show)") : h("pre", "code", f.content));
    };
    list.append(item);
  }
  if (!d.files.length) list.append(h("div", "muted", "none"));
  wrap.append(list, view);

  const subs = h("div"); subs.append(h("h2", null, "Submissions & videos"));
  if (!d.submissions.length) subs.append(h("div", "muted", "no submissions"));
  for (const s of d.submissions) {
    const c = h("div", "sub", h("h3", null, "submission " + s.name));
    if (s.error) c.append(h("div", "badge b-bad", s.error));
    const a = s.aggregate || {};
    c.append(h("div", "muted",
      `success ${pct(a.success_rate)} · levels ${a.mean_levels_completed ?? "—"} · steps ${a.mean_steps ?? "—"} · n=${a.n_episodes ?? "—"}`));
    for (const v of s.videos || []) {
      const vid = h("video"); vid.controls = true; vid.preload = "metadata";
      vid.src = `/video/${encodeURIComponent(name)}/${encodeURIComponent(s.name)}/${encodeURIComponent(v)}`;
      c.append(vid);
    }
    subs.append(c);
  }
  shell(name, "artifacts", h("div", null, wrap, subs));
}

// ---------------------------------------------------------------- logs tab
async function renderLogs(name) {
  const d = await api("/api/game/" + encodeURIComponent(name) + "/logs");
  const wrap = h("div");
  wrap.append(h("h2", null, "Events"));
  const errs = d.events.filter((e) => e.level === "ERROR" || e.error_category);
  if (errs.length) {
    const warn = h("div", "card"); warn.style.borderColor = "var(--bad)";
    warn.append(h("b", "bad", `${errs.length} error event(s)`));
    for (const e of errs) warn.append(h("pre", "err", `${e.event} (${e.error_category || ""}) ${JSON.stringify(e.detail || {})}`));
    wrap.append(warn);
  }
  const t = h("table");
  t.append(rowEl("th", ["component", "level", "event", "phase", "error"]));
  for (const e of d.events) {
    const tr = rowEl("td", [e.component, e.level, e.event, e.phase || "", e.error_category || ""]);
    if (e.level === "ERROR" || e.error_category) tr.classList.add("err");
    t.append(tr);
  }
  if (!d.events.length) wrap.append(h("div", "muted", "no events.jsonl"));
  else wrap.append(t);
  wrap.append(h("h2", null, "output.log"));
  wrap.append(h("pre", "code", d.output || "(empty)"));
  shell(name, "logs", wrap);
}

// ---------------------------------------------------------------- routing
async function route() {
  try {
    const parts = (location.hash || "").replace(/^#\/?/, "").split("/").filter(Boolean);
    if (parts[0] !== "game" || !parts[1]) return renderDashboard();
    const name = decodeURIComponent(parts[1]);
    const tab = parts[2] || "";
    if (tab === "conversation") await renderConversation(name);
    else if (tab === "artifacts") await renderArtifacts(name);
    else if (tab === "logs") await renderLogs(name);
    else await renderOverview(name);
  } catch (e) {
    clear(app); app.append(h("pre", "err", "error: " + e.message));
  }
}
window.addEventListener("hashchange", route);
route();
