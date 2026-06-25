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
const dur = (s) => { s = Math.round(s || 0); return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`; };
const levels = (m) => `${m.best_levels ?? 0} / ${m.total_levels ?? "?"}`;

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
      `${g.state.problem_name || "?"} · ${m.n_turns} iters · ${m.n_tool_calls} tools · ${m.n_submissions} submits`));
    card.append(h("div", null, statusBadge(m), " ",
      h("span", "badge", `levels ${levels(m)}`), " ",
      h("span", "badge", dur(m.duration_s)), " ",
      h("span", "badge", `out ${fmt(m.tokens.output)} tok`)));
    card.onclick = () => { location.hash = "game/" + encodeURIComponent(g.name); };
    grid.append(card);
  }
  clear(app); app.append(grid);
}

function statusOf(m) {
  return m.last_error_category || m.exit_reason || "running";  // no exit_reason yet ⇒ still running
}
function statusBadge(m) {
  const running = !m.last_error_category && !m.exit_reason;
  const cls = m.last_error_category ? "b-bad" : (running ? "b-warn" : "b-good");
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
    kpi("Iterations", m.n_turns, "agent turns"),
    kpi("Tool calls", m.n_tool_calls),
    kpi("Submissions", m.n_submissions),
    kpi("Output tokens", fmt(m.tokens.output), `cache ${fmt(m.tokens.cache_read)}`),
    kpi("Levels", levels(m), `reached / total`),
    kpi("Time", dur(m.duration_s)),
    kpi("Success", pct(m.success_rate)),
    kpi("Thinking", fmt(m.thinking_chars) + " ch"),
    kpi("Cheat attempts", m.cheat_attempts ?? 0));
  wrap.append(kpis, configBlock(d.config), barChart("Tool calls", m.tool_histogram));
  if (m.submission_trajectory.length) wrap.append(trajectory(m.submission_trajectory));
  shell(name, "", wrap);
}

function configBlock(c) {
  if (!c || !Object.keys(c).length) return h("div");
  const a = c.agent || {}, p = c.problem || {}, lim = c.limits || {}, sec = c.security || {};
  const args = a.args && Object.keys(a.args).length ? " · " + JSON.stringify(a.args) : "";
  const rows = [
    ["agent", `${a.name ?? "?"}${a.model ? " · " + a.model : ""}${args}`],
    ["problem", `${p.name ?? "?"} · ${p.lifecycle ?? "?"} · info=${p.info_mode ?? "?"} · obs=${p.obs_mode ?? "?"}`],
    ["features", (c.features || []).join(", ")],
    ["task_names", (c.task_names || []).join(", ") || "(all)"],
    ["limits", `keep_alive ${lim.keep_alive ?? "?"} · max_moves ${lim.max_moves ?? "?"}`],
    ["security", `sandbox ${sec.sandbox ?? "?"} · deny_egress ${sec.deny_egress}`],
  ];
  const wrap = h("div"); wrap.append(h("h2", null, "Run config"));
  const t = h("table");
  for (const [k, v] of rows) t.append(rowEl("td", [k, String(v)]));
  wrap.append(t); return wrap;
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
const _TAG_LABEL = { submit: "submit", submit_win: "submit ✓ level", cheat: "cheat" };

async function renderConversation(name) {
  const d = await gameDetail(name);
  const conv = h("div", "conv");
  const navItems = [];     // {id, tag, label} — submissions + cheats, to jump to
  let nSubmit = 0, nCheat = 0;
  if (!d.turns.length) conv.append(h("div", "muted", "no transcript"));
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
      } else if (it.kind === "system") {
        const det = h("details", "think"); det.append(h("summary", null, "⚙ system prompt"), h("pre", null, it.text));
        body.append(det);
      } else if (it.kind === "user") {
        const det = h("details", "think"); det.append(h("summary", null, "📨 sent to the agent"), h("pre", null, it.text));
        body.append(det);
      } else if (it.kind === "text") {
        body.append(h("pre", "text", it.text));
      } else if (it.kind === "tool" && it.tool) {
        const block = toolBlock(it.tool);
        const tag = it.tool.tag;
        if (tag === "submit" || tag === "submit_win") {
          block.id = "nav-submit-" + nSubmit;
          navItems.push({ id: block.id, tag, label: `submission ${nSubmit}${tag === "submit_win" ? " ✓ level" : ""}` });
          nSubmit++;
        } else if (tag === "cheat") {
          block.id = "nav-cheat-" + nCheat;
          navItems.push({ id: block.id, tag, label: `cheat ${nCheat + 1}` });
          nCheat++;
        }
        body.append(block);
      }
    }
    if (t.error) body.append(h("pre", "text", t.error.message));
    turn.append(head, body);
    conv.append(turn);
  });

  const nav = h("div", "convnav");
  nav.append(h("div", "h", "Jump to"));
  if (navItems.length) {
    for (const it of navItems) {
      const a = h("a", "navitem nav-" + it.tag, it.label);
      a.onclick = (e) => {
        e.preventDefault();
        document.getElementById(it.id)?.scrollIntoView({ behavior: "smooth", block: "center" });
      };
      nav.append(a);
    }
  } else {
    nav.append(h("div", "muted", "no submission yet"));
  }
  const layout = h("div", "convlayout");
  layout.append(nav, conv);
  shell(name, "conversation", layout);
}

function toolBlock(tool) {
  // The reader tags calls authoritatively: blue submit, green submit-that-won a level, red cheat.
  const cls = { cheat: " cheat", submit: " submit", submit_win: " submit-win" }[tool.tag] || "";
  const box = h("div", "tool" + cls);
  const inp = JSON.stringify(tool.input);
  const t = h("div", "t");
  if (tool.tag) t.append(h("span", "tag tag-" + tool.tag, _TAG_LABEL[tool.tag]), " ");
  t.append(h("b", null, tool.name), " ",
    h("span", "muted", inp.length > 240 ? inp.slice(0, 240) + "…" : inp));
  box.append(t);
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
