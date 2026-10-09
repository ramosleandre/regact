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
const pct1 = (x) => (x == null ? "\u2014" : (x * 100).toFixed(1) + "%");  // low scores (RHAE) need the decimal
const dur = (s) => { s = Math.round(s || 0); return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`; };
// Game-agnostic: format an opaque aggregate dict as "key val · key val", skipping
// bookkeeping keys. Each game owns its metric names (ARC: levels/rhae, MiniGrid: reward).
const _SKIP_KEYS = new Set(["n_episodes", "n_errors"]);
const fmtMetric = (v) => (v == null ? "—" : typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : v);
const aggLine = (agg) =>
  Object.entries(agg || {})
    .filter(([k, v]) => !_SKIP_KEYS.has(k) && typeof v === "number")
    .map(([k, v]) => `${k} ${fmtMetric(v)}`)
    .join(" · ") || "—";

let navigation = new AbortController();
let pageCleanups = [];
function checkNavigation(token) {
  if (token !== navigation || token.signal.aborted) throw new DOMException("Navigation changed", "AbortError");
}
async function api(path, opts = {}) {
  const token = navigation;
  const r = await fetch(path, {cache: "no-store", signal: token.signal, ...opts});
  if (!r.ok) {
    const text = await r.text();
    const hint = r.status === 404 && path.startsWith('/api/game/cwm')
      ? " The viewer server may predate CWM support; restart make viz." : "";
    throw new Error(`HTTP ${r.status}: ${text}${hint}`);
  }
  const data = await r.json();
  checkNavigation(token);
  return data;
}

const _cache = {};               // game name -> detail payload (shared across tabs)
async function gameDetail(name) {
  if (!_cache[name]) _cache[name] = await api("/api/game?name=" + encodeURIComponent(name));
  return _cache[name];
}

// ---------------------------------------------------------------- dashboard
const _gamesCache = {};                // /api/games payloads, keyed by the `under` subtree scope
async function gamesData(under = "") { // scoped, so the browser never parses the whole root at once
  if (!(under in _gamesCache))
    _gamesCache[under] = await api("/api/games" + (under ? "?under=" + encodeURIComponent(under) : ""));
  return _gamesCache[under];
}

// Top-level panels for a scoped subtree: the game grid ("Experiments") + cross-run "Graphs", plus a
// link back to the browse tree. `under` (the subtree path) rides each link so the scope holds.
function panelNav(active, under = "") {
  const nav = h("div", "panelnav");
  const suffix = under ? "/" + encodeURIComponent(under) : "";
  // Back goes ONE logical level up (benchmark -> global, experiment -> benchmark), never through
  // the run/timestamp level. Empty parent -> the global browse landing.
  const parent = under.split("/").slice(0, -1).join("/");
  const back = h("a", "panel", "< back");
  back.href = parent ? "#run/" + encodeURIComponent(parent) : "#";
  nav.append(back);
  for (const [slug, label] of [["run", "Experiments"], ["graphs", "Graphs"]]) {
    const on = (slug === "run" && active === "") || slug === active;
    const a = h("a", "panel" + (on ? " on" : ""), label);
    a.href = "#" + slug + suffix;
    nav.append(a);
  }
  return nav;
}

function gameCard(g) {
  const m = g.metrics;
  const stamp = g.name.split("/").slice(-2, -1)[0] || "";  // the run's timestamp dir (tells reruns apart)
  const head = h("div", "cardhead", taskThumb(g.task || g.name, 40), h("h3", null, g.task || g.name));
  const card = h("div", "card click", head);
  card.append(h("div", "muted",
    `${stamp} · ${m.n_turns} iters · ${m.n_tool_calls} tools · ${m.n_submissions} submits`));
  card.append(h("div", null, statusBadge(m), " ",
    h("span", "badge", aggLine(m.final_aggregate)), " ",
    h("span", "badge", dur(m.duration_s)), " ",
    h("span", "badge", `out ${fmt(m.tokens.output)} tok`)));
  card.onclick = () => { location.hash = "game/" + encodeURIComponent(g.name); };
  return card;
}

async function renderDashboard(under = "") {
  const token = navigation;
  crumb.textContent = "";
  const data = await gamesData(under);
  await loadIcons();   // the per-experiment header shows its agent + model contestant icons
  checkNavigation(token);
  crumb.textContent = `${under || data.experiment} · ${data.games.length} game(s)`;
  // Always list the tasks (even a single one), so pointing at an experiment shows the experiment
  // interface rather than diving straight into the lone game's overview.
  const byExp = new Map();          // one section per experiment, its task cards beneath
  for (const g of data.games) {
    if (!byExp.has(g.experiment)) byExp.set(g.experiment, []);
    byExp.get(g.experiment).push(g);
  }
  clear(app); app.append(panelNav("", under));
  for (const [exp, games] of byExp) {
    // An experiment may contain several agent/model configurations: show each recorded pair.
    const agents = new Map(games.map((g) => {
      const a = agentIdentity(g.agent);
      return [JSON.stringify(a), a];
    }));
    const head = h("h2", "expsection exphead", ...[...agents.values()].map(agentVsModel),
      h("span", "exptitle", expLeaf(exp)), h("span", "muted", ` · ${games.length} run(s)`));
    app.append(head);
    const grid = h("div", "grid");
    for (const g of games) grid.append(gameCard(g));
    app.append(grid);
  }
}

// What the viewer's root itself is decides whether we land on the browse lists or go straight to a
// dashboard. If EVERY top folder is a run/task the root is ONE experiment; if every top folder is an
// experiment the root is ONE benchmark - either way open its dashboard (the benchmark/experiment
// interface). Only a mixed collection (many benchmarks + bare experiments + legacy folders, i.e. the
// experiments root) gets the browse landing.
function _rootIsSingleScope(tree) {
  const kinds = new Set(tree.map((n) => n.kind));
  const onlyRunsOrTasks = [...kinds].every((k) => k === "run" || k === "task");
  return onlyRunsOrTasks || (kinds.size === 1 && kinds.has("experiment"));
}

// The browse landing: three plain clickable lists - Benchmarks, Experiments, and Undetected (legacy
// / ambiguous folders we cannot confidently place). Each row opens that folder's scoped dashboard.
const _BROWSE_LISTS = [
  { title: "Benchmarks", of: (n) => n.kind === "benchmark", unit: "experiment", count: (n) => n.n_children },
  { title: "Experiments", of: (n) => n.kind === "experiment", unit: "run", count: (n) => n.n_children },
  { title: "Undetected", of: (n) => n.kind !== "benchmark" && n.kind !== "experiment", unit: "task", count: (n) => n.n_tasks },
];

async function renderBrowse() {
  const token = navigation;
  crumb.textContent = "";
  const { root, tree } = await api("/api/tree");
  checkNavigation(token);
  crumb.textContent = root;
  if (!tree.length) { clear(app); app.append(h("div", "muted", "no runs under this folder")); return; }
  if (_rootIsSingleScope(tree)) return await renderDashboard("");
  clear(app);
  for (const list of _BROWSE_LISTS) {
    const items = tree.filter(list.of);
    if (!items.length) continue;
    app.append(h("h2", "expsection", list.title));
    const box = h("div", "tree");
    for (const node of items) box.append(browseItem(node, list.unit, list.count(node)));
    app.append(box);
  }
}

// A plain folder row (no box): click opens the scoped dashboard for its subtree (same "run" route
// the launch deep-link uses; renderDashboard groups the games under it by experiment and adds Graphs).
function browseItem(node, unit, n) {
  const link = h("span", "tlink trun", "> " + node.name);
  link.onclick = () => { location.hash = "run/" + encodeURIComponent(node.path); };
  return h("div", "tnode", link, h("span", "muted", ` · ${n} ${unit}${n === 1 ? "" : "s"}`));
}

function statusOf(m) {
  return m.last_error_category || m.exit_reason || "running";  // no exit_reason yet ⇒ still running
}
function statusBadge(m) {
  const running = !m.last_error_category && !m.exit_reason;
  const cls = m.last_error_category ? "b-bad" : (running ? "b-warn" : "b-good");
  return h("span", "badge " + cls, statusOf(m));
}

// ---------------------------------------------------------------- graphs (cross-run)
// Per-task metrics aggregated across every run of a task, grouped-bar per experiment. Framework
// metrics (below) exist for every run; problem-specific ones (mean_steps/reward, ARC levels/rhae…)
// are auto-discovered from each run's final_aggregate so the panel stays game-agnostic.
const _THEME = { good: "#4ec9a4", warn: "#e0c060", bad: "#e06c6c", muted: "#9aa3b2" };
const _EXP_PALETTE = ["#5aa9e6", "#4ec9a4", "#e0c060", "#e06c6c", "#7d8bd4", "#d47db0", "#8bd47d", "#d4a37d"];
const _AGG_SKIP = new Set(["n_episodes", "n_errors", "success_rate"]);  // shown elsewhere / bookkeeping
// `def: true` = a Main metric: shown in the game overview's Main-metrics table AND activated by
// default in the Graphs panel. Score metrics carry `score: <key>` instead: they are Main only when
// the run's problem lists that key in its main_metrics (ARC: levels/RHAE; MiniGrid: success/reward).
// ONE registry drives BOTH the Graphs panel and the game Overview (renderOverview reads the same
// list), so metric names / order / capitalization are identical everywhere by construction. The
// game's own aggregate keys (success_rate, mean_steps, mean_reward, ARC levels, ...) are appended
// dynamically in metricSpecs, so e.g. the "Score" line in the overview is split into those keys.
const FRAMEWORK_METRICS = [
  { key: "success_rate", label: "Success rate", get: (m) => m.success_rate, fmt: pct, score: "success_rate" },
  { key: "env_actions", label: "Env actions", get: (m) => m.env_moves, fmt: fmt, def: true },
  { key: "time", label: "Time", get: (m) => m.duration_s, fmt: dur, def: true },
  { key: "tool_calls", label: "Tool calls", get: (m) => m.n_tool_calls, fmt: fmt, def: true },
  { key: "flagged_calls", label: "Flagged calls", get: (m) => m.flagged_tool_calls, fmt: fmt, def: true },
  { key: "n_runs", label: "Number of runs", count: true },  // runs of this task (not aggregated)
  { key: "iterations", label: "Iterations", get: (m) => m.n_turns, fmt: fmt },
  { key: "submissions", label: "Submissions", get: (m) => m.n_submissions, fmt: fmt },
  { key: "output_tokens", label: "Output tokens", get: (m) => m.tokens && m.tokens.output, fmt: fmt },
  { key: "cache_read", label: "Cache tokens", get: (m) => m.tokens && m.tokens.cache_read, fmt: fmt },
  { key: "thinking_chars", label: "Thinking chars", get: (m) => m.thinking_chars, fmt: fmt },
];
const STATUS_METRIC = { key: "status", label: "Status", categorical: true };
const AGGREGATORS = {
  mean: (xs) => xs.reduce((a, b) => a + b, 0) / xs.length,
  median: (xs) => { const s = [...xs].sort((a, b) => a - b), i = s.length >> 1; return s.length % 2 ? s[i] : (s[i - 1] + s[i]) / 2; },
  min: (xs) => Math.min(...xs),
  max: (xs) => Math.max(...xs),
};
// A run "finished" iff it ended cleanly: it has an exit_reason and no error category. Crashed /
// unfinished runs (agent_api, eval_harness, loop_crash, still-running…) are dropped when masking.
const isValidRun = (m) => !m.last_error_category && !!m.exit_reason;
// Panel settings persist PER INTERFACE server-side (~/.regact via /api/settings, keyed by the graph
// `under` scope), so they survive sessions/browsers and viz updates. Checked-in per-model defaults
// (colors + order) live in static/model_defaults.json. Precedence: saved override > model default >
// built-in palette.
let _scope = "";                                   // current interface (the graph `under` path)
let _modelDefaults = { colors: {}, order_by: "model_name" };
let _defaultsLoaded = false;
let _saveTimer = null;
async function loadModelDefaults() {
  if (_defaultsLoaded) return;
  try { _modelDefaults = await api("static/model_defaults.json", { cache: "no-store" }); } catch (e) { if (e.name === "AbortError") throw e; /* keep empty */ }
  _defaultsLoaded = true;
}
async function loadSettings(scope) {
  try { return await api("/api/settings?scope=" + encodeURIComponent(scope)); } catch (e) { if (e.name === "AbortError") throw e; return {}; }
}
function saveSettings() {   // debounced PUT of the whole panel state for the current interface
  const body = {
    version: 1, agg: _graph.agg, err: _graph.err, mask: _graph.mask, barScale: _graph.barScale,
    order: _graph.order, colors: _graph.colors, curveX: _graph.curveX,
    hidden: [..._graph.hidden], active: [..._graph.active], known: _graph.known,
  };
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    fetch("/api/settings?scope=" + encodeURIComponent(_scope),
      { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .catch(() => { /* best-effort */ });
  }, 500);
}

// Experiment leaf -> model name, across both naming conventions:
//   alan-MiniMax-M2.7-Q8-fo   -> MiniMax-M2.7   (agent prefix + trailing -<quant>-<extra> stripped)
//   claude-opus_CWM / codex-gpt-5.5_arc -> opus / gpt-5.5  (agent prefix + trailing _<mode> stripped)
function modelName(exp) {
  let n = expLeaf(exp).replace(/^[^-]+-/, "");   // agent prefix (alan- / claude- / codex-)
  n = n.replace(/_[A-Za-z0-9]+$/, "");           // trailing _<mode> (_CWM / _arc), if any
  n = n.replace(/-(bf16|fp8|int\d+|Q\d+(_[A-Za-z0-9]+)?|IQ\d+(_[A-Za-z0-9]+)?)(-(fo|po))?$/i, "");
  return n.replace(/-(fo|po)$/i, "");
}

const _graph = {
  agg: "mean", err: "none", mask: false, barScale: 1,   // barScale: x-axis bar-width zoom (persisted)
  curveX: "env actions",   // x axis of the progress curves (a CURVE_X key)
  active: new Set(),  // set by applySettings: saved selection, else the Main metrics
  hidden: new Set(), colors: {}, order: [],
  known: [],   // metric keys offered when the selection was saved; a Main one added since is shown
};

// Apply a saved settings blob onto _graph for the current interface; a missing key falls back to the
// default (robust to viz updates, and resets cleanly when switching interfaces). `specs` seeds the
// default metric selection when the interface has none saved.
function applySettings(s, specs) {
  _graph.agg = s.agg || "mean";
  _graph.err = s.err || "none";
  _graph.mask = !!s.mask;
  _graph.barScale = typeof s.barScale === "number" ? s.barScale : 1;
  _graph.curveX = CURVE_X[s.curveX] ? s.curveX : "env actions";
  _graph.order = Array.isArray(s.order) ? s.order : [];
  _graph.colors = (s.colors && typeof s.colors === "object") ? s.colors : {};
  _graph.hidden = new Set(Array.isArray(s.hidden) ? s.hidden : []);
  _graph.known = specs.map((spec) => spec.key);
  if (Array.isArray(s.active)) {
    _graph.active = new Set(s.active);
    const known = new Set(Array.isArray(s.known) ? s.known : specs.filter((spec) => !spec.curve).map((spec) => spec.key));
    for (const spec of specs) if (spec.main && !known.has(spec.key)) _graph.active.add(spec.key);
  } else {   // no saved metric selection -> the Main metrics
    _graph.active = new Set(specs.filter((spec) => spec.main).map((spec) => spec.key));
  }
}

// Effective experiment order: the user's saved order if any, else the checked-in default (by model
// name, which groups families). Colors are looked up separately, so reordering never recolors.
function orderedExps(experiments) {
  if (_graph.order.length) {
    const pos = new Map(_graph.order.map((e, i) => [e, i]));
    return [...experiments].sort((a, b) => (pos.has(a) ? pos.get(a) : 1e9) - (pos.has(b) ? pos.get(b) : 1e9));
  }
  if (_modelDefaults.order_by === "model_name")
    return [...experiments].sort((a, b) => modelName(a).localeCompare(modelName(b)));
  return [...experiments];
}
function moveExp(exp, delta, allExps, redraw) {
  const cur = orderedExps(allExps);
  const i = cur.indexOf(exp), j = i + delta;
  if (j < 0 || j >= cur.length) return;
  [cur[i], cur[j]] = [cur[j], cur[i]];
  _graph.order = cur; redraw();   // redraw() persists (saveSettings runs inside it)
}

const expLeaf = (e) => String(e).split("/").pop();
const _AVG_COL = " avg";  // synthetic trailing "Averaged" column key (never collides with a real task)
const shortTask = (t) => (t === _AVG_COL ? "Averaged" : String(t).replace(/^MiniGrid-/, "").replace(/-v\d+$/, ""));
const statusColor = (s) => (s === "agent_exit" ? _THEME.good : /limit/.test(s) ? _THEME.warn : s === "running" ? _THEME.muted : _THEME.bad);
const txt = (s) => document.createTextNode(s);
function svg(tag, attrs, ...kids) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  for (const k of kids) if (k) e.append(k);
  return e;
}

function metricSpecs(games) {
  // A score metric no run reports (success_rate on ARC) is not offered at all.
  const specs = FRAMEWORK_METRICS
    .filter((s) => !s.score || games.some((g) => s.get(g.metrics) != null))
    .map((s) => ({ ...s }));  // copies: `main` below depends on these games
  const seen = new Set();
  for (const g of games)
    for (const [k, v] of Object.entries(g.metrics.final_aggregate || {}))
      if (typeof v === "number" && !_AGG_SKIP.has(k) && !seen.has(k)) {
        seen.add(k);
        specs.push({ key: "agg:" + k, label: k, score: k, get: (m) => m.final_aggregate && m.final_aggregate[k] });
      }
  // Per-feature submission metrics (e.g. cwm.n_conflicting_transitions), same dynamic treatment as
  // the game aggregate: each numeric key becomes a graphable metric, no per-feature viz code.
  const seenFeat = new Set();
  for (const g of games)
    for (const [feat, fm] of Object.entries(g.metrics.feature_metrics || {}))
      for (const [k, v] of Object.entries(fm || {}))
        if (typeof v === "number" && !seenFeat.has(feat + "." + k)) {
          seenFeat.add(feat + "." + k);
          specs.push({ key: "feat:" + feat + "." + k, label: feat + "." + k,
            get: (m) => m.feature_metrics && m.feature_metrics[feat] && m.feature_metrics[feat][k] });
        }
  // Problem-derived metrics (ARC RHAE, LRHAE-Uncapped).
  const seenDrv = new Set();
  for (const g of games)
    for (const [k, v] of Object.entries(g.metrics.derived_metrics || {}))
      if (typeof v === "number" && !seenDrv.has(k)) {
        seenDrv.add(k);
        specs.push({ key: "derived:" + k, label: k.replace(/_/g, "-").toUpperCase(), fmt: pct1, score: k,
          get: (m) => m.derived_metrics && m.derived_metrics[k] });
      }
  // The problem's progress (ARC: levels completed) as a curve over what each run spent.
  const prog = games.map((g) => g.metrics.progress).find((p) => p && p.label);
  if (prog) specs.push({ key: "progress", label: prog.label + " over the run", curve: true, def: true });
  specs.push(STATUS_METRIC);
  const mains = new Set(games.flatMap((g) => g.metrics.main_metrics || []));
  for (const s of specs) s.main = !!(s.def || (s.score && mains.has(s.score)));
  return specs;
}

function groupByExpTask(games) {
  const byExp = new Map();          // experiment -> Map(task -> [metrics of each run])
  const tasks = new Set();
  for (const g of games) {
    tasks.add(g.task);
    if (!byExp.has(g.experiment)) byExp.set(g.experiment, new Map());
    const t = byExp.get(g.experiment);
    if (!t.has(g.task)) t.set(g.task, []);
    t.get(g.task).push(g.metrics);
  }
  return { experiments: [...byExp.keys()], tasks: [...tasks].sort(), byExp };
}

function makeExpColor(experiments) {
  const idx = new Map(experiments.map((e, i) => [e, i]));  // natural index -> palette fallback
  // saved user override > checked-in per-model default (model_defaults.json) > built-in palette.
  return (e) => _graph.colors[e] || _modelDefaults.colors[modelName(e)] || (_modelDefaults.family_colors || {})[familyOf(modelName(e))] || _EXP_PALETTE[(idx.get(e) || 0) % _EXP_PALETTE.length];
}

// Padding computed from the longest x-label so end-anchored, -35deg labels never clip either edge.
function _chartPad(tasks) {
  const reach = Math.max(6, ...tasks.map((t) => shortTask(t).length)) * 6.2;  // ~px of the widest label
  return {
    padT: 10, padR: 16, H: 210,
    padL: Math.max(58, Math.round(reach * 0.82) + 8),   // y-labels + leftmost angled label (cos35)
    padB: Math.max(84, Math.round(reach * 0.57) + 24),  // angled label vertical drop (sin35)
  };
}

// Task previews are pre-rendered PNGs under static/icons_tasks/ (scripts/gen_task_previews.py, run
// offline) so the viz never builds an env while serving. Filename = task sanitized to match the script.
const taskIconSrc = (task) => "static/icons_tasks/" + String(task).replace(/[^A-Za-z0-9._-]/g, "_") + ".png";

// A small task-preview thumbnail under each x-axis label, so a task is recognizable at a glance.
// `centerXOf(ti)` is the task's group center; the image removes itself if no preview exists.
const _THUMB_GAP = 6;
function appendTaskThumbs(s, tasks, centerXOf, yTop, size) {
  tasks.forEach((task, ti) => {
    if (task === _AVG_COL) return;   // the "Averaged" column has no task to preview
    const img = svg("image", {
      x: centerXOf(ti) - size / 2, y: yTop, width: size, height: size,
      href: taskIconSrc(task),
      preserveAspectRatio: "xMidYMid meet",
    });
    img.addEventListener("error", () => img.remove());
    img.append(svg("title", {}, txt(task)));
    s.append(img);
  });
}

// A grouped bar chart: x = task, one bar per experiment = agg over that (exp,task)'s runs. `mask`
// drops crashed/unfinished runs first; `errMethod` adds error/interval bars (std around the MEAN,
// or the min-max range) - the interval is independent of which aggregate sets the bar height.
function groupedBarChart(spec, group, expColor, aggName, errMethod, mask) {
  const { experiments, byExp } = group;
  const agg = AGGREGATORS[aggName];
  const runsOf = (exp, task) => {
    const runs = (byExp.get(exp) && byExp.get(exp).get(task)) || [];
    return mask ? runs.filter(isValidRun) : runs;
  };
  const seriesOf = (exp, task) =>
    runsOf(exp, task).map((m) => spec.get(m)).filter((x) => x != null && !Number.isNaN(Number(x))).map(Number);
  const val = (exp, task) => {
    if (spec.count) return runsOf(exp, task).length || null;   // "number of runs": a count
    const xs = seriesOf(exp, task);
    return xs.length ? agg(xs) : null;
  };
  const errOf = (exp, task) => {                 // {lo, hi} interval, or null (needs >= 2 runs)
    if (errMethod === "none" || spec.count) return null;
    const xs = seriesOf(exp, task);
    if (xs.length < 2) return null;
    if (errMethod === "range") return { lo: Math.min(...xs), hi: Math.max(...xs) };
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;                            // std: centered
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1)); // on the mean
    return { lo: mean - sd, hi: mean + sd };
  };
  // Only keep tasks with at least one value: with the mask on, a task whose runs are all
  // crashed/unfinished (or that never reported this metric) gets no phantom x-axis column.
  const tasks = group.tasks.filter((task) => experiments.some((exp) => val(exp, task) != null));
  // A trailing "Averaged" column (only with >= 2 tasks): per experiment, the mean of the metric over
  // the tasks that HAVE a value - so absent tasks (and, with the mask on, all-crashed tasks, which
  // are already dropped from `tasks`) never count toward the mean.
  const avgOf = (exp) => {
    const vs = tasks.map((t) => val(exp, t)).filter((v) => v != null);
    return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null;
  };
  const cols = tasks.length >= 2 ? [...tasks, _AVG_COL] : tasks;
  const colVal = (exp, col) => (col === _AVG_COL ? avgOf(exp) : val(exp, col));
  const colErr = (exp, col) => (col === _AVG_COL ? null : errOf(exp, col));  // no interval on the mean
  let max = 0;
  for (const t of tasks) for (const e of experiments) {   // a mean <= max(tasks), so tasks bound the axis
    const v = val(e, t); if (v != null) max = Math.max(max, v);
    const eb = errOf(e, t); if (eb) max = Math.max(max, eb.hi);   // keep error bars in view
  }
  max = max || 1;
  const { padL, padB, padR, padT, H } = _chartPad(cols);
  const slot = Math.max(2, 20 * _graph.barScale);   // per-experiment bar width (bar-width slider)
  const groupW = experiments.length * slot + 16;    // + a fixed 16px inter-task gap (spacing kept)
  const W = padL + padR + cols.length * groupW;
  const plotH = H - padT - padB;
  const yOf = (v) => padT + plotH * (1 - Math.max(0, Math.min(1, v / max)));   // value -> y, clamped
  const thumb = Math.min(48, groupW - 6);          // task-preview size; fits inside one group slot
  const yThumb = H + _THUMB_GAP;                    // BELOW the angled label band (labels fill padB)
  const Hsvg = yThumb + thumb + 4;                  // extend the canvas for the thumbnail row
  const s = svg("svg", { class: "chart", width: W, height: Hsvg, viewBox: `0 0 ${W} ${Hsvg}` });
  for (const f of [0, 0.5, 1]) {                 // y gridlines + labels
    const y = padT + plotH * (1 - f);
    s.append(svg("line", { class: "gridline", x1: padL, y1: y, x2: W - padR, y2: y }));
    s.append(svg("text", { class: "ylab", x: padL - 6, y: y + 3, "text-anchor": "end" },
      txt(spec.fmt ? spec.fmt(max * f) : fmtMetric(max * f))));
  }
  cols.forEach((col, ci) => {
    const gx = padL + ci * groupW + 8;
    const bw = (groupW - 16) / experiments.length;   // one slot per experiment; bars fill it (no gap)
    const base = padT + plotH;                       // y of the zero baseline
    if (ci > 0) {                                     // thin separator between adjacent task groups
      const sepX = padL + ci * groupW;
      s.append(svg("line", { class: "gridline", x1: sepX, y1: padT, x2: sepX, y2: base }));
    }
    const colName = col === _AVG_COL ? "Averaged" : col;
    experiments.forEach((exp, ei) => {
      const v = colVal(exp, col);
      if (v == null) return;
      const x = gx + ei * bw, cx = x + bw / 2;        // bars packed edge-to-edge; cx = this bar's center
      const barTop = Math.min(yOf(v), base - 2);      // a value of 0 still shows a >=2px line
      const rect = svg("rect", { x, y: barTop, width: bw, height: base - barTop, fill: expColor(exp) });
      rect.append(svg("title", {}, txt(`${expLeaf(exp)} · ${colName}\n${spec.label}: ${spec.fmt ? spec.fmt(v) : fmtMetric(v)}`)));
      s.append(rect);
      const eb = colErr(exp, col);               // error/interval bar centered on the bar
      if (eb) {
        const yl = yOf(eb.lo), yh = yOf(eb.hi), cap = Math.max(2, bw * 0.3);
        s.append(svg("line", { class: "errbar", x1: cx, y1: yh, x2: cx, y2: yl }));
        s.append(svg("line", { class: "errbar", x1: cx - cap, y1: yh, x2: cx + cap, y2: yh }));
        s.append(svg("line", { class: "errbar", x1: cx - cap, y1: yl, x2: cx + cap, y2: yl }));
      }
    });
    const lx = gx + (groupW - 16) / 2, ly = H - padB + 12;   // x label (task), rotated; bold for the mean
    s.append(svg("text", { class: "xlab" + (col === _AVG_COL ? " avglab" : ""), x: lx, y: ly, "text-anchor": "end", transform: `rotate(-35 ${lx} ${ly})` }, txt(shortTask(col))));
  });
  appendTaskThumbs(s, cols, (ci) => padL + ci * groupW + 8 + (groupW - 16) / 2, yThumb, thumb);
  return s;
}

// Progress curves. Per run: the problem's progress as a step function of what the run had spent
// (`point` = the field of a progress point, `end` = where the run stopped). A finished run keeps its
// last value to the right edge; an unfinished one (crashed, killed, still running) stops counting
// after its last point, and is dropped altogether when the mask is on.
const CURVE_X = {
  "env actions": { point: "env_actions", end: (m) => m.env_moves, fmt: fmt },
  "tool calls": { point: "tool_calls", end: (m) => m.n_tool_calls, fmt: fmt },
  "time": { point: "seconds", end: (m) => m.duration_s, fmt: dur },
};
function curveRuns(exp, task, group, mask, X) {
  const runs = [];
  for (const m of (group.byExp.get(exp) && group.byExp.get(exp).get(task)) || []) {
    const done = isValidRun(m);
    if (!m.progress || (mask && !done)) continue;
    const pts = m.progress.points.map((q) => [q[X.point], q.value]);
    if (pts.some((q) => q[0] == null)) continue;   // no such measure recorded for this run
    runs.push({ pts, done, end: Math.max(Number(X.end(m)) || 0, ...pts.map((q) => q[0])) });
  }
  return runs;
}
const curveAt = (run, x) => { let v = 0; for (const [px, pv] of run.pts) if (px <= x) v = pv; return v; };
function progressChart(task, group, expColor, aggName, errMethod, mask, xName) {
  const X = CURVE_X[xName], agg = AGGREGATORS[aggName];
  const series = group.experiments.map((exp) => ({ exp, runs: curveRuns(exp, task, group, mask, X) }))
    .filter((sr) => sr.runs.length);
  if (!series.length) return null;
  const all = series.flatMap((sr) => sr.runs);
  const ref = xName === "env actions"
    ? ((((group.byExp.get(series[0].exp) || new Map()).get(task) || []).find((m) => m.progress) || {}).progress || {}).reference || []
    : [];
  // The axis stops a little after the last progress any run made: beyond it every line is flat, and
  // one run that kept acting without progress would otherwise squeeze all the others to the left.
  const xMax = Math.max(1, ...all.flatMap((r) => r.pts.map((q) => q[0]))) * 1.15;
  const yMax = Math.max(1, ...all.flatMap((r) => r.pts.map((q) => q[1])), ...ref.map((q) => q[1]));
  const grid = [...new Set([0, xMax, ...all.flatMap((r) => [r.end, ...r.pts.map((q) => q[0])])])]
    .filter((x) => x <= xMax).sort((a, b) => a - b);
  const padL = 44, padR = 16, padT = 10, padB = 30, W = 620, H = 270;
  const xOf = (x) => padL + (W - padL - padR) * Math.min(1, x / xMax);
  const yOf = (v) => padT + (H - padT - padB) * (1 - Math.max(0, Math.min(1, v / yMax)));
  const s = svg("svg", { class: "chart", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  for (let v = 0; v <= yMax; v++) {
    s.append(svg("line", { class: "gridline", x1: padL, y1: yOf(v), x2: W - padR, y2: yOf(v) }));
    s.append(svg("text", { class: "ylab", x: padL - 6, y: yOf(v) + 3, "text-anchor": "end" }, txt(String(v))));
  }
  for (const f of [0, 0.25, 0.5, 0.75, 1])
    s.append(svg("text", { class: "xlab", x: xOf(xMax * f), y: H - padB + 16, "text-anchor": "middle" }, txt(X.fmt(Math.round(xMax * f)))));
  const steps = (ys) => {   // step-after path through (grid[i], ys[i]); null = no run there
    let d = "", pen = false;
    grid.forEach((x, i) => {
      if (ys[i] == null) { pen = false; return; }
      d += pen ? ` V${yOf(ys[i])}` : ` M${xOf(x)},${yOf(ys[i])}`;
      pen = true;
      if (i + 1 < grid.length) d += ` H${xOf(grid[i + 1])}`;
    });
    return d;
  };
  if (ref.length) {
    const path = svg("path", { class: "curve-ref", d: steps(grid.map((x) => curveAt({ pts: ref }, x))) });
    path.append(svg("title", {}, txt("reference player")));
    s.append(path);
  }
  for (const { exp, runs } of series) {
    const at = grid.map((x) => runs.filter((r) => r.done || x <= r.end).map((r) => curveAt(r, x)));
    const color = expColor(exp);
    if (errMethod !== "none") {
      const band = at.map((xs) => {
        if (xs.length < 2) return null;
        if (errMethod === "range") return [Math.min(...xs), Math.max(...xs)];
        const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
        const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1));
        return [mean - sd, mean + sd];
      });
      grid.forEach((x, i) => {
        if (!band[i] || i + 1 >= grid.length) return;
        const yh = yOf(band[i][1]), yl = yOf(band[i][0]);
        s.append(svg("rect", { class: "curve-band", x: xOf(x), y: yh, width: xOf(grid[i + 1]) - xOf(x), height: yl - yh, fill: color }));
      });
    }
    const line = svg("path", { class: "curve-line", stroke: color, d: steps(at.map((xs) => (xs.length ? agg(xs) : null))) });
    line.append(svg("title", {}, txt(`${expLeaf(exp)} · ${task}\n${runs.length} run${runs.length > 1 ? "s" : ""}`)));
    s.append(line);
  }
  return { chart: s, hasRef: ref.length > 0 };
}

// Status is categorical: per (task, experiment) a full-height bar stacked by exit-reason share,
// colored by status; a thin underline ties each bar back to its experiment color.
function statusChart(group, expColor) {
  const { experiments, tasks, byExp } = group;
  const statusOfM = (m) => m.exit_reason || m.last_error_category || "running";
  const cats = new Set();
  for (const e of experiments) for (const t of tasks) for (const m of (byExp.get(e) && byExp.get(e).get(t)) || []) cats.add(statusOfM(m));
  const { padL, padB, padR, padT, H } = _chartPad(tasks);
  const slot = Math.max(2, 20 * _graph.barScale);   // bar-width slider (same as the metric charts)
  const groupW = experiments.length * slot + 16;
  const W = padL + padR + tasks.length * groupW, plotH = H - padT - padB;
  const thumb = Math.min(48, groupW - 6);
  const yThumb = H + _THUMB_GAP;                    // BELOW the angled label band (labels fill padB)
  const Hsvg = yThumb + thumb + 4;
  const s = svg("svg", { class: "chart", width: W, height: Hsvg, viewBox: `0 0 ${W} ${Hsvg}` });
  s.append(svg("line", { class: "gridline", x1: padL, y1: padT, x2: W - padR, y2: padT }));
  s.append(svg("line", { class: "gridline", x1: padL, y1: padT + plotH, x2: W - padR, y2: padT + plotH }));
  tasks.forEach((task, ti) => {
    const gx = padL + ti * groupW + 8, bw = (groupW - 16) / experiments.length;
    experiments.forEach((exp, ei) => {
      const runs = (byExp.get(exp) && byExp.get(exp).get(task)) || [];
      const x = gx + ei * bw;
      if (!runs.length) return;
      const counts = {};
      for (const m of runs) { const st = statusOfM(m); counts[st] = (counts[st] || 0) + 1; }
      let acc = 0;
      for (const [st, c] of Object.entries(counts)) {
        const frac = c / runs.length, segH = plotH * frac, y = padT + plotH - acc - segH;
        const rect = svg("rect", { x: x + 1, y, width: Math.max(1, bw - 2), height: segH, fill: statusColor(st) });
        rect.append(svg("title", {}, txt(`${expLeaf(exp)} · ${task}\n${st}: ${c}/${runs.length}`)));
        s.append(rect); acc += segH;
      }
      s.append(svg("rect", { x: x + 1, y: padT + plotH + 2, width: Math.max(1, bw - 2), height: 3, fill: expColor(exp) }));
    });
    const lx = gx + (groupW - 16) / 2, ly = H - padB + 12;
    s.append(svg("text", { class: "xlab", x: lx, y: ly, "text-anchor": "end", transform: `rotate(-35 ${lx} ${ly})` }, txt(shortTask(task))));
  });
  appendTaskThumbs(s, tasks, (ti) => padL + ti * groupW + 8 + (groupW - 16) / 2, yThumb, thumb);
  return { chart: s, cats: [...cats] };
}

function legend(items) {   // items: [{label, color}] — the static status legend
  const l = h("div", "legend");
  for (const it of items) l.append(h("span", "leg", swatch(it.color), " " + it.label));
  return l;
}
function swatch(color) { const s = h("span", "sw"); s.style.background = color; return s; }

// The Experiments panel (sidebar, below Metrics). Each row: up/down to reorder (= reorder the bars
// left-to-right), a swatch to recolor (persisted), and the name to show/hide. `experiments` is the
// display order; `allExps` is the full natural-order set the reorder operates over.
function expSection(experiments, expColor, redraw, allExps) {
  const panel = h("div", "controls", h("div", "h", "Experiments"));
  experiments.forEach((exp, i) => {
    const row = h("div", "exprow" + (_graph.hidden.has(exp) ? " off" : ""));
    const up = h("button", "ordbtn", "\u2191"); up.title = "move up / left"; up.disabled = i === 0;
    up.onclick = () => moveExp(exp, -1, allExps, redraw);
    const down = h("button", "ordbtn", "\u2193"); down.title = "move down / right";
    down.disabled = i === experiments.length - 1;
    down.onclick = () => moveExp(exp, +1, allExps, redraw);
    const pick = h("input"); pick.type = "color"; pick.className = "sw-pick";
    pick.value = expColor(exp); pick.title = "pick colour";
    pick.oninput = () => { _graph.colors[exp] = pick.value; redraw(); };
    const label = h("span", "leg-label", expLeaf(exp)); label.title = "click to show / hide";
    label.onclick = () => { _graph.hidden.has(exp) ? _graph.hidden.delete(exp) : _graph.hidden.add(exp); redraw(); };
    row.append(up, down, pick, label);
    panel.append(row);
  });
  return panel;
}

// A segmented single-choice control; buttons update their own "on" state on click (the chart
// redraw does not rebuild them), then run onPick.
function segBtns(label, options, current, onPick) {
  const seg = h("span", "seg", h("span", "muted", label));
  const btns = [];
  for (const name of options) {
    const b = h("button", "aggbtn" + (name === current ? " on" : ""), name);
    b.onclick = () => { for (const x of btns) x.classList.toggle("on", x === b); onPick(name); };
    btns.push(b);
    seg.append(b);
  }
  return seg;
}
function controlBar(redraw) {
  const agg = segBtns("aggregate:", Object.keys(AGGREGATORS), _graph.agg, (n) => { _graph.agg = n; redraw(); });
  const err = segBtns("error bars:", ["none", "std", "range"], _graph.err, (n) => { _graph.err = n; redraw(); });
  const cb = h("input"); cb.type = "checkbox"; cb.id = "mask-crashed"; cb.checked = _graph.mask;
  cb.onchange = () => { _graph.mask = cb.checked; redraw(); };
  const mask = h("label", "maskctl"); mask.htmlFor = "mask-crashed";
  mask.append(cb, " mask crashed / unfinished runs");
  // Bar-width zoom: narrower bars fit more tasks on screen at once (persisted per interface).
  const sl = h("input"); sl.type = "range"; sl.min = "0.15"; sl.max = "2"; sl.step = "0.05";
  sl.value = String(_graph.barScale); sl.title = "bar width";
  sl.oninput = () => { _graph.barScale = Number(sl.value); redraw(); };
  const width = h("label", "widthctl", h("span", "muted", "bar width"), sl);
  return h("div", "controlbar", agg, err, mask, width);
}
function metricControls(specs, redraw) {
  const panel = h("div", "controls", h("div", "h", "Metrics"));
  for (const spec of specs) {
    const id = "m-" + spec.key;
    const cb = h("input"); cb.type = "checkbox"; cb.id = id; cb.checked = _graph.active.has(spec.key);
    cb.onchange = () => { cb.checked ? _graph.active.add(spec.key) : _graph.active.delete(spec.key); redraw(); };
    const row = h("label", "ctl"); row.htmlFor = id; row.append(cb, " " + spec.label);
    panel.append(row);
  }
  return panel;
}

// Build the panel: controls + Experiments sidebar (left) + charts (right). `onlyExp` restricts to
// one experiment (the per-game tab); null = every experiment. `scope` keys this interface's saved
// settings (colors/order/toggles) under ~/.regact. Async: it loads the checked-in defaults + the
// interface's saved settings before the first render.
async function graphsView(games, onlyExp, scope) {
  _scope = scope || "";
  await loadModelDefaults();
  const shown = onlyExp ? games.filter((g) => g.experiment === onlyExp) : games;
  const specs = metricSpecs(shown);
  applySettings(await loadSettings(_scope), specs);
  const group = groupByExpTask(shown);
  const expColor = makeExpColor(group.experiments);   // saved > per-model default > palette
  const charts = h("div", "charts");
  const expBox = h("div");   // the Experiments panel (sidebar, below Metrics); rebuilt on redraw
  // redraw() re-renders AND persists (debounced) so any control change is saved for this interface;
  // the initial render passes save=false so merely opening the panel doesn't rewrite the file.
  const redraw = (save = true) => {
    const displayExps = orderedExps(group.experiments);   // user order = left-to-right bar order
    clear(expBox);
    expBox.append(expSection(displayExps, expColor, redraw, group.experiments));
    clear(charts);
    const activeExps = displayExps.filter((e) => !_graph.hidden.has(e));  // hidden ones drop out
    const g2 = { experiments: activeExps, tasks: group.tasks, byExp: group.byExp };
    let any = false;
    for (const spec of specs) {
      if (!_graph.active.has(spec.key)) continue;
      any = true;
      const card = h("div", "chartcard", h("h3", null, spec.label));
      const scroll = h("div", "chartscroll");
      if (spec.curve) {
        card.append(segBtns("x:", Object.keys(CURVE_X), _graph.curveX, (n) => { _graph.curveX = n; redraw(); }));
        let hasRef = false;
        for (const task of g2.tasks) {
          const drawn = progressChart(task, g2, expColor, _graph.agg, _graph.err, _graph.mask, _graph.curveX);
          if (!drawn) continue;
          hasRef = hasRef || drawn.hasRef;
          if (g2.tasks.length > 1) scroll.append(h("div", "muted", shortTask(task)));
          scroll.append(drawn.chart);
        }
        card.append(scroll);
        if (hasRef) card.append(h("div", "muted curve-note", "dashed: reference player (ARC: the human baseline)"));
      } else if (spec.categorical) {
        const { chart, cats } = statusChart(g2, expColor);
        scroll.append(chart);
        card.append(scroll, legend(cats.map((c) => ({ label: c, color: statusColor(c) }))));
      } else {
        scroll.append(groupedBarChart(spec, g2, expColor, _graph.agg, _graph.err, _graph.mask));
        card.append(scroll);
      }
      charts.append(card);
    }
    if (!any) charts.append(h("div", "muted", "no metric selected - pick some on the left"));
    if (save) saveSettings();
  };
  const sidebar = h("div", "gsidebar", metricControls(specs, redraw), expBox);
  const layout = h("div", "graphlayout", sidebar, charts);
  const wrap = h("div", "graphs", controlBar(redraw), layout);
  redraw(false);   // first paint: render without persisting
  return wrap;
}

async function renderGraphs(under = "") {
  crumb.textContent = "graphs";
  const data = await gamesData(under);
  const token = navigation;
  const body = await graphsView(data.games, null, under);
  checkNavigation(token);
  clear(app);
  app.append(panelNav("graphs", under), body);
}

// The per-game "Graphs" tab: the same charts but for THIS run's experiment only (scoped to the
// experiment subtree - the game path minus its /timestamp/task - so it never parses the whole root).
async function renderGameGraphs(name) {
  const token = navigation;
  const data = await gamesData(name.split("/").slice(0, -2).join("/"));
  const me = data.games.find((g) => g.name === name);
  const exp = me ? me.experiment : null;
  const body = h("div");
  body.append(h("div", "gnote muted",
    exp ? `Aggregated over every run in this experiment (${expLeaf(exp)}).` : "unknown experiment"));
  body.append(await graphsView(data.games, exp, name.split("/").slice(0, -2).join("/")));
  checkNavigation(token);
  shell(name, "graphs", body);
}

// ---------------------------------------------------------------- per-game shell
const TABS = [["", "Overview"], ["conversation", "Conversation"], ["artifacts", "Artifacts"], ["logs", "Logs"], ["graphs", "Graphs"]];

function shell(name, active, body) {
  crumb.textContent = name;
  const nav = h("div", "tabs");
  // Back to this game's EXPERIMENT dashboard (skip the run/timestamp level), so navigation is
  // benchmark -> experiment -> task, never stopping at a single-run page.
  const parent = name.split("/").slice(0, -2).join("/");
  const back = h("a", "tab", "< back");
  back.href = parent ? "#run/" + encodeURIComponent(parent) : "#";
  nav.append(back);
  for (const [slug, label] of TABS) {
    const href = "#game/" + encodeURIComponent(name) + (slug ? "/" + slug : "");
    const a = h("a", "tab" + (slug === active ? " on" : ""), label);
    a.href = href;
    nav.append(a);
  }
  clear(app); app.append(nav, body);
}

// ---------------------------------------------------------------- overview tab
// Metrics render as a compact 2-column table (label | value), like Run config - simpler and more
// legible than cards, and long values (the aggregate score) wrap cleanly. A value may carry a muted
// qualifier (e.g. "shadow-replay verified"). Rows are [label, value, sub?]; a null row is skipped.
function metricTable(title, rows) {
  const sec = h("div", "kpisection");
  sec.append(h("div", "kpih", title));
  const t = h("table", "metrics");
  for (const row of rows) {
    if (!row) continue;
    const [label, value, sub] = row;
    const val = h("td", null, String(value));
    if (sub) val.append(h("span", "muted", " · " + sub));
    t.append(h("tr", null, h("td", "mlabel", label), val));
  }
  sec.append(t);
  return sec;
}

// --- "who fights who": [agent] + [model, sized by params] -> [task png | green problem frame] -----
let _icons = { sigmoid: {}, agents: {}, families: {}, models: {} };
let _iconsLoaded = false;
async function loadIcons() {
  if (_iconsLoaded) return;
  // no-store: the registry is edited live (icons/params added by hand); never serve a stale cached copy.
  try { _icons = await api("static/viz_icons.json", { cache: "no-store" }); } catch (e) { if (e.name === "AbortError") throw e; /* keep empty (badge fallback) */ }
  _iconsLoaded = true;
}
// Identity comes from saved configuration, never from the user-chosen experiment label.
function agentIdentity(agent) {
  return { name: agent?.name || "unknown", model: agent?.model || "unknown" };
}
function modelLabel(model) {
  // Provider prefixes and quantization remain available in the full identifier tooltip.
  return String(model).split("/").pop()
    .replace(/-(bf16|fp8|int\d+|Q\d+(_[A-Za-z0-9]+)?|IQ\d+(_[A-Za-z0-9]+)?)$/i, "");
}
function modelRegistration(model) {
  return _icons.models[model] || _icons.models[modelLabel(model)] || {};
}
function familyOf(model) {   // Qwen3-235B -> Qwen; claude-sonnet-5 -> sonnet (a registered family in the name)
  const lead = (String(model).match(/^[A-Za-z]+/) || [""])[0];
  const families = Object.keys(_icons.families || {});
  if (families.includes(lead)) return lead;
  const tokens = modelLabel(model).toLowerCase().split(/[^a-z]+/);
  return families.find((f) => tokens.includes(f.toLowerCase())) || lead;
}
function modelParamsInfo(model) {   // {b, estimate}: registry params_b (may be flagged estimate), else parsed
  const reg = modelRegistration(model);
  if (reg && typeof reg.params_b === "number") return { b: reg.params_b, estimate: !!reg.estimate };
  const m = String(model).match(/(\d+(?:\.\d+)?)B\b/);  // first "NNN B" in the name = total params (exact)
  return { b: m ? parseFloat(m[1]) : null, estimate: false };
}
function modelSizePx(paramsB) {                          // sigmoidal icon size vs param count
  const c = _icons.sigmoid || {};
  const lo = c.min_px ?? 22, hi = c.max_px ?? 60, mid = c.midpoint_b ?? 100, k = c.steepness ?? 1.6;
  if (!paramsB) return Math.round((lo + hi) / 2);       // unknown params -> mid size
  const s = 1 / (1 + Math.exp(-(Math.log10(paramsB) - Math.log10(mid)) * k));
  return Math.round(lo + (hi - lo) * s);
}
const _badgeText = (name) => String(name || "?").replace(/[^A-Za-z0-9]/g, "").slice(0, 2).toUpperCase();
function iconBadge(name, size, text = _badgeText(name)) {   // registry may supply a badge label
  const b = h("span", "ibadge", text);
  b.style.width = b.style.height = size + "px";
  b.style.lineHeight = size + "px"; b.style.fontSize = Math.round(size * 0.42) + "px";
  return b;
}
function iconImg(src, size, alt) {
  const img = h("img", "vicon"); img.src = src; img.width = img.height = size; img.alt = alt || "";
  // Appearance belongs to the asset registry, independently of agent/model identity.
  const style = _icons.icon_styles?.[src] || {};
  if (style.background) img.style.backgroundColor = style.background;
  if (style.padding_px != null) {
    img.style.padding = style.padding_px + "px";
    img.style.boxSizing = "border-box";  // preserve the requested outer icon size
  }
  if (style.object_fit) img.style.objectFit = style.object_fit;
  img.addEventListener("error", () => img.replaceWith(iconBadge(alt, size)));   // missing file -> badge
  return img;
}
function agentIcon(agent, size) {
  const src = _icons.agents[agent];
  return src ? iconImg(src, size, agent) : iconBadge(agent, size);
}
function agentBlock(agent) {   // icon + name below, mirroring modelBlock (the "who" writing the policy)
  const box = h("div", "agentblock");
  box.append(agentIcon(agent, 40), h("div", "aname", agent));
  return box;
}
function modelBlock(identifier) {
  const model = modelLabel(identifier), reg = modelRegistration(identifier);
  const { b: pB, estimate } = modelParamsInfo(identifier);
  const size = modelSizePx(pB);
  const src = reg.icon || _icons.families[familyOf(model)];
  const box = h("div", "modelblock");
  box.title = identifier;
  box.append(src ? iconImg(src, size, model) : iconBadge(model, size, reg.badge), h("div", "mname", model));
  // params below the name; "~" marks a sibling-inferred estimate (see viz_icons.json), no tilde = exact.
  if (pB) box.append(h("div", "mparams", (estimate ? "~" : "") + (pB >= 1000 ? pB / 1000 + "T" : pB + "B")));
  return box;
}
function longArrow() {
  const s = svg("svg", { class: "matcharrow", width: 56, height: 18, viewBox: "0 0 56 18" });
  s.append(svg("line", { x1: 2, y1: 9, x2: 46, y2: 9 }), svg("polygon", { points: "46,3 56,9 46,15" }));
  return s;
}
function taskThumb(task, size) {
  const img = h("img", "taskthumb"); img.src = taskIconSrc(task);
  img.width = img.height = size; img.alt = task; img.title = task;
  img.addEventListener("error", () => img.replaceWith(iconBadge(shortTask(task), size)));
  return img;
}
// A problem = a set of tasks: its task previews in a green frame, marking "the whole env" vs one task.
function problemThumbs(tasks, size) {
  const box = h("div", "problembox");
  for (const t of tasks) box.append(taskThumb(t, size));
  return box;
}
// The contestant: agent (who writes the policy) + its model (sized by params). No task -> used on the
// experiment/benchmark headers, which span many tasks; matchup() adds the arrow + task for one game.
function agentVsModel(configuredAgent) {
  const agent = agentIdentity(configuredAgent);
  return h("div", "avm", agentBlock(agent.name), h("span", "mplus", "+"), modelBlock(agent.model));
}
// agent + model (sized by params) -> right (a single task thumb, or a green problem frame).
function matchup(agent, right) {
  return h("div", "matchup", agentVsModel(agent), longArrow(), right);
}

async function renderOverview(name) {
  const token = navigation;
  const d = await gameDetail(name);
  await loadIcons();
  checkNavigation(token);
  const m = d.metrics;
  const wrap = h("div");
  const task = d.state.task_name || name.split("/").pop();
  wrap.append(matchup(d.config.agent, taskThumb(task, 56)));
  const ptasks = (d.config.problem && d.config.problem.tasks) || null;
  if (Array.isArray(ptasks) && ptasks.length > 1)   // the whole problem (its tasks) in a green frame
    wrap.append(h("div", "gnote muted", "Problem:"), problemThumbs(ptasks, 40));
  const unverified = m.final_aggregate_unverified || {};
  const hasUnverified = Object.keys(unverified).length > 0;
  // Same registry as the Graphs panel -> identical names/order/case; the "Score" is split into the
  // game's aggregate keys. Main = the score (agg keys) + the def-flagged effort/cost; Other = rest.
  const specs = metricSpecs([{ metrics: m }]).filter((s) => s.get && !s.count && !s.categorical);
  const fmtOf = (s) => { const v = s.get(m); return v != null && s.fmt ? s.fmt(v) : fmtMetric(v); };
  const isMain = (s) => s.main;
  const main = [["Status", statusOf(m)], ...specs.filter(isMain).map((s) => [s.label, fmtOf(s)])];
  const other = [
    ...specs.filter((s) => !isMain(s)).map((s) => [s.label, fmtOf(s)]),
    hasUnverified ? ["Score (no replay)", aggLine(unverified), "controller-reported"] : null,
  ];
  wrap.append(
    metricTable("Main metrics", main),
    metricTable("Other metrics", other),
    configBlock(d.config),
    barChart("Tool calls", m.tool_histogram));
  if (m.submission_trajectory.length) wrap.append(trajectory(m.submission_trajectory));
  if (m.flagged_calls && m.flagged_calls.length) wrap.append(flaggedPanel(m.flagged_calls));
  shell(name, "", wrap);
}

// Exhaustive + grouped: one table per top-level config section (problem, agent, controller,
// features, ...) plus a "parameters" table for the scalar base fields - same look as the metric
// tables. Nested objects flatten to dotted keys (so `features.myfeature.*` shows, never "[object Object]").
function _confVal(v) {
  // Any object/array (incl. empty {} / []) -> JSON, never "[object Object]"; scalars -> String.
  return v !== null && typeof v === "object" ? JSON.stringify(v) : String(v);
}

function _flattenConfig(obj, prefix, rows) {
  for (const k of Object.keys(obj).sort()) {
    const key = prefix ? prefix + "." + k : k;
    const v = obj[k];
    if (v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).length) {
      _flattenConfig(v, key, rows);
    } else {
      rows.push([key, _confVal(v)]);
    }
  }
}

function configBlock(c) {
  if (!c || !Object.keys(c).length) return h("div");
  const wrap = h("div"); wrap.append(h("h2", null, "Run config"));
  const base = [], sections = [];
  for (const k of Object.keys(c).sort()) {
    const v = c[k];
    if (v && typeof v === "object" && !Array.isArray(v) && Object.keys(v).length) {
      const rows = []; _flattenConfig(v, "", rows);
      sections.push([k, rows]);
    } else {
      base.push([k, _confVal(v)]);
    }
  }
  if (base.length) wrap.append(metricTable("parameters", base));
  for (const [name, rows] of sections) wrap.append(metricTable(name, rows));
  return wrap;
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
  // Columns are the union of metric keys any submission reported — so a new game's
  // metrics show up with no viz change (ARC: success_rate/levels/rhae, MiniGrid: reward/steps).
  const keys = [];
  for (const s of traj) for (const k of Object.keys(s.metrics || {})) if (!keys.includes(k)) keys.push(k);
  const t = h("table");
  t.append(rowEl("th", ["#", ...keys, "error"]));
  for (const s of traj)
    t.append(rowEl("td", [s.submission, ...keys.map((k) => fmtMetric(s.metrics?.[k])), s.error || ""]));
  wrap.append(t); return wrap;
}
function flaggedPanel(flagged) {
  const wrap = h("div");
  wrap.append(h("h2", null, `Flagged tool calls (${flagged.length})`));
  const t = h("table");
  t.append(rowEl("th", ["turn", "tool", "command / args", "why flagged"]));
  for (const c of flagged)
    t.append(rowEl("td", [c.turn, c.tool, c.args, (c.flags || []).join("; ")]));
  wrap.append(t);
  return wrap;
}
function rowEl(cell, vals) {
  const tr = h("tr"); for (const v of vals) tr.append(h(cell, null, String(v))); return tr;
}

// ---------------------------------------------------------------- conversation tab
const _TAG_LABEL = { submit: "submit", submit_win: "submit ✓ level", cheat: "flagged" };
const _CWM_TOOL_STYLE = {
  UpdateCodeWorldModel: "cwm-model",
  PlanInCWM: "cwm-plan",
  SubmitExplorationController: "cwm-explore",
  RunController: "cwm-explore",
  ResetLevel: "cwm-explore",
  ResetEnvironment: "cwm-explore",
};

async function renderConversation(name) {
  const d = await gameDetail(name);
  const conv = h("div", "conv");
  const toTop = h("button", "totop", "\u2191 Top");
  toTop.title = "Back to the top of the page";
  toTop.onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });
  document.body.append(toTop);
  pageCleanups.push(() => toTop.remove());
  const navItems = [];     // {id, tag, label} — submissions + cheats, to jump to
  let nSubmit = 0, nCheat = 0, nCwm = 0;
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
        const block = toolBlock(it.tool, name);
        const tag = it.tool.tag;
        if (it.tool.framework_tool) {
          block.id = "nav-cwm-" + nCwm++;
          navItems.push({id: block.id, tag: _CWM_TOOL_STYLE[it.tool.framework_tool] || "cwm", label: it.tool.framework_tool, succeeded: it.tool.succeeded});
        } else if (tag === "submit" || tag === "submit_win") {
          block.id = "nav-submit-" + nSubmit;
          navItems.push({ id: block.id, tag, label: `submission ${nSubmit}${tag === "submit_win" ? " ✓ level" : ""}` });
          nSubmit++;
        }
        if (it.tool.flags?.length || tag === "cheat") {
          if (!block.id) block.id = "nav-cheat-" + nCheat;
          navItems.push({ id: block.id, tag, label: `flagged ${nCheat + 1}` });
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
      const a = h("a", "navitem nav-" + it.tag, h("span", "nav-label", it.label));
      if (it.succeeded) {
        const check = h("span", "tool-success", "✓");
        check.title = "Command completed successfully; this does not mean the full game is solved.";
        a.append(check);
      }
      a.onclick = (e) => {
        e.preventDefault();
        document.getElementById(it.id)?.scrollIntoView({ behavior: "smooth", block: "center" });
      };
      nav.append(a);
    }
  } else {
    nav.append(h("div", "muted", "no notable tool calls yet"));
  }
  const layout = h("div", "convlayout");
  layout.append(nav, conv);
  shell(name, "conversation", layout);
}

const _TOOL_TEXT_MAX = 4000;  // initial result preview; full results can be expanded
const _INLINE_ARG_MAX = 240;  // short args stay inline; longer ones get the scrollable block

const _FILE_TEXT_MAX = 20000;  // per extracted heredoc file (its own budget, larger than the args cap)

// Pull `cmd > FILE <<['"]?DELIM['"]? ... DELIM` heredocs out of a shell command. Robust by design:
// it matches the heredoc START (a line ending in `<<DELIM`) and then the line that IS the closing
// DELIM - it NEVER scans `<<` inside the body, so python bit-shifts (`1 << n`) can't fool it. An
// unclosed heredoc is left untouched (never eats the rest). Returns the command with each heredoc
// replaced by `[file: NAME]` plus the extracted {name, body} files (NAME = the redirect target,
// else standard input).
function parseHeredocs(command) {
  const lines = command.split("\n");
  const out = [];
  const files = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const m = line.match(/<<(-?)\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\2\s*$/);
    if (m) {
      const dash = m[1] === "-", delim = m[3];
      const prefix = line.slice(0, line.indexOf("<<"));
      const fm = prefix.match(/>>?\s*(\S+)|\btee\b\s+(?:-a\s+)?(\S+)/);
      let j = i + 1;
      const body = [];
      let closed = false;
      while (j < lines.length) {
        const cand = dash ? lines[j].replace(/^\t+/, "") : lines[j];
        if (cand === delim) { closed = true; break; }
        body.push(lines[j]);
        j++;
      }
      if (closed) {
        const name = fm && (fm[1] || fm[2]);
        files.push({ name, body: body.join("\n") });
        out.push(prefix.replace(/\s+$/, "") + (name ? " [file: " + name + "]" : " [stdin]"));
        i = j + 1;  // skip the closing delimiter line too
        continue;
      }
    }
    out.push(line);
    i++;
  }
  return { text: out.join("\n"), files };
}

function toolBlock(tool, gameName) {
  // The reader tags calls authoritatively: blue submit, green submit-that-won a level, red cheat.
  const cls = { cheat: " cheat", submit: " submit", submit_win: " submit-win" }[tool.tag] || "";
  const cwmStyle = _CWM_TOOL_STYLE[tool.framework_tool];
  const box = h("div", "tool" + cls + (cwmStyle ? " " + cwmStyle : ""));
  const t = h("div", "t");
  if (tool.framework_tool) {
    t.append(h("span", "tag tag-" + (cwmStyle || "cwm"), tool.framework_tool), " ");
    if (tool.succeeded) {
      const check = h("span", "tool-success", "✓");
      check.title = "Command completed successfully; this does not mean the full game is solved.";
      t.append(check);
    }
  } else if (tool.tag) t.append(h("span", "tag tag-" + tool.tag, _TAG_LABEL[tool.tag]), " ");
  if (tool.flags?.length && tool.tag !== "cheat") {
    t.append(h("span", "tag tag-cheat", "flagged"), " ");
  }
  t.append(h("b", null, tool.name), " ");
  box.append(t);
  renderToolArgs(tool, t, box);
  if (tool.result != null) {
    const res = h("div", "res" + (tool.is_error ? " err" : ""));
    const fullResult = String(tool.result);
    const resultText = h("pre", null, fullResult.slice(0, _TOOL_TEXT_MAX));
    res.append(resultText);
    if (fullResult.length > _TOOL_TEXT_MAX) {
      const expand = h("button", null, `Show full result (${fmt(fullResult.length)} characters)`);
      expand.onclick = () => { resultText.textContent = fullResult; expand.remove(); };
      res.append(expand);
    }
    for (const attachment of tool.images || []) {
      if (!attachment.filename) { res.append(h('p', 'err', attachment.error || 'Image unavailable')); continue; }
      const img = h('img'); img.alt = 'Image returned to the agent by this tool';
      img.src = `/api/game/tool-image?name=${encodeURIComponent(gameName)}&filename=${encodeURIComponent(attachment.filename)}`;
      img.style.cssText = 'max-width:100%;max-height:600px;object-fit:contain;image-rendering:pixelated';
      res.append(img);
    }
    box.append(res);
    const replayIds = new Set(controllerResultIds(fullResult));
    for (const id of tool.controller_playback_ids || []) replayIds.add(id);
    for (const id of replayIds) {
      box.append(controllerReplay(gameName, id));
    }
  }
  if (tool.flags?.length) box.append(h('p', 'err', 'Flagged: ' + tool.flags.join('; ')));
  return box;
}

// Shell tools (Claude Code's Bash, Codex's shell, Alan's Bash) carry the command as a string, or
// Codex as an argv list. Plain text reads far better than the escaped JSON of the raw arguments.
function shellCommand(input) {
  const cmd = input && (input.command ?? input.cmd);
  if (typeof cmd === "string") return cmd;
  if (Array.isArray(cmd) && cmd.every((part) => typeof part === "string")) return cmd.join(" ");
  return null;
}

// The arguments a renderer did not show (Claude Code's `description`, `timeout`, ...), as notes.
function argNotes(input, shown) {
  const rest = Object.entries(input || {}).filter(([key]) => !shown.includes(key));
  if (!rest.length) return null;
  const text = rest.map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`);
  return h("div", "argnotes muted", text.join("  ·  ").slice(0, _TOOL_TEXT_MAX));
}

function fileDrop(label, body) {
  const text = body.length > _FILE_TEXT_MAX ? body.slice(0, _FILE_TEXT_MAX) + "\n... (truncated)" : body;
  return h("details", "filedrop", h("summary", null, label), h("pre", "filebody", text));
}

function renderToolArgs(tool, header, box) {
  const input = tool.input || {};
  const cmd = shellCommand(input);
  if (cmd !== null) {
    // Heredocs that write files are pulled out into collapsed "file:" blocks.
    const parsed = parseHeredocs(cmd);
    header.append(h("pre", "args", parsed.text.slice(0, _TOOL_TEXT_MAX)));
    const notes = argNotes(input, ["command", "cmd"]);
    if (notes) header.append(notes);
    for (const f of parsed.files) box.append(fileDrop(f.name ? "file: " + f.name : "standard input", f.body));
    if (parsed.files.length) box.append(fileDrop("Original command", cmd));
    return;
  }
  if (typeof input.file_path === "string") {  // Read / Write / Edit and their kin
    header.append(h("span", "args-path", input.file_path));
    const shown = ["file_path"];
    if (typeof input.content === "string") { box.append(fileDrop("content", input.content)); shown.push("content"); }
    if (typeof input.old_string === "string" && typeof input.new_string === "string") {
      box.append(fileDrop("replace", input.old_string), fileDrop("with", input.new_string));
      shown.push("old_string", "new_string");
    }
    const notes = argNotes(input, shown);
    if (notes) header.append(notes);
    return;
  }
  // Any other tool: its raw arguments, inline when short.
  const raw = JSON.stringify(input);
  header.append(raw.length > _INLINE_ARG_MAX ? h("pre", "args", raw.slice(0, _TOOL_TEXT_MAX)) : h("span", "muted", raw));
}

function controllerResultIds(text) {
  // Parse complete JSON objects, never infer a replay link from shell input text.
  const ids = new Set();
  let start = -1, depth = 0, quoted = false, escaped = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (start < 0) { if (c === '{') { start = i; depth = 1; } continue; }
    if (quoted) { if (escaped) escaped = false; else if (c === '\\') escaped = true; else if (c === '"') quoted = false; continue; }
    if (c === '"') quoted = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) {
      // The current-observation field identifies managed-call feedback. Older
      // submissions did not record per-call sequences and cannot use this replay.
      try { const value = JSON.parse(text.slice(start, i + 1)); if (value.status && Number.isInteger(value.exploration_id) && Number.isInteger(value.current_observation_id)) ids.add(value.exploration_id); } catch (_) {}
      start = -1;
    }
  }
  return [...ids];
}

function controllerReplay(game, id) {
  const box = h('div', 'controller-replay');
  const button = h('button', null, `Load controller #${id} playback`);
  box.append(button);
  button.onclick = async () => {
    button.disabled = true;
    const query = `name=${encodeURIComponent(game)}&kind=controller&identifier=${id}`;
    try {
      const meta = await api('/api/game/cwm/load?' + query, {method: 'POST'});
      const img = h('img'); img.alt = 'Recorded real environment (viewer playback)';
      img.style.cssText = 'max-width:100%;max-height:400px;image-rendering:pixelated;object-fit:contain';
      const label = h('div', 'muted');
      const slider = h('input'); slider.type = 'range'; slider.min = 0; slider.max = meta.frames - 1; slider.value = 0;
      slider.style.width = `min(100%, ${Math.max(0, (meta.frames - 1) * 12)}px)`;
      slider.disabled = meta.frames < 2;
      const data = h('details', null, h('summary', null, 'Frame data'));
      const raw = h('pre'); data.append(raw);
      let sequence = 0, playing = false;
      const show = async () => {
        const generation = ++sequence, index = Number(slider.value);
        img.src = '/api/game/cwm/frame?' + query + `&index=${index}&image=true`;
        label.textContent = `Frame ${index + 1}/${meta.frames} · real recorded experience`;
        const result = await api('/api/game/cwm/frame?' + query + `&index=${index}`);
        if (generation === sequence) raw.textContent = JSON.stringify(result, null, 2);
      };
      slider.oninput = () => { playing = false; show().catch(err => label.textContent = String(err)); };
      const play = h('button', null, 'Play'); play.disabled = meta.frames < 2;
      play.onclick = async () => {
        playing = !playing;
        if (Number(slider.value) >= meta.frames - 1) slider.value = 0;
        while (playing && box.isConnected) {
          await show();
          if (Number(slider.value) >= meta.frames - 1) break;
          await new Promise(resolve => setTimeout(resolve, 250));
          if (!playing) break;
          slider.value = Number(slider.value) + 1;
        }
        playing = false;
      };
      const unload = h('button', null, 'Unload');
      unload.onclick = () => { playing = false; sequence++; clear(box); box.append(button); button.disabled = false; };
      clear(box); box.append(label, img, h('div', null, slider), play, unload, data);
      await show();
    } catch (err) { button.disabled = false; box.append(h('p', 'err', String(err))); }
  };
  return box;
}

// ---------------------------------------------------------------- artifacts tab
// A classic file tree: folders first, closed until clicked; onFile(file, element) opens a file.
function fileTree(files, onFile) {
  const root = { dirs: new Map(), files: [] };
  for (const f of files) {
    const parts = f.relpath.split("/");
    let node = root;
    for (const part of parts.slice(0, -1)) {
      if (!node.dirs.has(part)) node.dirs.set(part, { dirs: new Map(), files: [] });
      node = node.dirs.get(part);
    }
    node.files.push({ name: parts[parts.length - 1], file: f });
  }
  const render = (node, depth) => {
    const box = h("div");
    for (const name of [...node.dirs.keys()].sort()) {
      const head = h("div", "fileitem treedir", "\u25B8 " + name + "/");
      const body = render(node.dirs.get(name), depth + 1);
      head.style.paddingLeft = 10 + depth * 14 + "px";
      body.hidden = true;
      head.onclick = () => {
        body.hidden = !body.hidden;
        head.textContent = (body.hidden ? "\u25B8 " : "\u25BE ") + name + "/";
      };
      box.append(head, body);
    }
    for (const entry of node.files.sort((a, b) => a.name.localeCompare(b.name))) {
      const item = h("div", "fileitem", entry.name);
      item.style.paddingLeft = 10 + depth * 14 + "px";
      item.title = entry.file.relpath;
      item.onclick = () => onFile(entry.file, item);
      box.append(item);
    }
    return box;
  };
  return render(root, 0);
}

async function renderArtifacts(name) {
  const d = await api("/api/game/artifacts?name=" + encodeURIComponent(name));
  const wrap = h("div", "split");
  const list = h("div", "filelist");
  const view = h("div", "fileview", h("div", "muted", "select a file"));
  list.append(h("div", "h", "Workdir files"));
  const show = (f, item) => {
    [...list.querySelectorAll(".fileitem")].forEach((x) => x.classList.remove("on"));
    item.classList.add("on");
    clear(view);
    view.append(h("h3", null, f.relpath),
      f.too_large ? h("div", "muted", `Preview omitted: ${fmt(f.size_bytes)} bytes exceeds the 200,000-byte automatic preview limit. The complete file is retained in the run's workdir.`) : h("pre", "code", f.content));
  };
  list.append(fileTree(d.files, show));
  if (!d.files.length) list.append(h("div", "muted", "none"));
  wrap.append(list, view);

  const subs = h("div"); subs.append(h("h2", null, "Submissions & videos"));
  if (!d.submissions.length) subs.append(h("div", "muted", "no submissions"));
  for (const s of d.submissions) {
    const c = h("div", "sub", h("h3", null, "submission " + s.name));
    if (s.error) c.append(h("div", "badge b-bad", s.error));
    const a = s.aggregate || {};
    c.append(h("div", "muted", `${aggLine(a)} · n=${a.n_episodes ?? "—"}`));
    for (const v of s.videos || []) {
      const vid = h("video"); vid.controls = true; vid.preload = "metadata";
      vid.src = `/video?game=${encodeURIComponent(name)}&submission=${encodeURIComponent(s.name)}&filename=${encodeURIComponent(v)}`;
      c.append(vid);
    }
    subs.append(c);
  }
  const cwm = _cache[name]?.config?.protocol?.name === "cwm";
  shell(name, "artifacts", cwm ? wrap : h("div", null, wrap, subs));
}

// ---------------------------------------------------------------- logs tab
async function renderLogs(name) {
  const d = await api("/api/game/logs?name=" + encodeURIComponent(name));
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
  navigation.abort();
  for (const cleanup of pageCleanups) cleanup();
  pageCleanups = [];
  navigation = new AbortController();
  const token = navigation;
  for (const key of Object.keys(_cache)) delete _cache[key];
  for (const key of Object.keys(_gamesCache)) delete _gamesCache[key];
  try {
    const parts = (location.hash || "").replace(/^#\/?/, "").split("/").filter(Boolean);
    if (!parts.length) return await renderBrowse();
    if (parts[0] === "graphs") return await renderGraphs(parts[1] ? decodeURIComponent(parts[1]) : "");
    if (parts[0] === "run") return await renderDashboard(parts[1] ? decodeURIComponent(parts[1]) : "");
    if (parts[0] !== "game" || !parts[1]) return await renderBrowse();
    const name = decodeURIComponent(parts[1]);
    await gameDetail(name);
    const tab = parts[2] || "";
    if (tab === "conversation") await renderConversation(name);
    else if (tab === "artifacts") await renderArtifacts(name);
    else if (tab === "logs") await renderLogs(name);
    else if (tab === "graphs") await renderGameGraphs(name);
    else await renderOverview(name);
  } catch (e) {
    if (token !== navigation || e.name === "AbortError") return;
    clear(app); app.append(h("pre", "err", "error: " + e.message));
    const retry = h("button", null, "Retry"); retry.onclick = route; app.append(retry);
  }
}
window.addEventListener("hashchange", route);
route();
