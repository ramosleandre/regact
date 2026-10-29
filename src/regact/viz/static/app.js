"use strict";
// All DOM is built with createElement + textContent (via h()); no innerHTML, so
// transcript/log content is inserted as text, never parsed as HTML (XSS-safe).
const app = document.getElementById("app");
const crumb = document.getElementById("crumb");
document.getElementById("brand").onclick = () => { location.hash = ""; };
document.getElementById("goup").onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });

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

// An exported page (exportPage) carries its data, settings and images; it never calls a server.
const SNAPSHOT = (typeof window !== "undefined" && window.__REGACT_SNAPSHOT__) || null;
const _responses = {};   // path -> last response, what an export embeds
const asset = (src) => (SNAPSHOT && SNAPSHOT.images[src]) || src;
// A public copy (python -m regact.viz.export): every request is a file under data/, named by the
// FNV-1a hash of the request the server would have answered; replay frames come in one pack.
const STATIC = document.querySelector?.('meta[name="regact-static"]')?.content || null;
function requestKey(url) {
  let v = 0xcbf29ce484222325n;
  for (const b of new TextEncoder().encode(url)) v = ((v ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn;
  return v.toString(16).padStart(16, "0");
}
const _packs = {};       // request -> data URL, for the frames of the games loaded so far
function media(url) {
  if (!STATIC) return url;
  if (_packs[url]) return _packs[url];
  const name = new URLSearchParams(url.split("?")[1] || "").get("filename") || "";
  const ext = url.startsWith("/video") ? ".mp4" : (name.match(/\.\w+$/) || [".png"])[0];
  return STATIC + requestKey(url) + ext;
}
const _packIndex = {};   // request -> [pack chunk file, position], for the games loaded so far
async function loadPack(name) {
  if (!STATIC) return;
  const r = await fetch(STATIC + requestKey("pack:" + name) + ".json");
  if (!r.ok) return;
  for (const [url, [chunk, i]] of Object.entries((await r.json()).urls))
    _packIndex[url] = [STATIC + requestKey(`pack:${name}:${chunk}`) + ".json", i];
}
const _chunks = {};      // chunk file -> promise of its images
async function setMedia(img, url) {   // like img.src = media(url), fetching a pack chunk if needed
  img.dataset.want = url;
  const at = STATIC && !_packs[url] && _packIndex[url];
  if (at) {
    _chunks[at[0]] ??= fetch(at[0]).then((r) => r.json());
    _packs[url] = (await _chunks[at[0]])[at[1]];
  }
  if (img.dataset.want === url) img.src = media(url);   // a later frame may have been asked for since
}

let navigation = new AbortController();
let pageCleanups = [];
function checkNavigation(token) {
  if (token !== navigation || token.signal.aborted) throw new DOMException("Navigation changed", "AbortError");
}
async function api(path, opts = {}) {
  const token = navigation;
  if (SNAPSHOT) {
    if (!(path in SNAPSHOT.responses)) throw new Error("not included in this exported page");
    return JSON.parse(JSON.stringify(SNAPSHOT.responses[path]));
  }
  if (STATIC && path.startsWith("/api/")) {
    if (opts.method === "POST") throw new Error("Interactive playback runs in the local viewer only (make viz).");
    const r = await fetch(STATIC + requestKey(path) + ".json", {signal: token.signal});
    if (!r.ok) throw new Error("This page is not part of the public site.");
    const data = await r.json();
    checkNavigation(token);
    return data;
  }
  const r = await fetch(path, {cache: "no-store", signal: token.signal, ...opts});
  if (!r.ok) {
    const text = await r.text();
    const hint = r.status === 404 && path.startsWith('/api/game/cwm')
      ? " The viewer server may predate CWM support; restart make viz." : "";
    throw new Error(`HTTP ${r.status}: ${text}${hint}`);
  }
  const data = await r.json();
  checkNavigation(token);
  _responses[path] = data;
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
// A run "finished" iff it ended for a reason of the task itself (solved, a budget, gave up...).
// Dropped when masking: a run still going or whose process died (no exit_reason), and one that
// ended on something outside the task - the exits regact can resume. An error the run recovered
// from (last_error_category) does not count: it went on and finished.
const _CRASH_EXITS = new Set(["agent_api", "loop_crash", "interrupted", "usage_limit"]);
const isValidRun = (m) => !!m.exit_reason && !_CRASH_EXITS.has(m.exit_reason);
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
function currentSettings() {
  return {
    version: 1, agg: _graph.agg, err: _graph.err, mask: _graph.mask, barScale: _graph.barScale,
    order: _graph.order, colors: _graph.colors, curveX: _graph.curveX, lineStyles: _graph.lineStyles,
    gamma: _graph.gamma, comparable: _graph.comparable,
    hidden: [..._graph.hidden], active: [..._graph.active], known: _graph.known,
    perRun: [..._graph.perRun], muted: [..._graph.muted],
  };
}
function saveSettings() {   // debounced PUT of the whole panel state for the current interface
  if (SNAPSHOT || STATIC) return;
  const body = currentSettings();
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
  // A chart legend entry cycles: drawn -> not drawn (muted) -> drawn with each run (perRun) -> drawn.
  muted: new Set(), perRun: new Set(),
  zoom: {},   // x measure -> [from, to] shown by the progress charts (not saved: drag to set)
  lineStyles: {},   // experiment -> a LINE_STYLES key for its progress curves (persisted)
  gamma: 1,   // discount of the progress-curve area (persisted)
  comparable: false,   // efficiency scores cut where the shortest experiment ends (persisted)
};
const LINE_STYLES = { solid: "", dashed: "8 4", dotted: "2 3", "dash-dot": "8 3 2 3" };

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
  _graph.lineStyles = (s.lineStyles && typeof s.lineStyles === "object") ? s.lineStyles : {};
  _graph.gamma = typeof s.gamma === "number" && s.gamma >= 0 && s.gamma <= 1 ? s.gamma : 1;
  _graph.comparable = !!s.comparable;
  _graph.hidden = new Set(Array.isArray(s.hidden) ? s.hidden : []);
  _graph.perRun = new Set(Array.isArray(s.perRun) ? s.perRun : []);
  _graph.muted = new Set(Array.isArray(s.muted) ? s.muted : []);
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
        specs.push({ key: "agg:" + k, label: k.replace(/_/g, " "), score: k, get: (m) => m.final_aggregate && m.final_aggregate[k] });
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
  // Its discounted area, where a reference player gives a scale to compare with (ARC).
  if (prog && games.some((g) => g.metrics.progress && (g.metrics.progress.reference || []).length))
    specs.push({ key: "progress_auc", label: "Area under " + prog.label.toLowerCase(), auc: true, def: true });
  // Each level's actions over the reference player's on that level (mean relative actions, level by level).
  if (prog && games.some((g) => g.metrics.progress && (g.metrics.progress.reference || []).length))
    specs.push({ key: "relative_levels", label: "Relative actions over levels", levels: true, def: true });
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
    if (_graph.perRun.has(e) && !spec.count) max = Math.max(max, ...seriesOf(e, t));   // and run ticks
  }
  for (const t of tasks) if (spec.refOf && spec.refOf(t) != null) max = Math.max(max, spec.refOf(t));
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
      if (_graph.perRun.has(exp) && col !== _AVG_COL && !spec.count)
        for (const run of seriesOf(exp, col)) {     // one tick per run across the bar
          const tick = svg("line", { class: "runtick", x1: x - 1, y1: yOf(run), x2: x + bw + 1, y2: yOf(run) });
          tick.append(svg("title", {}, txt(`${expLeaf(exp)} · ${colName}\none run: ${spec.fmt ? spec.fmt(run) : fmtMetric(run)}`)));
          s.append(tick);
        }
      const eb = colErr(exp, col);               // error/interval bar centered on the bar
      if (eb) {
        const yl = yOf(eb.lo), yh = yOf(eb.hi), cap = Math.max(2, bw * 0.3);
        s.append(svg("line", { class: "errbar", x1: cx, y1: yh, x2: cx, y2: yl }));
        s.append(svg("line", { class: "errbar", x1: cx - cap, y1: yh, x2: cx + cap, y2: yh }));
        s.append(svg("line", { class: "errbar", x1: cx - cap, y1: yl, x2: cx + cap, y2: yl }));
      }
    });
    const ref = col !== _AVG_COL && spec.refOf ? spec.refOf(col) : null;   // the reference player's value
    if (ref != null) {
      const line = svg("line", { class: "barref", x1: gx - 3, y1: yOf(ref), x2: gx + groupW - 13, y2: yOf(ref) });
      line.append(svg("title", {}, txt(`reference player · ${colName}\n${spec.label}: ${spec.fmt ? spec.fmt(ref) : fmtMetric(ref)}`)));
      s.append(line, svg("text", { class: "reflab", x: gx + groupW - 11, y: yOf(ref) + 3 }, txt("ref")));
    }
    const lx = gx + (groupW - 16) / 2, ly = H - padB + 12;   // x label (task), rotated; bold for the mean
    s.append(svg("text", { class: "xlab" + (col === _AVG_COL ? " avglab" : ""), x: lx, y: ly, "text-anchor": "end", transform: `rotate(-35 ${lx} ${ly})` }, txt(shortTask(col))));
  });
  appendTaskThumbs(s, cols, (ci) => padL + ci * groupW + 8 + (groupW - 16) / 2, yThumb, thumb);
  return s;
}

// Progress curves. Per run: the problem's progress as a step function of what the run had spent
// (`point` = the field of a progress point, `end` = where the run stopped). In the aggregate a run
// that stopped keeps its last value for as long as another run of the experiment is still going,
// and the line ends with the last of them; the mask drops crashed / unfinished runs altogether. A
// run's own line (legend, third state) ends where that run stopped.
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
    const levels = (m.progress.reference || []).length;   // the levels to win, when known
    const won = levels > 0 && Math.max(0, ...pts.map((q) => q[1])) >= levels;
    runs.push({ pts, done, won, end: Math.max(Number(X.end(m)) || 0, ...pts.map((q) => q[0])) });
  }
  return runs;
}
// Per run, the actions it spent on each level it completed over the reference player's actions on
// that level; aggregated per level over the runs that completed it.
function relativeLevelsChart(task, group, expColor, aggName, errMethod, mask) {
  const X = CURVE_X["env actions"], agg = AGGREGATORS[aggName];
  const ms = group.experiments.flatMap((exp) => (group.byExp.get(exp) && group.byExp.get(exp).get(task)) || []);
  const reference = ((ms.find((m) => m.progress && (m.progress.reference || []).length) || {}).progress || {}).reference || [];
  if (!reference.length) return null;
  const baseline = reference.map(([x], i) => x - (i ? reference[i - 1][0] : 0));
  const ratios = (run) => {   // [ratio per completed level], from the levels-over-actions points
    const solved = [];
    let best = 0;
    for (const [x, v] of [...run.pts].sort((a, b) => a[0] - b[0])) while (best < v) { solved.push(x); best += 1; }
    return solved.slice(0, baseline.length).map((x, i) => (x - (i ? solved[i - 1] : 0)) / baseline[i]);
  };
  const series = group.experiments.map((exp) => ({ exp, runs: curveRuns(exp, task, group, mask, X).map(ratios) }))
    .filter((sr) => sr.runs.length);
  if (!series.length) return null;
  const L = baseline.length;
  const perLevel = series.map(({ exp, runs }) => ({ exp, runs, levels: Array.from({ length: L }, (_, i) =>
    runs.map((r) => r[i]).filter((v) => v != null && Number.isFinite(v))) }));
  const values = perLevel.flatMap((sr) => sr.levels.flat());
  const yMax = Math.max(1.5, ...values) * 1.05;
  const room = (typeof window !== "undefined" && window.innerWidth) || 0;
  const padL = 44, padR = 16, padT = 10, padB = 30, W = Math.max(420, Math.min(1100, room - 500)), H = 240;
  const xOf = (i) => padL + (W - padL - padR) * (L > 1 ? i / (L - 1) : 0.5);
  const yOf = (v) => padT + (H - padT - padB) * (1 - Math.min(1, v / yMax));
  const s = svg("svg", { class: "chart", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  const step = yMax > 4 ? 1 : 0.5;
  for (let v = 0; v <= yMax; v += step) {
    s.append(svg("line", { class: "gridline", x1: padL, y1: yOf(v), x2: W - padR, y2: yOf(v) }));
    s.append(svg("text", { class: "ylab", x: padL - 6, y: yOf(v) + 3, "text-anchor": "end" }, txt(String(v))));
  }
  for (let i = 0; i < L; i++)
    s.append(svg("text", { class: "xlab", x: xOf(i), y: H - padB + 16, "text-anchor": "middle" }, txt("L" + (i + 1))));
  s.append(svg("path", { class: "curve-ref", d: `M${xOf(0)},${yOf(1)} H${xOf(L - 1)}` }));
  for (const { exp, runs, levels } of perLevel) {
    const color = expColor(exp);
    if (_graph.perRun.has(exp))
      for (const r of runs) {
        const d = r.map((v, i) => `${i ? "L" : "M"}${xOf(i)},${yOf(v)}`).join(" ");
        if (d) s.append(svg("path", { class: "curve-run", stroke: color, d }));
      }
    const pts = levels.map((xs, i) => (xs.length ? [i, agg(xs), xs] : null)).filter(Boolean);
    if (errMethod !== "none")
      for (const [i, , xs] of pts) {
        if (xs.length < 2) continue;
        let lo, hi;
        if (errMethod === "range") { lo = Math.min(...xs); hi = Math.max(...xs); }
        else {
          const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
          const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1));
          lo = mean - sd; hi = mean + sd;
        }
        s.append(svg("line", { stroke: color, "stroke-width": 2, opacity: 0.5, x1: xOf(i), x2: xOf(i), y1: yOf(Math.max(0, lo)), y2: yOf(hi) }));
      }
    let d = "", prev = null;
    for (const [i, v] of pts) { d += `${prev === i - 1 ? " L" : " M"}${xOf(i)},${yOf(v)}`; prev = i; }
    s.append(svg("path", { class: "curve-line", stroke: color, d }));
    for (const [i, v, xs] of pts) {
      const dot = svg("circle", { cx: xOf(i), cy: yOf(v), r: 3.5, fill: color });
      dot.append(svg("title", {}, txt(`${exp}\nlevel ${i + 1}: ${v.toFixed(2)} (${aggName} of ${xs.length} run${xs.length > 1 ? "s" : ""})`)));
      s.append(dot);
    }
  }
  return s;
}
const curveAt = (run, x) => { let v = 0; for (const [px, pv] of run.pts) if (px <= x) v = pv; return v; };
let _clipIds = 0;
function progressChart(task, group, expColor, aggName, errMethod, mask, xName, onZoom) {
  const X = CURVE_X[xName], agg = AGGREGATORS[aggName];
  const series = group.experiments.map((exp) => ({ exp, runs: curveRuns(exp, task, group, mask, X) }))
    .filter((sr) => sr.runs.length);
  if (!series.length) return null;
  const all = series.flatMap((sr) => sr.runs);
  // The reference player is known in env actions only, but its top value sets the y axis for every
  // x, so the same run reads at the same height whichever x is chosen.
  const reference = ((((group.byExp.get(series[0].exp) || new Map()).get(task) || []).find((m) => m.progress) || {}).progress || {}).reference || [];
  const ref = xName === "env actions" ? reference : [];
  const xMax = Math.max(1, ...all.map((r) => r.end));
  const yMax = Math.max(1, ...all.flatMap((r) => r.pts.map((q) => q[1])), ...reference.map((q) => q[1]));
  // Each run also gets a point just after its end, so it leaves the aggregate there and not at
  // whichever point of another run comes next.
  const eps = xMax / 2000;
  const grid = [...new Set([0, xMax, ...all.flatMap((r) => [r.end, r.end + eps, ...r.pts.map((q) => q[0])])])]
    .filter((x) => x <= xMax).sort((a, b) => a - b);
  // As wide as the window leaves beside the two sidebars (metrics/experiments, and this chart's legend).
  const room = (typeof window !== "undefined" && window.innerWidth) || 0;
  const padL = 44, padR = 16, padT = 10, padB = 30, W = Math.max(620, Math.min(1700, room - 500)), H = W > 900 ? 340 : 270;
  // A zoom shows [x0, x1] of the same curves; what falls outside is clipped, not recomputed.
  const span = _graph.zoom[xName];
  const x0 = span ? span[0] : 0, x1 = span ? Math.max(span[1], span[0] + 1e-9) : xMax;
  const xOf = (x) => padL + (W - padL - padR) * (span ? (x - x0) / (x1 - x0) : Math.min(1, x / xMax));
  const yOf = (v) => padT + (H - padT - padB) * (1 - Math.max(0, Math.min(1, v / yMax)));
  const s = svg("svg", { class: "chart", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  const clipId = `curveclip${++_clipIds}`;
  const clip = svg("clipPath", { id: clipId });
  clip.append(svg("rect", { x: padL, y: 0, width: W - padL - padR, height: H }));
  s.append(clip);
  const clipped = (el) => { el.setAttribute("clip-path", `url(#${clipId})`); return el; };
  for (let v = 0; v <= yMax; v++) {
    s.append(svg("line", { class: "gridline", x1: padL, y1: yOf(v), x2: W - padR, y2: yOf(v) }));
    s.append(svg("text", { class: "ylab", x: padL - 6, y: yOf(v) + 3, "text-anchor": "end" }, txt(String(v))));
  }
  for (const f of [0, 0.25, 0.5, 0.75, 1])
    s.append(svg("text", { class: "xlab", x: xOf(x0 + (x1 - x0) * f), y: H - padB + 16, "text-anchor": "middle" }, txt(X.fmt(Math.round(x0 + (x1 - x0) * f)))));
  const steps = (ys, xs = grid) => {   // step-after path through (xs[i], ys[i]); null = no run there
    let d = "", pen = false;
    xs.forEach((x, i) => {
      if (ys[i] == null) { pen = false; return; }
      d += pen ? ` V${yOf(ys[i])}` : ` M${xOf(x)},${yOf(ys[i])}`;
      pen = true;
      if (i + 1 < xs.length) d += ` H${xOf(xs[i + 1])}`;
    });
    return d;
  };
  if (ref.length) {
    // On its own points: the runs' grid would move each of its steps to the next run event.
    const refXs = [0, ...ref.map((q) => q[0]).filter((x) => x < xMax), xMax];
    const path = svg("path", { class: "curve-ref", d: steps(refXs.map((x) => curveAt({ pts: ref }, x)), refXs) });
    path.append(svg("title", {}, txt("reference player")));
    s.append(clipped(path));
  }
  for (const { exp, runs } of series) {
    const lastEnd = Math.max(...runs.map((r) => r.end));   // the aggregate ends with the last run
    const at = grid.map((x) => (x <= lastEnd ? runs.map((r) => curveAt(r, x)) : []));
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
        s.append(clipped(svg("rect", { class: "curve-band", x: xOf(x), y: yh, width: xOf(grid[i + 1]) - xOf(x), height: yl - yh, fill: color })));
      });
    }
    if (_graph.perRun.has(exp))
      for (const run of runs) {
        s.append(clipped(svg("path", { class: "curve-run", stroke: color,
          d: steps(grid.map((x) => (x <= run.end ? curveAt(run, x) : null))) })));
        s.append(clipped(svg("circle", { class: "curve-end run", cx: xOf(run.end), cy: yOf(curveAt(run, run.end)), r: 2.5, fill: color })));
      }
    const line = svg("path", { class: "curve-line", stroke: color, d: steps(at.map((xs) => (xs.length ? agg(xs) : null))) });
    const dash = LINE_STYLES[_graph.lineStyles[exp]];
    if (dash) line.setAttribute("stroke-dasharray", dash);
    line.append(svg("title", {}, txt(`${expLeaf(exp)} · ${task}\n${runs.length} run${runs.length > 1 ? "s" : ""}`)));
    s.append(clipped(line));
    s.append(clipped(svg("circle", { class: "curve-end", cx: xOf(lastEnd), cy: yOf(agg(runs.map((r) => curveAt(r, lastEnd)))), r: 4, fill: color })));   // where it stops
  }
  if (onZoom) zoomByDrag(s, W, padL, padR, padT, H - padB, (px) => x0 + (x1 - x0) * Math.max(0, Math.min(1, (px - padL) / (W - padL - padR))), onZoom);
  return { chart: s, hasRef: ref.length > 0 };
}

// The discounted area under a progress step curve: the integral of gamma^x * value over [0, horizon],
// exact on each step (so it also holds for seconds). gamma = 1 is the plain area (levels x actions).
function discountedArea(pts, horizon, gamma) {
  const lg = Math.log(gamma);
  const piece = (a, b) => (gamma === 1 ? b - a : gamma === 0 ? 0 : Math.pow(gamma, a) * Math.expm1((b - a) * lg) / lg);
  let area = 0, value = 0, from = 0;
  for (const [x, v] of [...pts].sort((p, q) => p[0] - q[0])) {
    if (x >= horizon) break;
    if (x > from) { area += value * piece(from, x); from = x; }
    value = v;
  }
  return horizon > from ? area + value * piece(from, horizon) : area;
}
// Up to where every shown experiment still has a curve: the shortest of their aggregate lines.
function areaHorizon(task, group, mask, X) {
  const ends = group.experiments.map((exp) => curveRuns(exp, task, group, mask, X))
    .filter((runs) => runs.length).map((runs) => Math.max(...runs.map((r) => r.end)));
  return ends.length ? Math.min(...ends) : null;
}
// The bar-chart spec of the area: each run's area up to its game's horizon, a run that stopped
// earlier keeping its last value (as in the aggregate curve), and the reference player's area.
function areaSpec(spec, group, mask, xName, gamma) {
  const X = CURVE_X[xName], byRun = new Map(), refs = new Map(), horizons = new Map();
  for (const task of group.tasks) {
    const horizon = areaHorizon(task, group, mask, X);
    if (horizon == null) continue;
    horizons.set(task, horizon);
    for (const exp of group.experiments)
      for (const m of (group.byExp.get(exp) && group.byExp.get(exp).get(task)) || []) {
        const pts = ((m.progress || {}).points || []).map((q) => [q[X.point], q.value]);
        if (pts.length && pts.every((q) => q[0] != null)) byRun.set(m, discountedArea(pts, horizon, gamma));
        const reference = (m.progress || {}).reference || [];
        if (xName === "env actions" && reference.length && !refs.has(task)) refs.set(task, discountedArea(reference, horizon, gamma));
      }
  }
  return { ...spec, get: (m) => byRun.get(m), refOf: (task) => refs.get(task), horizons };
}
// RHAE, LRHAE-Uncapped and mean relative actions of a run as if it had stopped after `horizon`
// env actions, computed as problems/arc_agi/scoring.py does: a level completed later counts as not
// completed. `reference` is the human's cumulative actions at each level, so it also gives the
// levels to win.
const _LEVEL_CAP = 1.15;
function efficiencyScores(points, reference, horizon) {
  const solved = [];   // env actions when each level was first completed
  let best = 0;
  for (const [x, v] of [...points].sort((p, q) => p[0] - q[0])) {
    if (x > horizon) break;
    while (best < v) { solved.push(x); best += 1; }
  }
  let total = 0, solvedWeight = 0, capped = 0, linear = 0, relative = 0, nSolved = 0;
  reference.forEach(([x], i) => {
    const weight = i + 1, baseline = x - (i ? reference[i - 1][0] : 0);
    const actions = i < solved.length ? solved[i] - (i ? solved[i - 1] : 0) : 0;
    total += weight;
    if (actions > 0 && baseline > 0) {
      const ratio = baseline / actions;
      capped += Math.min(_LEVEL_CAP, ratio * ratio) * weight;
      linear += ratio * weight;
      solvedWeight += weight;
      relative += actions / baseline;
      nSolved += 1;
    }
  });
  return total ? { rhae: Math.min(solvedWeight, capped) / total, lrhae_uncapped: linear / total,
                   mean_relative_actions: nSolved ? relative / nSolved : null } : null;
}
// ROAE/LROAE are the same scores against an optimal player (grid games).
const _COMPARABLE = { "derived:rhae": "rhae", "derived:lrhae_uncapped": "lrhae_uncapped",
                      "derived:roae": "rhae", "derived:lroae": "lrhae_uncapped",
                      "derived:mean_relative_actions": "mean_relative_actions" };
// Where every experiment still has data: the end of each experiment's longest unfinished run, the
// earliest of those. A run that won the game cannot progress further, so it limits nothing; when
// every run won, nothing is cut (Infinity) and comparable equals all.
function comparableHorizon(task, group, mask, X) {
  const ends = group.experiments.map((exp) => curveRuns(exp, task, group, mask, X))
    .filter((runs) => runs.length).map((runs) => Math.max(...runs.map((r) => (r.won ? Infinity : r.end))));
  return ends.length ? Math.min(...ends) : null;
}
function comparableSpec(spec, group, mask) {
  const X = CURVE_X["env actions"], byRun = new Map(), horizons = new Map();
  for (const task of group.tasks) {
    const horizon = comparableHorizon(task, group, mask, X);
    if (horizon == null) continue;
    horizons.set(task, horizon);
    for (const exp of group.experiments)
      for (const m of (group.byExp.get(exp) && group.byExp.get(exp).get(task)) || []) {
        const pts = ((m.progress || {}).points || []).map((q) => [q[X.point], q.value]);
        const reference = (m.progress || {}).reference || [];
        if (!pts.length || pts.some((q) => q[0] == null) || !reference.length) continue;
        const scores = efficiencyScores(pts, reference, horizon);
        if (scores) byRun.set(m, scores[_COMPARABLE[spec.key]]);
      }
  }
  return { ...spec, get: (m) => byRun.get(m), horizons };
}

// gamma on a logit scale: log resolution near 1 and near 0, with the slider ends exactly 1 and 0.
const _GAMMA_LOGIT = 6, _GAMMA_STEPS = 1000;
const gammaOf = (p) => (p <= 0 ? 1 : p >= _GAMMA_STEPS ? 0 : 1 / (1 + Math.pow(10, (2 * p / _GAMMA_STEPS - 1) * _GAMMA_LOGIT)));
const gammaSlider = (g) => (g >= 1 ? 0 : g <= 0 ? _GAMMA_STEPS : Math.round((1 - Math.log10(g / (1 - g)) / _GAMMA_LOGIT) * _GAMMA_STEPS / 2));
const fmtGamma = (g) => (g === 1 || g === 0 ? String(g) : 1 - g < 0.01 ? `1 - ${(1 - g).toPrecision(2)}` : g < 0.01 ? g.toPrecision(2) : g.toFixed(3));
function gammaControl(redraw) {
  const label = h("span", "gammaval");
  const show = (g) => { label.textContent = `gamma = ${fmtGamma(g)}` + (g < 1 && g > 0 ? ` (horizon ~ ${fmt(Math.round(1 / (1 - g)))})` : ""); };
  const sl = h("input"); sl.type = "range"; sl.min = "0"; sl.max = String(_GAMMA_STEPS); sl.step = "1";
  sl.value = String(gammaSlider(_graph.gamma)); sl.title = "discount per unit of x: 1 (left) to 0 (right)";
  sl.oninput = () => show(gammaOf(Number(sl.value)));   // a redraw rebuilds the slider: only on release
  sl.onchange = () => { _graph.gamma = gammaOf(Number(sl.value)); redraw(); };
  show(_graph.gamma);
  return h("label", "widthctl gammactl", sl, label);
}

// Drag across a chart to show only that x range; a double-click shows the whole run again.
function zoomByDrag(s, W, padL, padR, top, bottom, toX, onZoom) {
  let from = null, band = null;
  s.style.cursor = "crosshair";
  const at = (e) => { const r = s.getBoundingClientRect(); return Math.max(padL, Math.min(W - padR, (e.clientX - r.left) * (W / r.width))); };
  s.onmousedown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    from = at(e);
    band = svg("rect", { class: "zoom-band", x: from, y: top, width: 0, height: bottom - top });
    s.append(band);
  };
  s.onmousemove = (e) => {
    if (from == null) return;
    const to = at(e);
    band.setAttribute("x", Math.min(from, to));
    band.setAttribute("width", Math.abs(to - from));
  };
  const finish = (e, keep) => {
    if (from == null) return;
    const to = at(e), a = Math.min(from, to), b = Math.max(from, to);
    from = null; band.remove(); band = null;
    if (keep && b - a > 4) onZoom([toX(a), toX(b)]);
  };
  s.onmouseup = (e) => finish(e, true);
  s.onmouseleave = (e) => finish(e, false);
  s.ondblclick = () => onZoom(null);
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
// The experiments of a chart, in bar order, beside it (scrolls when there are many). Clicking one
// cycles it on every chart: drawn, not drawn, drawn with each of its runs (a tick per run on its
// bars, a dashed line per run on the curves), then drawn again.
function expLegend(experiments, expColor, redraw, lines = false) {
  const l = h("div", "chartlegend");
  for (const exp of experiments) {
    const state = _graph.muted.has(exp) ? "off" : _graph.perRun.has(exp) ? "on" : "";
    const mark = lines ? lineSample(expColor(exp), _graph.lineStyles[exp]) : swatch(expColor(exp));
    const row = h("div", "leg " + state, mark, expLeaf(exp));
    row.title = exp + "\nclick: " + (state === "off" ? "show each run" : state === "on" ? "back to the aggregate" : "hide");
    row.onclick = () => {
      _graph.muted.delete(exp); _graph.perRun.delete(exp);
      if (state === "") _graph.muted.add(exp);
      else if (state === "off") _graph.perRun.add(exp);
      redraw();
    };
    l.append(row);
  }
  return l;
}
function swatch(color) { const s = h("span", "sw"); s.style.background = color; return s; }
function lineSample(color, style) {
  const s = svg("svg", { class: "linesample", width: 22, height: 10, viewBox: "0 0 22 10" });
  const line = svg("line", { x1: 1, y1: 5, x2: 21, y2: 5, stroke: color, "stroke-width": 2 });
  if (LINE_STYLES[style]) line.setAttribute("stroke-dasharray", LINE_STYLES[style]);
  s.append(line);
  return s;
}

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
    const styles = Object.keys(LINE_STYLES);
    const current = styles.includes(_graph.lineStyles[exp]) ? _graph.lineStyles[exp] : "solid";
    const look = h("button", "linebtn", lineSample(expColor(exp), current));
    look.title = `line: ${current} (click for ${styles[(styles.indexOf(current) + 1) % styles.length]})`;
    look.onclick = () => {
      const next = styles[(styles.indexOf(current) + 1) % styles.length];
      if (next === "solid") delete _graph.lineStyles[exp]; else _graph.lineStyles[exp] = next;
      redraw();
    };
    const label = h("span", "leg-label", expLeaf(exp)); label.title = "click to show / hide";
    label.onclick = () => { _graph.hidden.has(exp) ? _graph.hidden.delete(exp) : _graph.hidden.add(exp); redraw(); };
    row.append(up, down, pick, look, label);
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
    const row = h("label", "ctl"); row.htmlFor = id; row.append(cb, h("span", "ctltext", spec.label));
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
  if (SNAPSHOT) _graph.zoom = JSON.parse(JSON.stringify(SNAPSHOT.zoom || {}));
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
    // A redraw empties the panel for an instant, which would let the browser move the page: the
    // panel keeps its height meanwhile and the page, each legend and each chart keep their scroll.
    const pageY = typeof window !== "undefined" && window.scrollY ? window.scrollY : 0;
    charts.style.minHeight = (charts.offsetHeight || 0) + "px";
    const scrolled = new Map();
    for (const card of charts.children)
      scrolled.set(card.dataset.key, [".chartlegend", ".chartscroll"].map((q) => {
        const el = card.querySelector(q); return el ? [el.scrollTop, el.scrollLeft] : [0, 0];
      }));
    clear(charts);
    const activeExps = displayExps.filter((e) => !_graph.hidden.has(e));  // hidden ones drop out
    const g2 = { experiments: activeExps.filter((e) => !_graph.muted.has(e)), tasks: group.tasks, byExp: group.byExp };
    let any = false;
    for (const spec of specs) {
      if (!_graph.active.has(spec.key)) continue;
      any = true;
      const card = h("div", "chartcard", h("h3", null, spec.label));
      card.dataset.key = spec.key;
      const scroll = h("div", "chartscroll");
      const body = h("div", "chartbody", scroll, expLegend(activeExps, expColor, redraw, !!spec.curve));
      if (spec.curve) {
        const zoomed = _graph.zoom[_graph.curveX];
        const setZoom = (span) => { if (span) _graph.zoom[_graph.curveX] = span; else delete _graph.zoom[_graph.curveX]; redraw(); };
        const reset = h("button", "aggbtn", "show all");
        reset.onclick = () => setZoom(null);
        card.append(h("div", "zoomrow",
          segBtns("x:", Object.keys(CURVE_X), _graph.curveX, (n) => { _graph.curveX = n; redraw(); }),
          zoomed ? reset : h("span", "muted zoomhint", "drag across a chart to zoom")));
        let hasRef = false;
        for (const task of g2.tasks) {
          const drawn = progressChart(task, g2, expColor, _graph.agg, _graph.err, _graph.mask, _graph.curveX, setZoom);
          if (!drawn) continue;
          hasRef = hasRef || drawn.hasRef;
          if (g2.tasks.length > 1) scroll.append(h("div", "muted", shortTask(task)));
          scroll.append(drawn.chart);
        }
        card.append(body);
        if (hasRef) card.append(h("div", "muted curve-note", "dashed: reference player (ARC: the human baseline; grid games: optimal solutions)"));
      } else if (spec.levels) {
        for (const task of g2.tasks) {
          const chart = relativeLevelsChart(task, g2, expColor, _graph.agg, _graph.err, _graph.mask);
          if (!chart) continue;
          if (g2.tasks.length > 1) scroll.append(h("div", "muted", shortTask(task)));
          scroll.append(chart);
        }
        card.append(body);
        card.append(h("div", "muted curve-note", "agent / reference player actions on each level (1 = as many, dashed; lower is better); a level aggregates the runs that completed it (hover a point for how many)"));
      } else if (spec.auc) {
        const area = areaSpec(spec, g2, _graph.mask, _graph.curveX, _graph.gamma);
        card.append(h("div", "zoomrow",
          segBtns("x:", Object.keys(CURVE_X), _graph.curveX, (n) => { _graph.curveX = n; redraw(); }),
          gammaControl(redraw)));
        scroll.append(groupedBarChart(area, g2, expColor, _graph.agg, _graph.err, _graph.mask));
        const upTo = [...area.horizons].map(([t, x]) => `${shortTask(t)} ${CURVE_X[_graph.curveX].fmt(Math.round(x))}`).join(", ");
        card.append(body, h("div", "muted curve-note",
          `sum of gamma^x * ${spec.label.replace(/^Area under /, "")} up to where the shortest curve ends (${upTo})` +
          (_graph.curveX === "env actions" ? "; dashed: reference player" : "")));
      } else if (spec.categorical) {
        const { chart, cats } = statusChart(g2, expColor);
        scroll.append(chart);
        card.append(body, legend(cats.map((c) => ({ label: c, color: statusColor(c) }))));
      } else {
        let shown = spec;
        if (_COMPARABLE[spec.key]) {
          card.append(h("div", "zoomrow", segBtns("levels:", ["all", "comparable"], _graph.comparable ? "comparable" : "all",
            (n) => { _graph.comparable = n === "comparable"; redraw(); })));
          if (_graph.comparable) shown = comparableSpec(spec, g2, _graph.mask);
        }
        scroll.append(groupedBarChart(shown, g2, expColor, _graph.agg, _graph.err, _graph.mask));
        card.append(body);
        if (shown.horizons) {
          const cut = [...shown.horizons].filter(([, x]) => Number.isFinite(x));
          const upTo = cut.map(([t, x]) => `${shortTask(t)} ${fmt(Math.round(x))}`).join(", ");
          card.append(h("div", "muted curve-note", cut.length
            ? `levels completed within the env actions where an experiment's unfinished runs end (${upTo}); games where every run won are not cut`
            : "every run won its game: nothing is cut"));
        }
      }
      charts.append(card);
      (scrolled.get(spec.key) || []).forEach(([top, left], i) => {
        const el = card.querySelector([".chartlegend", ".chartscroll"][i]);
        if (el) { el.scrollTop = top; el.scrollLeft = left; }
      });
    }
    if (!any) charts.append(h("div", "muted", "no metric selected - pick some on the left"));
    charts.style.minHeight = "";
    if (pageY) window.scrollTo(0, pageY);
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
  const nav = panelNav("graphs", under);
  if (SNAPSHOT) {
    for (const a of nav.querySelectorAll("a")) a.removeAttribute("href");
    nav.append(h("span", "muted snapnote", `exported ${SNAPSHOT.exported}; navigation is off`));
  } else if (!STATIC) {
    const exp = h("button", "aggbtn exportbtn", "export");
    exp.title = "Save this page, with its data and current settings, as one HTML file";
    exp.onclick = () => exportPage(under, exp);
    nav.append(exp);
  }
  app.append(nav, body);
}

// One self-contained HTML file of the current Graphs page: code, styles, data, settings, images.
async function exportPage(under, button) {
  button.disabled = true;
  button.textContent = "exporting...";
  try {
    const text = async (url) => (await fetch(url, { cache: "no-store" })).text();
    const [js, css] = await Promise.all([text("static/app.js"), text("static/style.css")]);
    const images = {};
    for (const img of document.querySelectorAll("img")) {
      const src = img.getAttribute("src");
      if (!src || !src.startsWith("static/") || src in images) continue;
      const blob = await (await fetch(src)).blob();
      images[src] = await new Promise((done) => {
        const reader = new FileReader();
        reader.onload = () => done(reader.result);
        reader.readAsDataURL(blob);
      });
    }
    const games = "/api/games" + (under ? "?under=" + encodeURIComponent(under) : "");
    const responses = { [games]: _responses[games] };   // this page's data only
    for (const path of ["static/model_defaults.json", "static/viz_icons.json"])
      if (path in _responses) responses[path] = _responses[path];
    responses["/api/settings?scope=" + encodeURIComponent(_scope)] = currentSettings();
    const exported = new Date().toISOString().slice(0, 16).replace("T", " ");
    const snapshot = { under, exported, responses, images, zoom: _graph.zoom };
    const safe = (t) => t.replace(/<\/(script)/gi, "<\\/$1");
    const title = "regact graphs - " + (under.split("/").pop() || "all");
    const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${title.replace(/[<&]/g, "")}</title>
<style>${safe(css)}</style>
</head>
<body>
<header><span id="brand">regact</span><span id="crumb"></span><button id="goup" title="Back to the top of the page">&uarr; go up</button></header>
<main id="app">loading...</main>
<script>window.__REGACT_SNAPSHOT__ = ${JSON.stringify(snapshot).replace(/</g, "\\u003c")};</script>
<script>${safe(js)}</script>
</body>
</html>`;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    link.download = title.replace(/[^A-Za-z0-9._-]+/g, "_") + ".html";
    document.body.append(link);
    link.click();
    link.remove();
  } finally {
    button.disabled = false;
    button.textContent = "export";
  }
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
const TABS = [["", "Overview"], ["conversation", "Conversation"], ["replay", "Replay"], ["artifacts", "Artifacts"], ["logs", "Logs"], ["graphs", "Graphs"]];

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
  const img = h("img", "vicon"); img.src = asset(src); img.width = img.height = size; img.alt = alt || "";
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
  const img = h("img", "taskthumb"); img.src = asset(taskIconSrc(task));
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
          navItems.push({id: block.id, tag: _CWM_TOOL_STYLE[it.tool.framework_tool] || "cwm", label: commandLabel(it.tool), succeeded: it.tool.succeeded});
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

// Every framework command a call ran (a script can chain several), else the one its text names.
function commandLabel(tool) {
  const ran = tool.framework_tools || [];
  return ran.length ? ran.join(" + ") : tool.framework_tool;
}

function toolBlock(tool, gameName) {
  // The reader tags calls authoritatively: blue submit, green submit-that-won a level, red cheat.
  const cls = { cheat: " cheat", submit: " submit", submit_win: " submit-win" }[tool.tag] || "";
  const cwmStyle = _CWM_TOOL_STYLE[tool.framework_tool];
  const box = h("div", "tool" + cls + (cwmStyle ? " " + cwmStyle : ""));
  const t = h("div", "t");
  if (tool.framework_tool) {
    t.append(h("span", "tag tag-" + (cwmStyle || "cwm"), commandLabel(tool)), " ");
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
      img.src = media(`/api/game/tool-image?name=${encodeURIComponent(gameName)}&filename=${encodeURIComponent(attachment.filename)}`);
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
  if (STATIC) return box;   // playback runs the agent's code: local viewer only
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

// ---------------------------------------------------------------- replay tab
// The whole run as a video: the real frame at a position on the chosen x-axis, beside the
// conversation up to that position. /api/game/replay places every frame and conversation item on
// the three axes; items are [turn, item, env_actions, tool_calls, seconds], nondecreasing.
const _REPLAY_AXES = { env_actions: "Env actions", tool_calls: "Tool calls", seconds: "Time" };
const _REPLAY_COLUMN = { env_actions: 2, tool_calls: 3, seconds: 4 };
const _REPLAY_RATES = { env_actions: [1, 2, 4, 8, 16, 32], tool_calls: [0.5, 1, 2, 4, 8], seconds: [10, 30, 60, 120, 300, 600] };
const _REPLAY_TEXT_MAX = 1500;   // agent text and reasoning: a large preview, the rest on click
const _REPLAY_TOOL_MAX = 300;    // a tool call's command and result previews
const _MARKER_COLORS = { progress: "#e0c060", model: "#4ec9a4", model_refused: "#9aa3b2", mismatch: "#e06c6c" };
const _MARKER_LABELS = { progress: "level / progress", model: "model accepted", model_refused: "model refused", mismatch: "prediction mismatch" };

// Number of entries whose value is <= x, values nondecreasing; a null value is always shown.
function replayCount(values, x) {
  let lo = 0, hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] == null || values[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function replayExtent(tl, axis) {
  let max = 0;
  for (const f of tl.frames) if (f[axis] != null) max = Math.max(max, f[axis]);
  for (const row of tl.items) { const v = row[_REPLAY_COLUMN[axis]]; if (v != null) max = Math.max(max, v); }
  return max;
}

function replayPosition(axis, x) {
  if (axis === "seconds") return dur(x);
  return `${fmt(Math.round(x))} ${axis === "env_actions" ? "env actions" : "tool calls"}`;
}

function replayPreview(text, max) {
  const s = String(text || "");
  return s.length > max ? s.slice(0, max) + " …" : s;
}

// Like segBtns, with display labels: options are [value, label] pairs.
function replaySeg(label, options, current, onPick) {
  const seg = h("span", "seg", h("span", "muted", label));
  const btns = [];
  for (const [value, text] of options) {
    const b = h("button", "aggbtn" + (value === current ? " on" : ""), text);
    b.onclick = () => { for (const other of btns) other.classList.toggle("on", other === b); onPick(value); };
    btns.push(b);
    seg.append(b);
  }
  return seg;
}

function replayModal(title, body) {
  const overlay = h("div", "rmodal");
  const box = h("div", "rmodalbox");
  const close = h("button", "rclose", "Close");
  const done = () => { overlay.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") done(); };
  close.onclick = done;
  overlay.onclick = (e) => { if (e.target === overlay) done(); };
  document.addEventListener("keydown", onKey);
  box.append(h("div", "rmodalhead", h("b", null, title), close), body);
  overlay.append(box);
  document.body.append(overlay);
  pageCleanups.push(done);
}

// One conversation item as a compact card; a click shows it in full.
function replayItem(it, gameName) {
  if (it.kind === "tool" && it.tool) {
    const tool = it.tool;
    const style = _CWM_TOOL_STYLE[tool.framework_tool];
    const card = h("div", "ritem rtool" + (style ? " " + style : "") + (tool.tag === "cheat" ? " cheat" : ""));
    const head = h("div", "rhead");
    if (tool.framework_tool) {
      head.append(h("span", "tag tag-" + (style || "cwm"), commandLabel(tool)));
      if (tool.succeeded) head.append(h("span", "tool-success", "✓"));
    } else if (tool.tag) head.append(h("span", "tag tag-" + tool.tag, _TAG_LABEL[tool.tag]));
    head.append(h("b", null, tool.name));
    card.append(head);
    const cmd = shellCommand(tool.input || {});
    const args = cmd !== null ? parseHeredocs(cmd).text
      : typeof tool.input?.file_path === "string" ? tool.input.file_path : JSON.stringify(tool.input || {});
    card.append(h("pre", "rargs", replayPreview(args, _REPLAY_TOOL_MAX)));
    if (tool.result != null) card.append(h("pre", "rres" + (tool.is_error ? " err" : ""), replayPreview(tool.result, _REPLAY_TOOL_MAX)));
    for (const attachment of tool.images || []) {
      if (!attachment.filename) continue;
      const img = h("img", "rthumb"); img.alt = "Image returned to the agent"; img.loading = "lazy";
      img.src = media(`/api/game/tool-image?name=${encodeURIComponent(gameName)}&filename=${encodeURIComponent(attachment.filename)}`);
      card.append(img);
    }
    card.onclick = () => replayModal(tool.name, toolBlock(tool, gameName));
    return card;
  }
  const label = { thinking: "\u{1F4AD} reasoning", system: "⚙ system prompt", user: "\u{1F4E8} sent to the agent" }[it.kind];
  const card = h("div", "ritem r" + it.kind);
  if (label) card.append(h("div", "rhead muted", label));
  const short = it.kind === "system" ? 200 : _REPLAY_TEXT_MAX;
  card.append(h("pre", null, replayPreview(it.text, short)));
  card.onclick = () => replayModal(label || "agent text", h("pre", "rfull", it.text));
  return card;
}

async function renderReplay(name) {
  const d = await gameDetail(name);
  const tl = await api("/api/game/replay?name=" + encodeURIComponent(name));
  await loadPack(name);
  const frameUrl = (f, view) => `/api/game/replay/frame?name=${encodeURIComponent(name)}&obs=${f.obs}&view=${view}`;
  const rows = tl.items.filter(([turn, item]) => d.turns[turn]?.items?.[item]);
  const items = rows.map(([turn, item]) => d.turns[turn].items[item]);
  // A public copy opens on the game's own picture rather than the agent's encoding of it.
  let axis = tl.axes[0], view = STATIC && tl.views.includes("viewer") ? "viewer" : tl.views[0];
  let x = 0, playing = null, carry = 0, placed = false;
  let values = [], frameValues = [];

  const page = h("div", "replay");
  const bar = h("div", "rbar");
  const slider = h("input", "rslider"); slider.type = "range"; slider.min = 0;
  const strip = svg("svg", { class: "rstrip", height: 14 });
  const position = h("span", "rpos");
  const frameBox = h("div", "rframe");
  const mismatchBox = h("div", "rframe rmismatch");
  mismatchBox.hidden = true;
  const conv = h("div", "rconv");
  page.append(bar, h("div", "rslide", slider, strip), h("div", "rmain", h("div", "rleft", frameBox, mismatchBox), conv));

  const playBtn = h("button", "aggbtn", "▶ Play");
  const rate = h("select", "rrate");
  bar.append(replaySeg("x-axis", tl.axes.map((a) => [a, _REPLAY_AXES[a]]), axis, (a) => setAxis(a)), playBtn, rate, position);
  if (tl.views.length > 1) {
    const views = tl.views.map((v) => [v, v === "viewer" ? "Human view" : "Agent view"]);
    bar.append(replaySeg("view", views, view, (v) => { view = v; drawFrame(true); }));
  }
  const legendItems = [...new Set(tl.markers.map((m) => m.kind))].map((k) => ({ label: _MARKER_LABELS[k] || k, color: _MARKER_COLORS[k] || "#9aa3b2" }));
  if (legendItems.length) page.insertBefore(legend(legendItems), page.children[2]);

  // Frame panel
  const img = h("img", "rimg"); img.alt = "Real environment frame";
  const caption = h("div", "muted rcap");
  if (tl.frames.length) frameBox.append(img, caption);
  else frameBox.append(h("div", "muted", "No recorded frames for this run: only runs with an experience store (CWM and vanilla protocols) keep every real step. The conversation still replays."));

  let shownFrame = -1;
  function drawFrame(force = false) {
    if (!tl.frames.length) return;
    const k = Math.max(0, replayCount(frameValues, x) - 1);
    if (k === shownFrame && !force) return;
    shownFrame = k;
    const f = tl.frames[k];
    setMedia(img, frameUrl(f, view));
    for (let j = k + 1; j <= k + 3 && j < tl.frames.length; j++) setMedia(new Image(), frameUrl(tl.frames[j], view));
    const action = f.action == null ? (f.reset ? "reset" : "start") : "action " + (typeof f.action === "object" ? JSON.stringify(f.action) : f.action);
    caption.textContent = `frame ${k + 1}/${tl.frames.length} · ${action} · ${fmt(f.env_actions)} env actions`
      + (tl.thinned ? ` · a long run: 1 recorded frame in ${tl.thinned} is shown` : "");
    clear(mismatchBox);
    mismatchBox.hidden = f.diagnostic == null;
    if (f.diagnostic != null) {
      const pred = h("img", "rimg"); pred.alt = "What the model predicted";
      setMedia(pred, `/api/game/cwm/evidence-image?name=${encodeURIComponent(name)}&kind=diagnostic&identifier=${f.diagnostic}&side=predicted&view=${view}`);
      pred.onerror = () => pred.replaceWith(h("div", "muted", "prediction image unavailable"));
      mismatchBox.append(h("div", "rwarn", "Prediction mismatch: the model predicted this frame instead"), pred);
    }
  }

  // Conversation panel: nodes are built once, on first display, and appended or detached.
  const nodes = [];
  let shownItems = 0;
  function drawConversation() {
    const k = replayCount(values, x);
    while (shownItems < k) {
      if (!nodes[shownItems]) nodes[shownItems] = replayItem(items[shownItems], name);
      conv.append(nodes[shownItems++]);
    }
    while (shownItems > k) nodes[--shownItems].remove();
    for (const node of conv.querySelectorAll(".rnow")) node.classList.remove("rnow");
    const latest = k ? values[k - 1] : null;
    for (let j = k - 1; j >= 0 && values[j] === latest; j--) nodes[j].classList.add("rnow");
    conv.scrollTop = conv.scrollHeight;
  }

  function draw() {
    position.textContent = `${replayPosition(axis, x)} / ${replayPosition(axis, Number(slider.max))}`;
    drawFrame();
    drawConversation();
  }

  function drawStrip() {
    clear(strip);
    const max = Number(slider.max) || 1;
    // Levels get their own lane, so a model update at the same moment cannot cover them.
    const lanes = tl.markers.some((m) => m.kind === "progress") && tl.markers.some((m) => m.kind !== "progress");
    strip.setAttribute("height", lanes ? 22 : 14);
    for (const m of tl.markers) {
      if (m[axis] == null) continue;
      const y = lanes && m.kind !== "progress" ? 12 : 0;
      const tick = svg("rect", { x: `${(m[axis] / max) * 100}%`, y, width: 3, height: lanes ? 10 : 14, fill: _MARKER_COLORS[m.kind] || "#9aa3b2", class: "rtick" });
      tick.append(svg("title", {}, txt(`${m.label} · ${replayPosition(axis, m[axis])}`)));
      tick.onclick = () => { x = m[axis]; slider.value = x; draw(); };
      strip.append(tick);
    }
  }

  function setAxis(a) {
    axis = a;
    values = rows.map((row) => row[_REPLAY_COLUMN[axis]]);
    frameValues = tl.frames.map((f) => f[axis]);
    const max = replayExtent(tl, axis);
    slider.max = max;
    slider.step = axis === "seconds" ? Math.max(1, max / 2000) : 1;
    // Keep the place: the frame shown stays the frame shown.
    const f = tl.frames[shownFrame];
    x = placed && f && f[axis] != null ? f[axis] : Math.min(x, max);
    placed = true;
    slider.value = x;
    clear(rate);
    for (const r of _REPLAY_RATES[axis]) {
      const o = h("option", null, axis === "seconds" ? `${r}x` : `${r} / s`); o.value = r; rate.append(o);
    }
    rate.value = _REPLAY_RATES[axis][2];
    drawStrip();
    draw();
  }

  slider.oninput = () => { x = Number(slider.value); draw(); };
  function stop() { if (playing) clearInterval(playing); playing = null; playBtn.textContent = "▶ Play"; }
  playBtn.onclick = () => {
    if (playing) return stop();
    if (x >= Number(slider.max)) x = 0;
    playBtn.textContent = "⏸ Pause";
    carry = 0;
    playing = setInterval(() => {
      if (!page.isConnected) return stop();
      carry += Number(rate.value) / 10;
      const step = axis === "seconds" ? carry : Math.floor(carry);
      if (!step) return;
      carry -= step;
      x = Math.min(Number(slider.max), x + step);
      slider.value = x;
      draw();
      if (x >= Number(slider.max)) stop();
    }, 100);
  };
  pageCleanups.push(stop);

  shell(name, "replay", page);
  if (!items.length) conv.append(h("div", "muted", "no transcript"));
  setAxis(axis);
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
      vid.src = media(`/video?game=${encodeURIComponent(name)}&submission=${encodeURIComponent(s.name)}&filename=${encodeURIComponent(v)}`);
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


// ---------------------------------------------------------------- public landing (static mode)
// The curated benchmarks (experiments_public/benchmarks.yaml), by problem, then one row per
// (agent, model, protocol): its runs, solved runs and mean main metric.
const _AGENT_NAMES = { claude: "Claude Code", codex: "Codex", alan: "Alan Code", agy: "Antigravity" };
async function renderBenchmarks() {
  const token = navigation;
  const r = await fetch(STATIC + "site.json", {signal: token.signal});
  const site = await r.json();
  await loadIcons();
  checkNavigation(token);
  crumb.textContent = "";
  clear(app);
  app.append(h("h1", "benchtitle", site.title));
  if (site.intro) app.append(h("p", "benchintro", site.intro));
  for (const problem of site.problems) {
    app.append(h("h2", "expsection", problem.name));
    if (problem.description) app.append(h("p", "muted", problem.description));
    for (const bench of problem.benchmarks) {
      const graphs = h("a", "aggbtn", "graphs");
      graphs.href = "#graphs/" + encodeURIComponent(bench.under);
      const runs = h("a", "aggbtn", "runs");
      runs.href = "#run/" + encodeURIComponent(bench.under);
      const card = h("div", "card benchcard", h("div", "benchhead", h("h3", null, bench.title), graphs, runs));
      if (bench.note) card.append(h("p", "muted", bench.note));
      const table = h("table", "benchtable");
      table.append(rowEl("th", ["agent", "model", "protocol", "runs", "solved", "mean"]));
      for (const row of bench.rows) {
        const mean = row.mean == null ? "\u2014" : `${row.metric} ${fmtMetric(row.mean)}`;
        table.append(h("tr", null,
          h("td", null, agentIcon(row.agent, 20), " " + (_AGENT_NAMES[row.agent] || row.agent)),
          h("td", null, modelLabel(row.model)),
          h("td", null, row.protocol), h("td", null, fmt(row.runs)), h("td", null, fmt(row.solved)),
          h("td", null, mean)));
      }
      card.append(table);
      app.append(card);
    }
  }
  const all = h("a", null, "browse every published run");
  all.href = "#browse";
  app.append(h("p", "muted", all));
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
    if (SNAPSHOT) return await renderGraphs(SNAPSHOT.under);
    const parts = (location.hash || "").replace(/^#\/?/, "").split("/").filter(Boolean);
    if (!parts.length) return STATIC ? await renderBenchmarks() : await renderBrowse();
    if (parts[0] === "browse") return await renderBrowse();
    if (parts[0] === "graphs") return await renderGraphs(parts[1] ? decodeURIComponent(parts[1]) : "");
    if (parts[0] === "run") return await renderDashboard(parts[1] ? decodeURIComponent(parts[1]) : "");
    if (parts[0] !== "game" || !parts[1]) return await renderBrowse();
    const name = decodeURIComponent(parts[1]);
    await gameDetail(name);
    const tab = parts[2] || "";
    if (tab === "conversation") await renderConversation(name);
    else if (tab === "replay") await renderReplay(name);
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
