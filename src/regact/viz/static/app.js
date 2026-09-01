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
// Game-agnostic: format an opaque aggregate dict as "key val · key val", skipping
// bookkeeping keys. Each game owns its metric names (ARC: levels/rhae, MiniGrid: reward).
const _SKIP_KEYS = new Set(["n_episodes", "n_errors"]);
const fmtMetric = (v) => (v == null ? "—" : typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : v);
const aggLine = (agg) =>
  Object.entries(agg || {})
    .filter(([k, v]) => !_SKIP_KEYS.has(k) && typeof v === "number")
    .map(([k, v]) => `${k} ${fmtMetric(v)}`)
    .join(" · ") || "—";

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
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
  const card = h("div", "card click", h("h3", null, g.task || g.name));
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
  crumb.textContent = "";
  const data = await gamesData(under);
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
    app.append(h("h2", "expsection", expLeaf(exp), h("span", "muted", ` · ${games.length} run(s)`)));
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
  crumb.textContent = "";
  const { root, tree } = await api("/api/tree");
  crumb.textContent = root;
  if (!tree.length) { clear(app); app.append(h("div", "muted", "no runs under this folder")); return; }
  if (_rootIsSingleScope(tree)) { renderDashboard(""); return; }
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
// default in the Graphs panel (the two are kept in sync - see MAIN_METRIC_KEYS + renderOverview).
const FRAMEWORK_METRICS = [
  { key: "success_rate", label: "success rate", get: (m) => m.success_rate, fmt: pct, def: true },
  { key: "env_actions", label: "env actions", get: (m) => m.env_moves, def: true },
  { key: "time", label: "time", get: (m) => m.duration_s, fmt: dur, def: true },
  { key: "tool_calls", label: "tool calls", get: (m) => m.n_tool_calls, def: true },
  { key: "flagged_calls", label: "flagged calls", get: (m) => m.flagged_tool_calls, def: true },
  { key: "n_runs", label: "number of runs", count: true },  // runs of this task (not aggregated)
  { key: "iterations", label: "iterations", get: (m) => m.n_turns },
  { key: "output_tokens", label: "output tokens", get: (m) => m.tokens && m.tokens.output },
  { key: "cache_read", label: "cache tokens", get: (m) => m.tokens && m.tokens.cache_read },
  { key: "thinking_chars", label: "thinking chars", get: (m) => m.thinking_chars },
];
const STATUS_METRIC = { key: "status", label: "status", categorical: true };
const AGGREGATORS = {
  mean: (xs) => xs.reduce((a, b) => a + b, 0) / xs.length,
  median: (xs) => { const s = [...xs].sort((a, b) => a - b), i = s.length >> 1; return s.length % 2 ? s[i] : (s[i - 1] + s[i]) / 2; },
  min: (xs) => Math.min(...xs),
  max: (xs) => Math.max(...xs),
};
// A run "finished" iff it ended cleanly: it has an exit_reason and no error category. Crashed /
// unfinished runs (agent_api, eval_harness, loop_crash, still-running…) are dropped when masking.
const isValidRun = (m) => !m.last_error_category && !!m.exit_reason;
const _COLORS_KEY = "regact_viz_expcolors";
const _loadColors = () => { try { return JSON.parse(localStorage.getItem(_COLORS_KEY)) || {}; } catch (e) { return {}; } };
const _saveColors = () => { try { localStorage.setItem(_COLORS_KEY, JSON.stringify(_graph.colors)); } catch (e) { /* ignore */ } };
// Persist the panel's controls across redraws: metric toggles, aggregate + error-bar method, the
// crashed-run mask, which experiments are hidden, and per-experiment color overrides (localStorage).
const _graph = {
  agg: "mean", err: "none", mask: false,
  // Defaults span both problem families; a metric a game never reports draws no bar. success_rate
  // is MiniGrid's; mean_levels_completion_rate is ARC's graded headline (its success_rate role).
  active: new Set(["success_rate", "time", "agg:mean_levels_completion_rate", "agg:mean_levels_completed"]),
  hidden: new Set(), colors: _loadColors(),
};

const expLeaf = (e) => String(e).split("/").pop();
const shortTask = (t) => String(t).replace(/^MiniGrid-/, "").replace(/-v\d+$/, "");
const statusColor = (s) => (s === "agent_exit" ? _THEME.good : /limit/.test(s) ? _THEME.warn : s === "running" ? _THEME.muted : _THEME.bad);
const txt = (s) => document.createTextNode(s);
function svg(tag, attrs, ...kids) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  for (const k of kids) if (k) e.append(k);
  return e;
}

function metricSpecs(games) {
  const specs = [...FRAMEWORK_METRICS];
  const seen = new Set();
  for (const g of games)
    for (const [k, v] of Object.entries(g.metrics.final_aggregate || {}))
      if (typeof v === "number" && !_AGG_SKIP.has(k) && !seen.has(k)) {
        seen.add(k);
        specs.push({ key: "agg:" + k, label: k, get: (m) => m.final_aggregate && m.final_aggregate[k] });
      }
  specs.push(STATUS_METRIC);
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
  const idx = new Map(experiments.map((e, i) => [e, i]));  // stable index -> stable default color
  return (e) => _graph.colors[e] || _EXP_PALETTE[(idx.get(e) || 0) % _EXP_PALETTE.length];
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

// A small task-preview thumbnail under each x-axis label (rendered on demand by the server at
// /api/task_preview), so a task is recognizable at a glance. `centerXOf(ti)` is the task's group
// center; the image removes itself if the server can't render that task.
const _THUMB_GAP = 6;
function appendTaskThumbs(s, tasks, centerXOf, yTop, size) {
  tasks.forEach((task, ti) => {
    const img = svg("image", {
      x: centerXOf(ti) - size / 2, y: yTop, width: size, height: size,
      href: "api/task_preview?task=" + encodeURIComponent(task),
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
  let max = 0;
  for (const t of tasks) for (const e of experiments) {
    const v = val(e, t); if (v != null) max = Math.max(max, v);
    const eb = errOf(e, t); if (eb) max = Math.max(max, eb.hi);   // keep error bars in view
  }
  max = max || 1;
  const { padL, padB, padR, padT, H } = _chartPad(tasks);
  const groupW = Math.max(44, experiments.length * 20 + 16);
  const W = padL + padR + tasks.length * groupW;
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
  tasks.forEach((task, ti) => {
    const gx = padL + ti * groupW + 8;
    const bw = (groupW - 16) / experiments.length;   // one slot per experiment (task spacing = groupW)
    const barW = Math.max(1, bw / 2);                // bars are half the slot -> clearer separation
    const base = padT + plotH;                       // y of the zero baseline
    experiments.forEach((exp, ei) => {
      const v = val(exp, task);
      if (v == null) return;
      const cx = gx + ei * bw + bw / 2;              // bar centered in its slot
      const barTop = Math.min(yOf(v), base - 2);     // a value of 0 still shows a >=2px line
      const rect = svg("rect", { x: cx - barW / 2, y: barTop, width: barW, height: base - barTop, rx: 2, fill: expColor(exp) });
      rect.append(svg("title", {}, txt(`${expLeaf(exp)} · ${task}\n${spec.label}: ${spec.fmt ? spec.fmt(v) : fmtMetric(v)}`)));
      s.append(rect);
      const eb = errOf(exp, task);               // error/interval bar centered on the bar
      if (eb) {
        const yl = yOf(eb.lo), yh = yOf(eb.hi), cap = Math.max(2, barW * 0.44);
        s.append(svg("line", { class: "errbar", x1: cx, y1: yh, x2: cx, y2: yl }));
        s.append(svg("line", { class: "errbar", x1: cx - cap, y1: yh, x2: cx + cap, y2: yh }));
        s.append(svg("line", { class: "errbar", x1: cx - cap, y1: yl, x2: cx + cap, y2: yl }));
      }
    });
    const lx = gx + (groupW - 16) / 2, ly = H - padB + 12;   // x label (task), rotated
    s.append(svg("text", { class: "xlab", x: lx, y: ly, "text-anchor": "end", transform: `rotate(-35 ${lx} ${ly})` }, txt(shortTask(task))));
  });
  appendTaskThumbs(s, tasks, (ti) => padL + ti * groupW + 8 + (groupW - 16) / 2, yThumb, thumb);
  return s;
}

// Status is categorical: per (task, experiment) a full-height bar stacked by exit-reason share,
// colored by status; a thin underline ties each bar back to its experiment color.
function statusChart(group, expColor) {
  const { experiments, tasks, byExp } = group;
  const statusOfM = (m) => m.exit_reason || m.last_error_category || "running";
  const cats = new Set();
  for (const e of experiments) for (const t of tasks) for (const m of (byExp.get(e) && byExp.get(e).get(t)) || []) cats.add(statusOfM(m));
  const { padL, padB, padR, padT, H } = _chartPad(tasks);
  const groupW = Math.max(44, experiments.length * 20 + 16);
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

// The experiment legend is interactive: click the label to show/hide that experiment; click the
// swatch (a native color input) to recolor it - the color persists to localStorage.
function expLegend(experiments, expColor, redraw) {
  const l = h("div", "legend");
  for (const exp of experiments) {
    const item = h("span", "leg exp-leg" + (_graph.hidden.has(exp) ? " off" : ""));
    const pick = h("input"); pick.type = "color"; pick.className = "sw-pick";
    pick.value = expColor(exp); pick.title = "pick colour";
    pick.oninput = () => { _graph.colors[exp] = pick.value; _saveColors(); redraw(); };
    const label = h("span", "leg-label", expLeaf(exp)); label.title = "click to show / hide";
    label.onclick = () => { _graph.hidden.has(exp) ? _graph.hidden.delete(exp) : _graph.hidden.add(exp); redraw(); };
    item.append(pick, label);
    l.append(item);
  }
  return l;
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
  return h("div", "controlbar", agg, err, mask);
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

// Build the panel: agg selector + charts (left) + metric checklist (right). `onlyExp` restricts to
// one experiment (the per-game tab); null = every experiment (top-level Graphs).
function graphsView(games, onlyExp) {
  const shown = onlyExp ? games.filter((g) => g.experiment === onlyExp) : games;
  const specs = metricSpecs(shown);
  // Seed defaults on first open (only keep keys that still exist).
  for (const spec of specs) if (spec.def && !_graph._seeded) _graph.active.add(spec.key);
  _graph._seeded = true;
  const group = groupByExpTask(shown);
  const expColor = makeExpColor(group.experiments);   // stable colors over ALL experiments
  const charts = h("div", "charts");
  const redraw = () => {
    clear(charts);
    // Only render experiments not hidden in the legend; the legend still lists them all.
    const activeExps = group.experiments.filter((e) => !_graph.hidden.has(e));
    const g2 = { experiments: activeExps, tasks: group.tasks, byExp: group.byExp };
    charts.append(expLegend(group.experiments, expColor, redraw));
    let any = false;
    for (const spec of specs) {
      if (!_graph.active.has(spec.key)) continue;
      any = true;
      const card = h("div", "chartcard", h("h3", null, spec.label));
      const scroll = h("div", "chartscroll");
      if (spec.categorical) {
        const { chart, cats } = statusChart(g2, expColor);
        scroll.append(chart);
        card.append(scroll, legend(cats.map((c) => ({ label: c, color: statusColor(c) }))));
      } else {
        scroll.append(groupedBarChart(spec, g2, expColor, _graph.agg, _graph.err, _graph.mask));
        card.append(scroll);
      }
      charts.append(card);
    }
    if (!any) charts.append(h("div", "muted", "no metric selected — pick some on the right"));
  };
  const layout = h("div", "graphlayout", charts, metricControls(specs, redraw));
  const wrap = h("div", "graphs", controlBar(redraw), layout);
  redraw();
  return wrap;
}

async function renderGraphs(under = "") {
  crumb.textContent = "graphs";
  const data = await gamesData(under);
  clear(app);
  app.append(panelNav("graphs", under), graphsView(data.games, null));
}

// The per-game "Graphs" tab: the same charts but for THIS run's experiment only (scoped to the
// experiment subtree - the game path minus its /timestamp/task - so it never parses the whole root).
async function renderGameGraphs(name) {
  const data = await gamesData(name.split("/").slice(0, -2).join("/"));
  const me = data.games.find((g) => g.name === name);
  const exp = me ? me.experiment : null;
  const body = h("div");
  body.append(h("div", "gnote muted",
    exp ? `Aggregated over every run in this experiment (${expLeaf(exp)}).` : "unknown experiment"));
  body.append(graphsView(data.games, exp));
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

async function renderOverview(name) {
  const d = await gameDetail(name);
  const m = d.metrics;
  const wrap = h("div");
  const unverified = m.final_aggregate_unverified || {};
  const hasUnverified = Object.keys(unverified).length > 0;
  // Main = status + the shadow-replay-verified problem score + the run's headline effort/cost.
  // Other = the un-replayed (controller-reported) score + secondary telemetry.
  const main = [
    ["Status", statusOf(m)],
    ["Score", aggLine(m.final_aggregate), hasUnverified ? "shadow-replay verified" : "final submission"],
    ["Env actions", fmt(m.env_moves)],
    ["Time", dur(m.duration_s)],
    ["Tool calls", m.n_tool_calls],
    ["Flagged calls", m.flagged_tool_calls ?? 0],
  ];
  const other = [
    hasUnverified ? ["Score · no replay", aggLine(unverified), "controller-reported"] : null,
    ["Iterations", m.n_turns, "agent turns"],
    ["Thinking", fmt(m.thinking_chars) + " ch"],
    ["Submissions", m.n_submissions],
    ["Output tokens", fmt(m.tokens.output)],
    ["Cache tokens", fmt(m.tokens.cache_read)],
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
// tables. Nested objects flatten to dotted keys (so `features.cwm.*` shows, never "[object Object]").
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

const _TOOL_TEXT_MAX = 4000;  // char cap for both call args and result body (then the box scrolls)
const _INLINE_ARG_MAX = 240;  // short args stay inline; longer ones get the scrollable block

function toolBlock(tool) {
  // The reader tags calls authoritatively: blue submit, green submit-that-won a level, red cheat.
  const cls = { cheat: " cheat", submit: " submit", submit_win: " submit-win" }[tool.tag] || "";
  const box = h("div", "tool" + cls);
  const inp = JSON.stringify(tool.input);
  const t = h("div", "t");
  if (tool.tag) t.append(h("span", "tag tag-" + tool.tag, _TAG_LABEL[tool.tag]), " ");
  t.append(h("b", null, tool.name), " ");
  // Long call args get the SAME treatment as a long result: a scrollable block, never a "…" cut,
  // so a full command / Write / Edit payload is readable. Short args stay compact inline.
  if (inp.length > _INLINE_ARG_MAX)
    t.append(h("pre", "args", inp.slice(0, _TOOL_TEXT_MAX)));
  else
    t.append(h("span", "muted", inp));
  box.append(t);
  if (tool.result != null) {
    const res = h("div", "res" + (tool.is_error ? " err" : ""));
    res.append(h("pre", null, String(tool.result).slice(0, _TOOL_TEXT_MAX)));
    box.append(res);
  }
  return box;
}

// ---------------------------------------------------------------- artifacts tab
async function renderArtifacts(name) {
  const d = await api("/api/game/artifacts?name=" + encodeURIComponent(name));
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
    c.append(h("div", "muted", `${aggLine(a)} · n=${a.n_episodes ?? "—"}`));
    for (const v of s.videos || []) {
      const vid = h("video"); vid.controls = true; vid.preload = "metadata";
      vid.src = `/video?game=${encodeURIComponent(name)}&submission=${encodeURIComponent(s.name)}&filename=${encodeURIComponent(v)}`;
      c.append(vid);
    }
    subs.append(c);
  }
  shell(name, "artifacts", h("div", null, wrap, subs));
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
  try {
    const parts = (location.hash || "").replace(/^#\/?/, "").split("/").filter(Boolean);
    if (!parts.length) return renderBrowse();
    if (parts[0] === "graphs") return renderGraphs(parts[1] ? decodeURIComponent(parts[1]) : "");
    if (parts[0] === "run") return renderDashboard(parts[1] ? decodeURIComponent(parts[1]) : "");
    if (parts[0] !== "game" || !parts[1]) return renderBrowse();
    const name = decodeURIComponent(parts[1]);
    const tab = parts[2] || "";
    if (tab === "conversation") await renderConversation(name);
    else if (tab === "artifacts") await renderArtifacts(name);
    else if (tab === "logs") await renderLogs(name);
    else if (tab === "graphs") await renderGameGraphs(name);
    else await renderOverview(name);
  } catch (e) {
    clear(app); app.append(h("pre", "err", "error: " + e.message));
  }
}
window.addEventListener("hashchange", route);
route();
