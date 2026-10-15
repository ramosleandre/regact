"""Small browser-independent checks; real rendering is covered by manual viewer QA."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_heredocs_and_cancelled_navigation():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    source = Path(__file__).parents[1] / "src/regact/viz/static/app.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const sandbox = {
  document: {getElementById: () => ({})},
  window: {addEventListener: () => {}},
  AbortController, DOMException, Intl, URL,
  fetch: async () => ({ok:true, json:async () => ({value:1})}),
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/route\(\);\s*$/, ''), sandbox);
(async () => {
  await vm.runInContext(String.raw`(async () => {
    const stdin = parseHeredocs("python - <<'PY'\nprint(1 << 2)\nPY");
    if (stdin.text !== 'python - [stdin]' || stdin.files[0].name !== null)
      throw Error('stdin label');
    const file = parseHeredocs('cat > code.py <<EOF\nprint(1)\nEOF');
    if (file.files[0].name !== 'code.py') throw Error('file label');
    const original = 'python - <<EOF\nno closing marker';
    if (parseHeredocs(original).text !== original) throw Error('unclosed heredoc');
    const replay = controllerResultIds('prefix\n{"status":"Completed","exploration_id":2,"current_observation_id":5}\nPhase transition');
    if (replay.length !== 1 || replay[0] !== 2) throw Error('managed replay missing');
    if (controllerResultIds('{"status":"Completed","exploration_id":2}').length)
      throw Error('historical submission has no per-call recording');
    const pending = api('/api/game');
    navigation.abort(); navigation = new AbortController();
    try { await pending; throw Error('stale response accepted'); }
    catch (e) { if (e.name !== 'AbortError') throw e; }
  })()`, sandbox);
})().catch(e => { console.error(e); process.exitCode=1; });
"""
    subprocess.run([node, "-e", script, str(source)], check=True, timeout=10)


def test_main_metrics_follow_the_problem():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    source = Path(__file__).parents[1] / "src/regact/viz/static/app.js"
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const sandbox = {document: {getElementById: () => ({})}, window: {addEventListener: () => {}},
  AbortController, DOMException, Intl, URL, fetch: async () => ({ok: true, json: async () => ({})})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/route\(\);\s*$/, ''), sandbox);
vm.runInContext(String.raw`
  const arc = {metrics: {main_metrics: ['mean_levels_completed', 'rhae'], success_rate: null,
    final_aggregate: {mean_levels_completed: 3, win_rate: 0}, derived_metrics: {rhae: 0.1}}};
  const specs = metricSpecs([arc]);
  const main = specs.filter((s) => s.main).map((s) => s.key);
  if (specs.some((s) => s.key === 'success_rate')) throw Error('ARC offered success_rate');
  for (const k of ['agg:mean_levels_completed', 'derived:rhae', 'time'])
    if (!main.includes(k)) throw Error('missing main ' + k);
  if (main.includes('agg:win_rate')) throw Error('win_rate not listed by this problem');
  const grid = {metrics: {main_metrics: ['success_rate'], success_rate: 1, final_aggregate: {}}};
  if (!metricSpecs([grid]).find((s) => s.key === 'success_rate').main) throw Error('MiniGrid success');
  if (FRAMEWORK_METRICS.some((s) => 'main' in s)) throw Error('shared spec mutated');
  const ended = (exit_reason, last_error_category) => isValidRun({exit_reason, last_error_category});
  if (!ended('walltime_limit', 'agent_api')) throw Error('a run that recovered and finished is masked');
  if (!ended('solved') || !ended('no_tool_progress')) throw Error('a finished run is masked');
  for (const crash of ['agent_api', 'loop_crash', 'interrupted', 'usage_limit', null])
    if (ended(crash)) throw Error('a crashed or unfinished run is kept: ' + crash);
`, sandbox);
"""
    subprocess.run([node, "-e", script, str(source)], check=True, timeout=10)


def test_progress_curves_aggregate_runs_and_stop_counting_unfinished_ones():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    source = Path(__file__).parents[1] / "src/regact/viz/static/app.js"
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const sandbox = {document: {getElementById: () => ({})}, window: {addEventListener: () => {}},
  AbortController, DOMException, Intl, URL, fetch: async () => ({ok: true, json: async () => ({})})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/route\(\);\s*$/, ''), sandbox);
vm.runInContext(String.raw`
  const run = (points, env_moves, exit_reason) => ({exit_reason, env_moves, n_tool_calls: 9,
    duration_s: 9, progress: {label: 'Levels completed', reference: [], points}});
  const pt = (value, env_actions, tool_calls) => ({value, env_actions, tool_calls, seconds: 1});
  const finished = run([pt(1, 10, 2), pt(2, 30, 5)], 40, 'solved');
  const killed = run([pt(1, 20, 3)], 25, null);
  const old = run([pt(1, 20, null)], 25, 'solved');
  const games = [finished, killed, old].map((metrics) => ({experiment: 'e', task: 't', metrics}));
  const spec = metricSpecs(games).find((s) => s.key === 'progress');
  if (!spec || !spec.curve || !spec.main) throw Error('no progress curve offered');
  const group = groupByExpTask(games);
  const byActions = curveRuns('e', 't', group, false, CURVE_X['env actions']);
  if (byActions.length !== 3) throw Error('all runs have env actions');
  if (curveRuns('e', 't', group, true, CURVE_X['env actions']).length !== 2) throw Error('mask');
  const byCalls = curveRuns('e', 't', group, false, CURVE_X['tool calls']);
  if (byCalls.length !== 2) throw Error('a run without tool calls recorded');
  const [f, k] = byActions;
  if (curveAt(f, 9) !== 0 || curveAt(f, 10) !== 1 || curveAt(f, 500) !== 2) throw Error('steps');
  if (!f.done || k.done || k.end !== 25 || f.end !== 40) throw Error('where a run stops');
`, sandbox);
"""
    subprocess.run([node, "-e", script, str(source)], check=True, timeout=10)


def test_progress_chart_geometry():
    """Every line of the chart sits at its own x: the reference player on the reference's points
    whatever runs are shown, a run on its own points, and each line ends where it should."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    source = Path(__file__).parents[1] / "src/regact/viz/static/app.js"
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const el = (tag) => ({tag, attrs: {}, kids: [], style: {}, dataset: {},
  setAttribute(k, v) { this.attrs[k] = String(v); }, append(...k) { this.kids.push(...k); }});
const sandbox = {document: {getElementById: () => ({}), createElementNS: (_, t) => el(t),
    createElement: (t) => el(t), createTextNode: (t) => ({text: t})},
  window: {addEventListener: () => {}, innerWidth: 1500},
  AbortController, DOMException, Intl, URL, fetch: async () => ({ok: true, json: async () => ({})})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/route\(\);\s*$/, ''), sandbox);
vm.runInContext(String.raw`
  const reference = [[22, 1], [145, 2], [218, 3]];
  const run = (points, env_moves) => ({exit_reason: 'solved', env_moves, n_tool_calls: 50,
    duration_s: 500, progress: {label: 'Levels completed', reference, points}});
  const pt = (value, env_actions) => ({value, env_actions, tool_calls: value * 10, seconds: value * 100});
  const a1 = run([pt(1, 100), pt(2, 300)], 400), a2 = run([pt(1, 60)], 200), b1 = run([pt(1, 37)], 1000);
  const game = (experiment, metrics) => ({experiment, task: 't', metrics});
  const draw = (games, xName, perRun) => {
    _graph.perRun = new Set(perRun || []);
    const chart = progressChart('t', groupByExpTask(games), () => '#fff', 'mean', 'none', false, xName).chart;
    const by = (cls) => chart.kids.filter((k) => k.attrs && k.attrs.class === cls);
    const W = Number(chart.attrs.width), xMax = Math.max(...games.map((g) => g.metrics.env_moves));
    return {chart, by, x: (v) => 44 + (W - 60) * (v / xMax)};
  };
  const near = (a, b) => Math.abs(a - b) < 0.01;
  const xs = (d) => [...d.matchAll(/[MH]([0-9.]+)/g)].map((m) => Number(m[1]));
  const has = (d, x) => xs(d).some((v) => near(v, x));

  // The reference steps at 22, 145 and 218 actions, with or without another experiment shown.
  for (const games of [[game('A', a1), game('A', a2)], [game('A', a1), game('A', a2), game('B', b1)]]) {
    const {by, x} = draw(games, 'env actions');
    const d = by('curve-ref')[0].attrs.d;
    for (const at of [22, 145, 218]) if (!has(d, x(at))) throw Error('reference step missing at ' + at);
    if (xs(d).length !== 5) throw Error('reference has extra steps: ' + d);
  }

  // The aggregate of A steps at its runs' points (60, 100, 300) and ends at its last run (400),
  // also when B runs on to 1000; a run's own line ends where that run stopped.
  const {by, x} = draw([game('A', a1), game('A', a2), game('B', b1)], 'env actions', ['A']);
  const lineA = by('curve-line')[0].attrs.d;
  for (const at of [60, 100, 300]) if (!has(lineA, x(at))) throw Error('aggregate step missing at ' + at);
  if (Math.max(...xs(lineA)) > x(400) + 1) throw Error('aggregate runs past its last run');
  const ends = by('curve-run').map((p) => Math.max(...xs(p.attrs.d))).sort((p, q) => p - q);
  if (ends.length !== 2 || ends[0] > x(200) + 1 || ends[0] < x(200) - 1 || ends[1] > x(400) + 1)
    throw Error('run lines end at ' + ends);
  const dots = by('curve-end').map((c) => Number(c.attrs.cx));
  if (!near(dots[0], x(400)) || !near(dots[1], x(1000))) throw Error('aggregate dots at ' + dots);

  // Same height scale whichever x is chosen, and no reference outside env actions.
  const heights = ['env actions', 'tool calls', 'time'].map((name) => {
    const drawn = draw([game('A', a1)], name);
    if ((name === 'env actions') !== (drawn.by('curve-ref').length === 1)) throw Error('reference on ' + name);
    return drawn.by('ylab').length;
  });
  if (new Set(heights).size !== 1 || heights[0] !== 4) throw Error('y axis differs: ' + heights);

  // A zoom on [100, 300] maps those x to the plot's edges, labels that range, and clips every line.
  _graph.zoom = {'env actions': [100, 300]};
  const zoomed = draw([game('A', a1), game('A', a2)], 'env actions', ['A']);
  const W = Number(zoomed.chart.attrs.width), z = (v) => 44 + (W - 60) * (v - 100) / 200;
  const labels = zoomed.by('xlab').map((t) => t.kids[0].text);
  if (labels[0] !== '100' || labels[4] !== '300') throw Error('zoomed labels ' + labels);
  if (!has(zoomed.by('curve-line')[0].attrs.d, z(300))) throw Error('zoomed step at 300 misplaced');
  const lines = ['curve-line', 'curve-run', 'curve-ref', 'curve-end'].flatMap((c) => zoomed.by(c));
  if (!lines.length || lines.some((k) => !String(k.attrs['clip-path']).startsWith('url(#'))) throw Error('unclipped line');
  _graph.zoom = {};
`, sandbox);
"""
    subprocess.run([node, "-e", script, str(source)], check=True, timeout=10)


def test_shell_commands_render_as_text_with_their_extra_arguments():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    source = Path(__file__).parents[1] / "src/regact/viz/static/app.js"
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const sandbox = {document: {getElementById: () => ({})}, window: {addEventListener: () => {}},
  AbortController, DOMException, Intl, URL, fetch: async () => ({ok: true, json: async () => ({})})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/route\(\);\s*$/, ''), sandbox);
vm.runInContext(String.raw`
  if (shellCommand({command: 'ls -la'}) !== 'ls -la') throw Error('string command');
  if (shellCommand({command: ['bash', '-lc', 'ls']}) !== 'bash -lc ls') throw Error('argv command');
  if (shellCommand({file_path: 'x'}) !== null) throw Error('not a shell call');
  globalThis.h = (tag, cls, ...kids) => ({tag, cls, kids});
  const notes = argNotes({command: 'ls', description: 'List files', timeout: 1800000}, ['command']);
  if (!notes.kids[0].includes('description: List files') || !notes.kids[0].includes('timeout: 1800000'))
    throw Error('extra arguments are shown as notes');
  if (argNotes({command: 'ls'}, ['command']) !== null) throw Error('no notes when nothing else');
`, sandbox);
"""
    subprocess.run([node, "-e", script, str(source)], check=True, timeout=10)
