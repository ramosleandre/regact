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
  if (!f.done || k.done || k.end !== 25) throw Error('where a run stops');
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
