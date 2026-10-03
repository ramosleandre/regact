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
