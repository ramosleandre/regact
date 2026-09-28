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
    const pending = api('/api/game');
    navigation.abort(); navigation = new AbortController();
    try { await pending; throw Error('stale response accepted'); }
    catch (e) { if (e.name !== 'AbortError') throw e; }
  })()`, sandbox);
})().catch(e => { console.error(e); process.exitCode=1; });
"""
    subprocess.run([node, "-e", script, str(source)], check=True, timeout=10)
