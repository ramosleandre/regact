'use strict';
// node puzzlescript_bfs.js <game source> <engine level index> <budget ms>
// node puzzlescript_bfs.js <game source> info
// -> one JSON line: a shortest plan (0 up, 1 left, 2 down, 3 right, 4 action) or null.
// Breadth-first over full level states; `exhausted` means no win is reachable at all.
// Run with PS_SRC set to the engine's src directory.
const fs = require('fs');
const path = require('path');
const ps = require(path.join(__dirname, '../../src/regact/problems/grid_games/puzzlescript_node/psnode.js')).load();

const [file, levelIdx, budget] = [process.argv[2], +process.argv[3], +process.argv[4]];
const text = fs.readFileSync(file, 'utf8');
ps.compileOnly(text);
const dirs = ps.state().metadata.noaction !== undefined ? [0, 1, 2, 3] : [0, 1, 2, 3, 4];
if (process.argv[3] === 'info') {  // the playable (non-message) level indices and the action set
  const playable = ps.state().levels.map((l, i) => (l.message === undefined ? i : -1)).filter(i => i >= 0);
  console.log(JSON.stringify({ playable, action: dirs.includes(4) }));
  process.exit(0);
}
ps.compileGame(text, levelIdx, 'regact');
const key = b => Buffer.from(Int32Array.from(b.dat).buffer).toString('base64');
const root = ps.backup();
const seen = new Map([[key(root), null]]);  // state -> [parent state, dir]
let frontier = [root], plan = null, timeout = false;
const t0 = Date.now();
outer: while (frontier.length) {
  const next = [];
  for (const node of frontier) {
    if (Date.now() - t0 > budget) { timeout = true; break outer; }
    const k = key(node);
    for (const d of dirs) {
      ps.restoreBackup(node);
      const r = ps.step(d);
      if (r.againCapHit) continue;
      if (r.winning) {
        plan = [d];
        for (let p = seen.get(k); p; p = seen.get(p[0])) plan.unshift(p[1]);
        break outer;
      }
      const b = ps.backup(), kb = key(b);
      if (!seen.has(kb)) { seen.set(kb, [k, d]); next.push(b); }
    }
  }
  frontier = next;
}
console.log(JSON.stringify({ level: levelIdx, plan, states: seen.size,
  exhausted: !timeout && plan === null, ms: Date.now() - t0 }));
