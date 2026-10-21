'use strict';
// JSON-lines server for one game: a request per stdin line, a response per stdout line.
//   {"cmd": "load", "file": <game source>, "level": <engine level index>}  -> {cells, winning}
//   {"cmd": "step", "dir": 0-4}                                           -> {cells, winning}
//   {"cmd": "info"}  -> the compiled game's objects, sprites, layers and levels (for rendering)
// cells[y][x] lists the ids of the objects in that cell, in increasing order.
const fs = require('fs');
const readline = require('readline');
const ps = require('./psnode.js').load();

const SEED = 'regact';  // the strict games draw no randomness; a fixed seed keeps any draw stable
let text = null;

function handle(req) {
  if (req.cmd === 'load') {
    text = fs.readFileSync(req.file, 'utf8');
    const r = ps.compileGame(text, req.level, SEED);
    if (r.errorCount) return { error: 'compile errors: ' + r.errors.join(' | ') };
    return { cells: ps.grid(), winning: false };
  }
  if (req.cmd === 'step') {
    const r = ps.step(req.dir);
    if (r.againCapHit) return { error: 'an again cascade did not settle' };
    return { cells: ps.grid(), winning: r.winning };
  }
  if (req.cmd === 'info') {
    const st = ps.state();
    return {
      objects: st.idDict,
      layers: st.idDict.map(name => st.objects[name].layer),
      sprites: st.idDict.map(name => ({ colors: st.objects[name].colors,
        matrix: st.objects[name].spritematrix.map(row => Array.from(row)) })),
      background: String(st.bgcolor || '#000000'),
      levels: st.levels.map(l => (l.message !== undefined ? null : [l.width, l.height])),
      noaction: st.metadata.noaction !== undefined,
    };
  }
  return { error: 'unknown command ' + req.cmd };
}

readline.createInterface({ input: process.stdin }).on('line', line => {
  let resp;
  try {
    resp = handle(JSON.parse(line));
  } catch (e) {
    resp = { error: String(e && e.stack) };
  }
  process.stdout.write(JSON.stringify(resp) + '\n');
});
