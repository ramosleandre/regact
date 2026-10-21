'use strict';
// Headless loader for the original PuzzleScript engine (increpare/PuzzleScript, MIT), whose
// src/js files are run unmodified in a vm context with browser shims (the approach of the
// engine's own src/tests/run_tests_node.js). PS_SRC is the engine's src directory.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SRC = process.env.PS_SRC;
const FILES = ['js/storagewrapper.js', 'js/bitvec.js', 'js/level.js', 'js/languageConstants.js',
  'js/globalVariables.js', 'js/debug.js', 'js/font.js', 'js/rng.js', 'js/riffwave.js', 'js/sfxr.js',
  'js/codemirror/stringstream.js', 'js/colorhelpers.js', 'js/colors.js', 'js/engine.js',
  'js/parser.js', 'js/compiler.js', 'js/soundbar.js'];

function makeSandbox() {
  const store = {};
  const el = () => ({ style: {}, innerHTML: '', textContent: '', addEventListener() {}, removeEventListener() {},
    appendChild() {}, setAttribute() {}, getContext() { return null; } });
  const sb = { console: { log() {}, warn() {}, error() {}, info() {} }, setTimeout, clearTimeout, setInterval, clearInterval,
    Int32Array, Uint8Array, Uint32Array, Float32Array, Float64Array, ArrayBuffer, Math, Date, JSON,
    performance: require('perf_hooks').performance };
  sb.localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: k => { delete store[k]; } };
  sb.document = { URL: 'node://ps', body: { classList: { contains() { return false; } }, addEventListener() {}, removeEventListener() {}, appendChild() {} },
    createElement: el, getElementById() { return null; }, addEventListener() {}, removeEventListener() {} };
  sb.window = sb; sb.globalThis = sb; sb.self = sb;
  sb.Audio = function () { return { play() {}, pause() {}, load() {}, addEventListener() {}, setAttribute() {}, cloneNode() { return this; }, canPlayType() { return ''; } }; };
  sb.canvas = null; sb.lastDownTarget = null; sb.input = el(); sb.IDE = false;
  for (const f of ['canvasResize', 'redraw', 'forceRegenImages', 'consolePrintFromRule', 'consolePrint', 'consoleError',
    'consoleCacheDump', 'killAudioButton', 'showAudioButton', 'regenSpriteImages', 'jumpToLine', 'printLevel', 'playSound',
    'clearInputHistory', 'pushInput', 'resetCommands', 'tryLoadCustomFont']) sb[f] = function () {};
  sb.addToDebugTimeline = () => 0;
  sb.console_print_raw = () => {};
  sb.storage_quota_infinite = true;
  vm.createContext(sb);
  return sb;
}

const AGAIN_CAP = 500;

const API_CODE = `
unitTesting = true;            // 'message' commands do not open a modal text screen
lazyFunctionGeneration = false;
function DoWin() {             // record the win, never auto-advance (the wrapper owns level order)
  if (winning) return;
  againing = false;
  winning = true;
}
function __flushAgain(cap) {
  let n = 0;
  while (againing && n < cap) { againing = false; processInput(-1); n++; }
  return n;
}
function __cellObjects(i) {
  const cell = level.getCell(i);
  const out = [];
  for (let k = 0; k < state.objectCount; k++) if (cell.get(k)) out.push(k);
  return out;
}
globalThis.__API = {
  compileOnly(text) {
    errorStrings = []; errorCount = 0;
    if (typeof resetParserErrorState === 'function') resetParserErrorState();
    compile(['restart'], text);
    return { errorCount, errors: errorStrings.slice(0, 5) };
  },
  compileGame(text, levelIdx, seed) {     // a fresh compile per level: no state survives a reset
    errorStrings = []; errorCount = 0;
    if (typeof resetParserErrorState === 'function') resetParserErrorState();
    compile(['loadLevel', levelIdx], text, seed);
    winning = false;
    __flushAgain(${AGAIN_CAP});
    return { errorCount, errors: errorStrings.slice(0, 5) };
  },
  step(dir) {                             // 0 up, 1 left, 2 down, 3 right, 4 action
    processInput(dir);
    __flushAgain(${AGAIN_CAP});
    return { winning, againCapHit: againing };
  },
  backup() { return backupLevel(); },
  restoreBackup(b) { restoreLevel(b); winning = false; againing = false; },
  grid() {
    const W = level.width, H = level.height, rows = [];
    for (let y = 0; y < H; y++) { const r = []; for (let x = 0; x < W; x++) r.push(__cellObjects(y + x * H)); rows.push(r); }
    return rows;
  },
  state() { return state; },
};
`;

function load() {
  const sb = makeSandbox();
  let code = '';
  for (const f of FILES) code += `\n// ---- ${f} ----\n` + fs.readFileSync(path.join(SRC, f), 'utf8') + '\n';
  vm.runInContext(code, sb, { filename: 'ps_sources.js' });
  vm.runInContext(API_CODE, sb, { filename: 'ps_api.js' });
  return sb.__API;
}

module.exports = { load };
