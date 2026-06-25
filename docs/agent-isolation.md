# Agent isolation & anti-cheat

How regact stops a code agent from cheating, why the naive approach fails, what
the chosen design covers, and the limits. Self-contained: no prior context needed.

---

## 1. The problem

regact drives a **code agent** (Claude Code, codex, an in-process agent) to write
code — a *deliverable*, e.g. a controller `act(obs) -> action` — that plays a game
(MiniGrid, ARC-AGI-3). We then **score** that deliverable. The game's source and
data sit on the **same machine** as the agent during development.

This is an exam where **the answer key is in the room with the student**. Instead of
*solving* the game (what we want to measure), the agent can simply walk over and
**read the answer**: open the game's source files, copy the winning sequence, submit
it. The score then measures "can it find a file on disk", which is worthless for
research — and it is **silent**: a cheating deliverable looks like a great result
unless someone digs.

This is not hypothetical. In an early run, an agent left its working directory and
read the game's source via an absolute path:

```
python3 -c "print(open('/abs/path/to/environnement/<game>/<game>.py').read())"
```

It bypassed the per-tool permission deny-list entirely, because **a tool-permission
deny-list is not an OS sandbox**: it governs the agent's *named tools* (Claude's
`Read`), never *arbitrary code* the agent runs.

### Why "block each trick" cannot work

The instinct is to forbid the bad calls one by one. This loses, always:

> forbid `open` -> it uses `import`; forbid `import` -> it uses `find` + read; forbid
> `find` -> it reads `/proc`; forbid that -> it builds the path from `os.environ`...

The secret is reachable, and there are unbounded ways to reach something reachable.
A **blocklist of techniques** can never enumerate them all. The lesson:

> **Do not block the thousand paths to the secret. Remove the secret from the agent's
> reach.** If the bytes are absent from the agent's filesystem, `open`, `import`,
> `find`, `os.environ` all return nothing — there is nothing to find.

This is how a hosted competition platform is safe without inspecting your code: the
hidden answer is **not in the box**, and the **network is cut** during scoring. We
copy that property: make "the answer is unreachable" the default for any scored run.

---

## 2. The threat catalog

Every way an agent can try to reach the secret, grouped by *mechanism*. These are
test cases, not a clean partition — a single attack can span groups, which only means
it is defended by more than one layer. Each group lists the layer that defeats it
(see section 3 for R-numbers, section 4 for the layers).

| # | Family | Concrete sub-cases | Defeated by |
|---|---|---|---|
| **A** | Read a file directly | A1 `open('/abs/game.py')` · A2 `open('../../environnement/...')` (relative climb) · A3 shell `cat`/`head`/`less`/`xxd` · A4 `grep -r`/`rg` · A5 **locate first**: `find / -name '*.py'`, `ls -R /`, `locate` · A6 globs `Path('/').rglob('*game*')` · A7 symlink into the secret dir, read through it · A8 `os.walk('/')`, `os.scandir` · A9 `/proc/self/root`, `/proc/<pid>/cwd`, `/proc/<pid>/fd/*` · A10 low-level `os.open`/`mmap`/`shutil.copy` | **R2** — the path is absent from the agent's filesystem view |
| **B** | Load the game as code | B1 `import arc_agi` · B2 `importlib.import_module('arc_agi')` · B3 `__import__('arc'+'_agi')` (name built at runtime, evades a static scan) · B4 `exec` of game source read from disk · B5 `inspect.getsource(obj)` · B6 `sys.modules[...]` / patched `sys.path` · B7 read `module.__file__` then open it | **R2** + deliverable scan (defense-in-depth) |
| **C** | Reach into the orchestrator process | C1 grab the in-memory game object via `sys.modules` · C2 `gc.get_objects()` · C3 read the orchestrator's `os.environ` / cwd / `PYTHONPATH` | **process separation** (game object not in the agent's/eval's memory) |
| **D** | Discover the secret's location | D1 `os.environ` · D2 parent cmdline `/proc/<ppid>/cmdline` · D3 logs / config / error messages · D4 `sys.argv`, `sys.path` | **R2** — knowing an address you cannot reach is harmless |
| **E** | Exfiltrate / pull data | E1 open a socket to an external host and send what it read · E2 download the game/answer from the internet · E3 DNS exfiltration | **R5** — external egress denied (or absent by platform) |
| **F** | Memorize after legitimate play | F1 a deliverable that ignores `obs` and replays a fixed action list · F2 an embedded answer grid · F3 high constant-to-logic ratio | **Not cheating** — see note below; addressed by *held-out evaluation*, not the sandbox |
| **G** | Attack the sandbox / denial of service | G1 retry the attack from a **child** process · G2 try to lift its own restriction · G3 fork-bomb / OOM · G4 modify the framework/venv for a later run | **R6** (inheritance) · resource limits · **R3** (write-deny) |

**Note on F (memorization).** This is *not* reading the answer — no secret is stolen.
The agent played the game through the legitimate interface and learned the winning
moves. If reading the answer is prevented (A–E closed), reaching the solution *means*
the agent solved it; encoding that solution is just an efficient deliverable. F only
matters if you want to claim **generalization to unseen instances**, which is a
*measurement* question answered by held-out evaluation, not a security hole.

---

## 3. The contract: invariants R1–R6

The defense is expressed as six invariants the agent's sandbox must satisfy. The
contract is **identical on every platform**; only the enforcing mechanism differs
(section 4). A conformance **probe** checks these same invariants everywhere, so "the
same restriction everywhere" is testable, not asserted.

| Invariant | Intent | Statement |
|---|---|---|
| **R1** | allow | The agent **can** read & write its own workdir, the framework, and the interpreter — it must still work. |
| **R2** | deny | The agent **cannot** read the game files, nor climb out of its workdir. |
| **R3** | deny | The agent **can write only** its workdir (+ `/tmp`) — not the framework, venv, or game data. |
| **R4** | allow | The agent **can** reach `localhost` (the env server + a local model). |
| **R5** | deny | The agent **cannot** reach the external internet on a scored run. |
| **R6** | deny | A **child** process inherits the restriction — the agent cannot escape by spawning. |

The probe attempts the catalog (section 2) and reports, per invariant, *defended* or
*vulnerable*. Run it with no sandbox (expected vulnerable) and inside the sandbox
(expected defended) to measure the difference; run it by hand on each target host to
produce a conformance table.

---

## 4. How the isolation works

Four layers. The first three are generic (agnostic to any agent, environment, or
feature); the deliverable check is feature-specific.

**(a) The OS sandbox — the wall.** The agent's subprocess is launched *inside* an OS
sandbox that presents a **filesystem allowlist**: only the workdir, the framework
source, and the interpreter are visible; everything else — the game files, other
experiments — is simply **not present**. A read of an absent path fails at the
syscall level (`open(...)` raises `PermissionError`/`FileNotFoundError`), enforced by
the OS, for every call of the process *and every child it spawns* (R6). We do not
inspect individual calls — we wrap the process once and the kernel does the rest.

The mechanism is platform-specific but the contract is not:

| Platform | Mechanism | Filesystem | Egress |
|---|---|---|---|
| macOS (dev) | `sandbox-exec` (Seatbelt) | deny-by-default, allow the system layer + interpreter + allowlist | open (the model is remote) |
| Linux | bubblewrap (`bwrap`) -> Landlock fallback | deny-by-default mount namespace, bind only the allowed paths | seccomp (keep loopback) |
| Jean-Zay / Adastra (HPC) | Apptainer/Singularity SIF (`--containall --no-home`) | deny-by-default, bind only the allowed paths | intrinsic (compute nodes have no internet) |
| Kaggle | none needed | intrinsic (game data is remote) | intrinsic (network cut during scoring) |

A single function, `wrap_argv(runtime, argv, ...)`, prepends the right launcher; the
rest of regact does not depend on which backend runs. The backend is selected by a
config knob (`security.runtime`, default `auto`, which probes the host); `none` runs
the agent unwrapped (a development convenience and a baseline for measuring cheating).

**(b) Prevention by absence.** The sandbox only helps if the secret is genuinely
outside the allowlist. The env server (which owns the game library and data) runs as
a **separate process**, reached only over a localhost HTTP boundary, so the game
object is never in the agent's — or the evaluator's — memory (C1–C3), and the game
files are never bound into the agent's view (R2). *(Process separation of the env
server and the evaluator is the next implementation step; today the HTTP object
boundary is in place and the sandbox provides the filesystem confinement.)*

**(c) The deliverable scan — feature-specific.** Each feature validates *its own*
deliverable. The controller feature statically scans `solution.py` (AST) before it
runs and rejects code that imports the game library or calls an escape hatch — a
contract on the *shape of the submission*, with a clear error. This is hygiene on the
artifact, not OS-level confinement, and it lives with the feature so a different
feature can apply different rules.

**(d) The detection signal — observe, never block.** Independently of the sandbox, a
cheap scan over the agent's tool-call arguments flags any call that names a forbidden
path or module. It **never blocks** (the sandbox is what blocks); it **records** the
attempt as a forensic signal: a `cheat_attempt` count on the run state and a WARNING
log line naming the tool and the flag. This works even when the sandbox is disabled —
so one can run an agent unconfined and observe *whether and how* it tries to cheat.

---

## 5. Limitations

Stated plainly, because they belong in any honest account.

- **Threat model.** We defend against a *cheating agent*, not a kernel attacker. The
  sandbox shares the host kernel, so a kernel exploit could escape. Defending against
  that needs microVMs (Firecracker/gVisor), which are unavailable on the target HPC
  systems and out of scope here.
- **Backend availability.** `bwrap` needs unprivileged user namespaces, often disabled
  on hardened/HPC kernels; the fallbacks are Landlock (kernel >= 5.13) or Apptainer.
  Where none is available, confinement falls back to *prevention by absence* (the game
  on a different machine) plus the detection signal. The runtime is probed at startup.
- **Claude Code does not run under the macOS sandbox.** Seatbelt is deny-by-default like
  the other backends, and CPython and the codex CLI run fine under it — but `claude.exe`
  (a Bun binary) crashes at the native level under a deny-default Seatbelt profile on macOS
  (it logs nothing, and `dtruss` is SIP-blocked, so the cause is not observable). So Claude
  on macOS runs `runtime=none` (macOS is the trusted dev box); Claude is confined on the
  Linux/HPC run hosts instead, where an un-allowed path is *absent* (ENOENT) rather than
  *denied* (EPERM), which the Bun binary tolerates. `sandbox-exec` is also deprecated though
  still functional.
- **The evaluator must also be confined.** A deliverable that passes the static scan
  still runs at evaluation time; if evaluation runs in an unconfined process, that
  reopens the read-the-answer hole. Running the evaluator in its own confined
  subprocess is part of the prevention-by-absence work.
- **The detection signal is best-effort.** It scans tool-call text, so an obfuscated
  attempt may evade the *log* — but not the sandbox, which still blocks it. Obfuscation
  costs us a label, not security.
- **Egress granularity.** "Deny external, keep loopback" needs a custom seccomp filter
  on local Linux; on HPC/Kaggle egress is already absent, so the knob mostly matters
  for a local Linux host serving a local model.
- **The orchestrator is trusted.** All of this assumes regact's own process is honest.

---

## 6. Summary

**How the sandboxing works, and what it covers.** regact confines the agent's process
inside an OS sandbox that exposes a filesystem allowlist: the agent sees only its
working directory, the framework source it imports, and the interpreter; the game's
source and data are absent, so any attempt to read them — by `open`, `import`, shell
search, `/proc`, symlink, or arbitrary subprocess — fails at the syscall level, and a
spawned child inherits the same restriction. The contract is six invariants (R1–R6)
that hold identically on every target — macOS, Linux, the Jean-Zay and Adastra HPC
clusters, and Kaggle — while the enforcing mechanism varies per platform (Seatbelt,
bubblewrap/Landlock, Apptainer, or the platform's intrinsic isolation), and a
conformance probe verifies the invariants on each host. This closes the *read-the-
answer* class of cheating (catalog families A–E); it deliberately does **not** defend
against kernel-level escape, which is out of scope for a cheating-agent threat and
unsupported on the HPC targets. Confinement is selected by one configuration knob and
is fully decoupled from how an experiment is launched, so the same isolation applies
to research and competition runs alike.

**What signal it lets us analyze.** Because confinement is enforced by the operating
system rather than by inspecting each action, the framework is free to *observe* the
agent's behaviour as data. Every tool call that reaches for a forbidden path or module
is recorded as a cheat-attempt count and a warning, surfaced per game in the
visualizer — so even an unconfined run yields a clean measurement of whether, and how,
an agent tries to cheat. On top of this generic signal, each feature contributes its
own analysis of its deliverable: in the controller approach, the submitted policy is
statically checked to be a pure `act(obs)` function, and its behaviour can be
re-examined to distinguish genuine problem-solving from memorization. For re-seedable
environments such as MiniGrid, re-running a kept controller on a fresh seed reveals a
solution that merely hard-coded a seed-specific answer; for fixed-instance puzzles such
as ARC-AGI-3, where re-running cannot expose memorization, the appropriate signal is
evaluation on held-out instances the agent never had access to during development.
Together these give a defensible picture: the operating system guarantees the agent
*could not* read the answer, and the recorded signals show what it actually did.
