# Sandboxing

How Regact restricts access to hidden game information, isolates submitted code, and checks those restrictions on the host running an experiment.

## The threat

The agent runs arbitrary code. Nothing stops it from writing
`open("/path/to/ar25.py").read()` to read the game's source or its answer key, then
hard-coding the result. If it can reach the game files, the score measures memorization,
not understanding. The defense is a **filesystem allowlist**: the agent sees its workspace and required runtime/backend paths, while game source/data are excluded. This controls access; it does not establish that every valid solution reflects general understanding.

## The contract (R1–R6)

The same six invariants hold on every platform; only the enforcing mechanism differs.
They live in [`security/contract.py`](../src/regact/security/contract.py):

| # | Invariant | Rule |
|---|---|---|
| **R1** | workdir | CAN read & write its own workdir + framework + venv (must work) |
| **R2** | secret | CANNOT read the game files or climb out of the workdir |
| **R3** | write | can write ONLY its workdir (+ `/tmp`), not framework / venv / games |
| **R4** | loopback | CAN reach the env server + a local LLM via the sanctioned path |
| **R5** | egress | CANNOT reach the external internet (on a scored run) |
| **R6** | no-escape | a child process inherits the restriction (no escape) |

## The backends

`sandbox: true` auto-detects the strongest backend available; the run **fails** rather
than proceed unconfined. Resolution lives in
[`security/runtime.py`](../src/regact/security/runtime.py):

| Platform | Backend | Mechanism |
|---|---|---|
| Linux (local, Adastra, Jean Zay) | **bwrap** | bubblewrap mount namespace: bind only the allowlist |
| macOS | **seatbelt** | `sandbox-exec`: deny-by-default profile, allow the workdir |
| Kaggle | **none** | intrinsic — the answer is absent and the kernel has no internet |

This table describes the agent sandbox. Vanilla/CWM additionally require an available OS sandbox for submitted-code workers; they do not support an unconfined worker fallback, including when `sandbox=false` is used for the coding agent. The legacy Kaggle integration is not a validated managed-protocol deployment.

There is no container/SIF backend: bwrap + user namespaces are available on both HPC
clusters we target, so nothing needs building. Force a specific backend with
`+sandbox_opts.backend=<seatbelt|bwrap>` if auto-detection is wrong.

## How egress is blocked without breaking loopback (R4 vs R5)

The hard part on Linux: the agent must **not** reach the internet (R5) but **must** reach
the env server and the local LLM on `127.0.0.1` (R4). bwrap's `--unshare-net` gives the
sandbox its own empty network namespace — which blocks the internet but also cuts host
loopback.

The **netbridge** ([`security/netbridge.py`](../src/regact/security/netbridge.py)) bridges
only the sanctioned ports back in: a `LoopbackMirror` on the host forwards a unix socket
to `127.0.0.1:<port>`, and a tiny relay inside the namespace re-exposes that socket on the
same port. Unix sockets are mount-namespace objects, so a socket file bound into the
sandbox still reaches a listener outside it. Ports are preserved on both sides, so URLs
minted outside stay valid inside. The framework services and configured local model endpoint remain reachable; cloud backends separately declare the LLM hosts they need. Disabling `sandbox_opts.network_isolation` changes this network policy.

The HTTP bridge transports both environment operations and framework commands. The selected protocol controls which operations are permitted: policy search exposes an environment client; vanilla/CWM deny direct reset/step calls and expose command/data routes instead. A reachable port is not permission to bypass the protocol.

## Which paths the sandbox exposes

Deny-by-default. Only the workdir, the interpreter, and the loaded agent's declared host
dirs are bound; the game engine/data packages are carved back out even from inside the
allowed venv (`deny_read`). Each agent backend declares its own host paths
([`host_read_paths`, `host_rw_paths`, `host_egress_hosts`](../src/regact/agent/base.py)),
so the allowlist only ever contains the *loaded* agent's paths — never another backend's
ambient data.

### HPC interpreter resolution

A relocated Python (HPC module trees) is a chain of symlinks pointing at libraries
scattered across install prefixes. Binding only the final realpath is not enough for
`execvp` or the dynamic loader. Three generic mechanisms in `runtime.py` handle this —
validated unchanged on both Adastra (Spack) and Jean Zay (conda):

- `symlink_chain_dirs` — bind **every hop** of the interpreter/CLI symlink chain;
- `interpreter_chain_prefixes` — lift each `<prefix>/bin` hop to the whole `<prefix>` so
  the loader finds its sibling `lib` (RPATH `$ORIGIN/../lib`);
- `_interpreter_lib_dirs` — the full `ldd` closure of the binary + stdlib extensions, so
  every scattered `.so` dir is present up front instead of one crash per missing library.

## Verify it on your machine

The contract is checkable, not assumed:

```bash
make doctor        # which backend resolves here, and why
make probe         # run the R1-R6 contract under that backend — must print CONTRACT HELD
make agentcheck    # do the installed agent backends actually launch, bare and sandboxed?
```

`make probe` runs the checks twice: **bare** (unconfined baseline — the VULNERABLE lines
are expected, they prove the sandbox is doing something) and **sandboxed** (every attack
must flip to DEFENDED while R1 and R4 keep working). A run logs which backend it actually
got, so an unconfined run is never silent.

## The camera (detection, not enforcement)

Alongside the sandbox, a lightweight **detection camera** flags suspicious tool calls (a
call reaching for a forbidden path/module) and blocked-egress denials. It is
**detection-only** — it records flags and can send warnings to the agent; flagging itself does not block a call or invalidate a score. The sandbox and protocol handlers separately enforce permissions.

`flagging_warning_cap` defaults to 3; 0 disables delivered warnings while retaining flags. Warnings quote the relevant command with middle truncation. Alan can receive them after completed tools; CLI agents receive queued warnings on the next outer Regact send cycle, if the task continues. Attribution can be uncertain for concurrent calls, in which case the warning identifies candidate commands rather than claiming certainty. See [Protocols](protocols.md#flagging).

## Submitted vanilla/CWM code

The coding agent can read experience while developing. Submitted CWM/controller/goal callbacks instead run from immutable Python snapshots, without the agent workspace, experience database, live environment, game modules or network. CWM and controller/goal execution use separate processes. Values cross that boundary through serialization; normal static local Python imports are included in the snapshot.

Callback time, controller-call time and per-process memory limits are independent of the coding agent's shell timeout. Validation/planning also have their own whole-operation budgets. These workers stay isolated when the coding agent's sandbox is disabled. See [Managed execution](managed_protocols.md#code-isolation-and-practical-limits) for limits and [CWM](cwm.md) for the exact validation contract.

Isolation prevents runtime database lookup, but cannot prevent an agent copying observed facts into submitted Python constants. Repeatability/mutation checks are useful tests, not a proof of mathematical purity or generalization. Local `simulate.py` runs with the coding agent's permissions and current workspace code; official framework operations use the frozen, restricted code.

## What the sandbox does NOT do

- It is a **filesystem/network allowlist**, not a kernel-attack defense (no gVisor/microVM)
  — the threat model is a cheating agent, not a kernel exploit.
- On **macOS**, seatbelt puts `allow_read` paths in the read-write set (looser than bwrap's
  read-only intent) — a known, documented gap.
- The **policy-search evaluator** follows the run's sandbox configuration. For subprocess agents,
  its network isolation is enabled when `sandbox=true` and
  `sandbox_opts.network_isolation` is enabled (the default). Setting `sandbox=false`
  disables OS confinement; setting `sandbox_opts.network_isolation=false` disables
  the egress block while retaining filesystem confinement when sandboxing is on.
  The scripted backend evaluates in-process without an evaluator sandbox.

## HPC diagnostics

Ready-to-submit probe jobs for the two validated clusters (CPU, no GPU needed):

- [`scripts/adastra/isolation_probe.sh`](../scripts/adastra/isolation_probe.sh)
- [`scripts/jeanzay/isolation_probe.sh`](../scripts/jeanzay/isolation_probe.sh)

Each inventories the node (bwrap? user namespaces?), then runs doctor + probe + agentcheck
and prints the interpreter's `ldd` closure. Submit, then return the `.out`.
