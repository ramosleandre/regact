# contexte_isolation_state — read this first on a new machine

This file carries the isolation context that, until now, lived only in a working
session on the macOS dev box. **A fresh session on another machine (e.g. the Linux
box) should read this to get the full, current, *corrected* picture** before
touching the sandbox/egress code. It pairs with the runnable
[debug_isolation_linux.md](debug_isolation_linux.md).

Status as of this writing: the networking facts below were established by an
**adversarial documentation/source audit** (Linux man-pages, the bubblewrap
source, kernel docs, codex/claude issue trackers — mid-2026). They were **not**
verified empirically, because the audit ran on macOS where `bwrap`/`unshare`
cannot run. `debug_isolation_linux.md` exists to confirm them on real Linux.

---

## TL;DR — the three orthogonal mechanisms

Do not conflate these (the code comments historically did):

| # | Mechanism | What it does | Blocks? |
|---|---|---|---|
| 1 | **OS filesystem sandbox** (`security.sandbox` = seatbelt/bwrap/apptainer/none) | "the answer key is not on the agent's disk" — Linux = a bwrap **mount** namespace; an un-bound path is **absent** (`ENOENT`) | **yes** (filesystem) |
| 2 | **`deny_egress`** (+ the `EgressProxy`) | "no internet except the model's API" — block external egress, keep loopback. The proxy is the *hostname-granular* half (`HTTPS_PROXY` → localhost CONNECT proxy → allow-list) | **yes** (network) |
| 3 | **Detection camera** (`security/detection.py`: `flag_tool_call`, `flag_os_denial` — the "deny os" scan) | **forensic only**: counts/records when the agent reaches for a forbidden path (1) or egress is refused (2). Reads tool ARGS + tool-RESULT text | **no — never** |

`flag_os_denial` (the "deny os" you may have heard) is **observation**, not
enforcement: it just notices that mechanism 1 or 2 already fired.

---

## The corrected networking truth (what I previously got wrong)

1. **`bwrap --unshare-net` does NOT "sever loopback."** bubblewrap unconditionally
   calls `loopback_setup()` and brings `lo` **UP** with 127.0.0.1 inside the new
   netns (it even aborts if it can't). Sources: bubblewrap `network.c`
   (`ifi_flags=IFF_UP`, `RTM_NEWADDR INADDR_LOOPBACK`), `bubblewrap.c`
   (`if (opt_unshare_net) loopback_setup()`), containers/bubblewrap#745. *Verify
   with `[L1]`.*

2. **The real constraint is network-namespace ISOLATION.** The sandbox gets its
   *own* loopback stack; the host's `127.0.0.1` is a different namespace. So a
   server on the **host** loopback (our env server, our `EgressProxy`) is
   unreachable from inside — *not* because `lo` is down. *Verify with `[L2]`.*

3. **THE CURRENT BUG (unfixed): the egress allow-list is broken under
   `deny_egress`+bwrap.** `EgressProxy` binds the host's `127.0.0.1`
   (`security/egress_proxy.py`), the agent runs under `--unshare-net`
   (`orchestration/task.py`), so the agent's `HTTPS_PROXY=http://127.0.0.1:<port>`
   hits its **own empty loopback** → it reaches **no** URL, even allow-listed ones.
   The url-allow-list isn't impossible — `--unshare-net` just throws loopback +
   proxy out together with the external net.

4. **"claude is not possible on Linux" is FALSE.** Claude Code is officially
   supported on Linux (native Bun binary, headless `claude -p`); the Bun crash is
   *macOS-Seatbelt-specific* and even that cause (file-EPERM) is unverified — the
   documented Bun/Seatbelt bug is empty `process.env`, and bwrap yields `ENOENT`,
   not `EPERM`. *Verify with `[A1]`.*

5. **The doc's "seccomp (keep loopback)" is not implementable as written.** A
   classic seccomp-BPF filter cannot allow-list `connect()` by destination (the
   `sockaddr` is passed by pointer; seccomp-BPF can't dereference pointers — kernel
   `seccomp_filter` docs, LWN 822256). Only `SECCOMP_USER_NOTIF`/ptrace can, and
   that is IP-granular, not hostname.

6. **The CONNECT proxy alone is advisory.** `HTTPS_PROXY` is opt-in; a hostile
   arbitrary-code agent can raw-socket past it. It's a security boundary only with
   an un-bypassable backstop (no direct external route). For a *cheating* agent
   (our actual threat model — cheating is a metric we analyse, the camera is
   non-blocking) the soft proxy + logging is a reasonable level.

---

## HPC / local-model: there is NO egress problem here

- **HPC** compute nodes have **no internet** (intrinsic). The model runs locally
  (vLLM on `127.0.0.1`), the env runs locally. The agent needs only **loopback**.
  Since there's nothing external to block, you simply **don't `--unshare-net`** →
  loopback is shared → env + local model reachable over normal TCP. No socket, no
  proxy, no bug.
- **Local Linux + a local model:** same — only loopback is needed.
- The whole egress headache exists **only** for *local Linux + a REMOTE model*
  (codex→OpenAI, claude→Anthropic), the one case the agent must reach the outside.

---

## Current code state (after reverting the unix-socket experiment)

- The unix-socket env transport that was briefly added has been **removed**. The
  env transport is back to: in-process `TestClient` (scripted) or a real uvicorn
  TCP server on `127.0.0.1:<port>` (`orchestration/env_transport.py`).
- `deny_egress` on bwrap is still implemented as `--unshare-net`
  (`security/runtime.py` `_bwrap`) — the blunt tool. On macOS seatbelt it is a
  loopback-allow rule (the *right* behaviour: block external, keep loopback).
- `EgressProxy` (host `127.0.0.1`) + per-agent allow-list
  (`api.anthropic.com`; `api.openai.com`/`auth.openai.com`/`chatgpt.com`).
- Detection camera unchanged (forensic).

### Comments/docs still factually WRONG (fix only after `[L1]/[L2]` confirm on Linux)

- `security/runtime.py` `_bwrap`: the `--unshare-net … also severs loopback` comment.
- `docs/sandbox-testing.md` (note ~line 75): same "also cuts loopback" wording.
- `docs/agent-isolation.md` (table ~line 120 "Egress: seccomp (keep loopback)";
  limitation ~line 180): "deny external, keep loopback needs a seccomp filter" —
  a classic seccomp filter cannot do per-host egress (see point 5).

---

## The decision pending (egress on local Linux + remote model)

The goal: block external, **keep loopback** (env + proxy reachable), allow only the
model's URL. `--unshare-net` is the wrong tool. Options, with the current lean:

- **(leaning) Soft proxy, no `--unshare-net`:** keep the host netns so loopback +
  `EgressProxy` are reachable; let the CONNECT proxy be the egress control and
  **log every CONNECT** (allowed + denied) as the cheat signal. Unblocks
  claude/codex on Linux immediately. Soft against a hostile agent (raw-socket
  bypass), which matches the "cheating is a metric" stance.
- **Hardened (later):** agent in an orchestrator-**owned** netns (lo up, no
  external route), proxy = sole egress → hostname-granular **and** un-bypassable.
  Or nftables egress-drop in a **parent**-owned netns (not the agent's userns — it
  has `CAP_NET_ADMIN` there and could flush its own rules).
- `slirp4netns`/`pasta` keep loopback and give user-mode egress but ship **no**
  allow-list — they'd need the proxy or nftables on top.

The user's instinct: since we know the allowed URLs, allow them (so it
communicates) and **flag** egress via the camera/proxy log, rather than a brittle
case-by-case "forbid curl" blocklist (which the threat-model doc itself rejects).

---

## What to do on the Linux box

1. Run [debug_isolation_linux.md](debug_isolation_linux.md) `[ENV] [L1] [L2] [L3]
   [P1] [A1]` and paste the output.
2. From `[L1]/[L2]` we confirm the loopback truth and the proxy-reachability bug.
3. Then decide the egress direction (above) and implement it; **then** fix the
   wrong comments/docs to match the confirmed reality.

### Sources (audited)
bubblewrap `network.c`/`bubblewrap.c` + containers/bubblewrap#745; man
`network_namespaces(7)` (isolates only the *abstract* unix namespace),
`unix(7)`, `mount_namespaces(7)`; kernel `seccomp_filter` docs + LWN 822256;
`code.claude.com/docs` (setup/headless/network-config); openai/codex#4242;
oven-sh/bun#27802; anthropics/claude-code#6637,#14719,#45541;
anthropic-experimental/sandbox-runtime#74; passt.top, slirp4netns docs.
