# debug_isolation_linux — commands to run on the Linux box

Purpose: **empirically settle** how the OS isolation actually behaves on a real
Linux host (this could not be tested on the macOS dev box — see
[contexte_isolation_state.md](contexte_isolation_state.md)). Run the blocks below
on the Linux/Fedora machine and **paste the output back, labelled by tag** (e.g.
`[L2] output: …`). Each block says what it checks and the *expected* result; I
analyse any deviation.

Setup once:

```bash
git clone <repo> regact && cd regact
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # or: pip install .   (if the editable .pth misbehaves)
python -c "import regact; print('regact import OK')"
```

> All `bwrap` blocks below use `--ro-bind / / --dev /dev` for brevity: these are
> **network** experiments, not the real regact filesystem sandbox. They only need
> `curl`, `ip`, `python3` to exist inside — binding `/` read-only provides them.

---

## [ENV] Host facts

```bash
uname -a
grep -E '^(NAME|VERSION)=' /etc/os-release
python3 --version
which bwrap && bwrap --version
which curl ip
unshare -Urm true && echo USERNS_OK || echo USERNS_BLOCKED
cat /proc/sys/user/max_user_namespaces 2>/dev/null
sysctl kernel.unprivileged_userns_clone 2>/dev/null || echo "(no unprivileged_userns_clone sysctl — fine on Fedora)"
```

**What it checks:** Python version, that `bwrap` exists, and that unprivileged
user namespaces work (bwrap needs them). **Expect on Fedora:** `USERNS_OK`.
If `USERNS_BLOCKED`, paste it — the bwrap blocks below will fail and we adapt.

---

## [L1] Is loopback UP inside `bwrap --unshare-net`?

This is the claim I got wrong. The audit says bwrap brings `lo` up automatically.

```bash
bwrap --ro-bind / / --dev /dev --unshare-net ip addr show lo
```

**Expect:** `lo: <LOOPBACK,UP,LOWER_UP> … inet 127.0.0.1/8`. If `lo` is **UP**, the
old comment "`--unshare-net` severs loopback" is **wrong** — the real issue is
namespace *isolation* (next block). If it errors with
`RTM_NEWADDR: Operation not permitted` and bwrap exits, paste that (it's the
hardened-kernel failure mode where the sandbox refuses to start at all).

---

## [L2] Can a process inside `--unshare-net` reach a server on the HOST's 127.0.0.1?

This is the heart of it: the env server **and** the egress proxy run on the host's
loopback. Does the sandboxed agent reach them?

```bash
# start a throwaway server on the HOST loopback:
python3 -m http.server 8999 --bind 127.0.0.1 >/dev/null 2>&1 & SRV=$!; sleep 1

# A) inside a fresh network namespace (deny_egress path today):
bwrap --ro-bind / / --dev /dev --unshare-net \
  sh -c 'curl -sS --max-time 3 -o /dev/null -w "A unshare-net: HTTP %{http_code}\n" http://127.0.0.1:8999/ \
         || echo "A unshare-net: UNREACHABLE (exit $?)"'

# B) sharing the host network namespace (no --unshare-net):
bwrap --ro-bind / / --dev /dev \
  sh -c 'curl -sS --max-time 3 -o /dev/null -w "B shared-net: HTTP %{http_code}\n" http://127.0.0.1:8999/ \
         || echo "B shared-net: UNREACHABLE (exit $?)"'

kill $SRV 2>/dev/null
```

**Expect:** `A unshare-net: UNREACHABLE` (the sandbox has its *own* loopback stack,
separate from the host's) and `B shared-net: HTTP 200`. This demonstrates **the
current bug**: under `deny_egress`+bwrap the host-side `EgressProxy` (and a host
TCP env server) are unreachable — and that simply **not** unsharing net (B) fixes
reachability.

---

## [L3] External internet — blocked under `--unshare-net`, reachable without?

```bash
bwrap --ro-bind / / --dev /dev --unshare-net \
  sh -c 'curl -sS --max-time 5 -o /dev/null -w "unshare-net ext: %{http_code}\n" https://api.github.com \
         || echo "unshare-net ext: BLOCKED (exit $?)"'
bwrap --ro-bind / / --dev /dev \
  sh -c 'curl -sS --max-time 5 -o /dev/null -w "shared-net ext: %{http_code}\n" https://api.github.com \
         || echo "shared-net ext: BLOCKED (exit $?)"'
```

**Expect:** `unshare-net ext: BLOCKED` (no external route in a fresh netns) and
`shared-net ext: 200`. Confirms `--unshare-net` *does* block external egress — it
just throws loopback out with it.

---

## [L4] (informative) Does a pathname unix socket cross `--unshare-net`?

Confirms *why* a unix-socket transport would work (I removed that code, but this
verifies the kernel fact behind it).

```bash
mkdir -p /tmp/rgsock && rm -f /tmp/rgsock/t.sock
python3 - <<'PY' & SOCK=$!
import socket, os
p = "/tmp/rgsock/t.sock"
s = socket.socket(socket.AF_UNIX); s.bind(p); s.listen()
c, _ = s.accept(); c.sendall(b"HELLO-FROM-HOST-UDS"); c.close()
PY
sleep 1
bwrap --ro-bind / / --dev /dev --bind /tmp/rgsock /tmp/rgsock --unshare-net \
  python3 -c "import socket; s=socket.socket(socket.AF_UNIX); s.connect('/tmp/rgsock/t.sock'); print('L4 uds across --unshare-net:', s.recv(64))"
kill $SOCK 2>/dev/null; rm -f /tmp/rgsock/t.sock
```

**Expect:** `L4 uds across --unshare-net: b'HELLO-FROM-HOST-UDS'` — a pathname unix
socket (filesystem-addressed) crosses the network-namespace boundary that TCP
loopback cannot. (Informative only; not needed for the chosen direction.)

---

## [P1] regact's own backend detection + conformance probe

```bash
python -m regact.security.probe --json            # baseline: NO sandbox -> expect breaches (R2/R5 vulnerable)
python -m regact.security.probe --sandbox --json   # confined: expect detect() -> bwrap, R1-R6 mostly DEFENDED
```

**Expect:** baseline reports the game-secret read as **vulnerable**; `--sandbox`
reports `detect() -> bwrap` and the deny invariants **DEFENDED** (R2 read-game,
R3 write-out, R6 child). Paste both JSON blobs; the per-invariant table is the
real conformance answer for this host.

---

## [A1] Do the CLI agents even run on Linux? (refutes "claude not possible on Linux")

Only if `claude` / `codex` are installed and authenticated (see
[agents-setup.md](agents-setup.md)).

```bash
which claude codex
claude --version 2>&1 | head -1
codex --version 2>&1 | head -1
# claude runs headless on Linux at all?
claude -p "reply with exactly: PONG" --output-format json 2>&1 | head -c 400; echo
# does claude survive the real regact bwrap filesystem sandbox? (no network needed for this)
python -m regact.security.probe --sandbox --json >/dev/null 2>&1 && echo "probe ran under bwrap OK"
```

**Expect:** both report versions; `claude -p` returns a normal JSON stream (a
`PONG`), proving claude is fine on Linux. If claude crashes/among bwrap, paste the
error — that's the thing we could not observe on macOS.

---

## [A2] (after a direction is chosen) full regact run, watching egress

Once we pick the egress approach, run one real game and check the agent reaches
its model and the env. Placeholder — we fill this in together:

```bash
# example (soft-proxy direction, model = codex/OpenAI):
python -m regact.run_exp agent=codex problem=arc_agi 'task_names=[ls20]' \
    security.sandbox=bwrap security.deny_egress=false limits.keep_alive=2
# then inspect:  experiments/<run>/ls20/logs/transcript.jsonl  and  workdir/submissions/<n>/results.json
```

**What I'll look for:** did the agent reach its LLM? did the controller submit +
score? any `egress_denied` cheat flags in `logs/events.jsonl`?

---

### How to send results

Paste each block's output under its tag. If a `bwrap` block errors before running
the inner command (e.g. `bwrap: … Operation not permitted`), that's the
userns/AppArmor situation — paste it verbatim and we adjust (per-binary AppArmor
profile, or treat the box like HPC with Apptainer).
