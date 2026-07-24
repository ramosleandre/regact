# Testing the sandbox per machine

How to (a) **verify** the OS sandbox confines an agent on a given machine, and
(b) **discover** the exact files/endpoints `claude`/`codex` need so we can run them
deny-by-default. The probe ships with regact (`python -m regact.security.probe`), so
everything here works after a normal `pip install -e .` / `PYTHONPATH=src` — no extra
files needed. The `make debug D=sec_*` shortcuts are local conveniences (the `debug/`
dir is git-ignored), but every command below is also given raw so it works anywhere.

## The one universal check

```bash
python -m regact.security.probe --sandbox --json   # confined: should be ALL DEFENDED on Linux/HPC
python -m regact.security.probe                     # baseline (no sandbox): expect breaches
```

`--sandbox` re-runs the probe inside the auto-detected backend and prints `detect() -> <backend>`.
"DEFENDED" means: for an *allow* invariant the legitimate action worked (R1, R4); for a *deny*
invariant the attack failed (R2 read game / R3 write-out / R5 egress / R6 child inherits).

Two helper scripts (local): `make debug D=sec_backend_check` (what backend will I get here?) and
`make debug D=sec_probe ARGS="--sandbox"` (run + interpret the probe).

---

## macOS — deny-by-default (codex + the probe; NOT claude)

macOS uses `sandbox-exec` (Seatbelt), **deny-by-default** like the other backends. The probe
and `codex` run fully confined (`--sandbox` → R1–R4/R6 DEFENDED; only R5 egress is open, since
the model is remote on dev). **`claude` does not run under it** — `claude.exe` (a Bun binary)
crashes at the native level under deny-default Seatbelt (logs nothing; `dtruss` is SIP-blocked),
so **claude on macOS runs `runtime=none`** (the trusted dev box); claude is confined on Linux/HPC.

```bash
make debug D=sec_probe ARGS="--sandbox"                 # ALL DEFENDED (with --no-egress) on this box
make debug D=sec_agent_smoke ARGS=codex                 # codex runs inside the sandbox -> 'ok'
make debug D=sec_discover_paths ARGS=codex              # what codex touches (strace/log stream)
make debug D=sec_discover_endpoints ARGS=codex          # endpoints from the binary / lsof
```

Gotchas baked into the scripts: call `/usr/bin/log` by absolute path (`log` is a zsh builtin);
`claude.exe` is a Bun binary that SIGABRTs silently on an early denied read, so the discovery
profile seeds `(allow file-read-metadata)` first; `codex` needs `--skip-git-repo-check` (else it
exits on "untrusted dir", not a sandbox denial).

---

## Linux workstation — the strong-isolation box

Linux uses **bubblewrap** (`bwrap`): a deny-by-default mount namespace, so `--sandbox` should report
**ALL DEFENDED**.

```bash
# 1. prerequisites (this is exactly what detect() checks):
unshare -Urm true && echo USERNS_OK || echo USERNS_BLOCKED
which bwrap || sudo apt-get install -y bubblewrap
make debug D=sec_backend_check          # expect: detect() -> bwrap

# 2. verify (expect ALL DEFENDED):
make debug D=sec_probe ARGS="--sandbox"

# 3. discover what claude/codex need (strace):
make debug D=sec_discover_paths ARGS=claude
make debug D=sec_discover_endpoints ARGS=codex
```

If `USERNS_BLOCKED` (Ubuntu 23.10+/24.04 restrict unprivileged userns via AppArmor): add a
per-binary profile, then re-test —
```bash
printf 'profile bwrap /usr/bin/bwrap flags=(unconfined) {\n  userns,\n}\n' | sudo tee /etc/apparmor.d/bwrap
sudo systemctl reload apparmor && unshare -Urm true && echo FIXED
```
If userns truly cannot be enabled, fall back to Landlock (`landrun`) or treat the box like HPC (Apptainer).

> Note: regact's bwrap `deny_egress=True` uses `--unshare-net`, which **also cuts loopback**. Only use
> it when the model/env-server is local-via-unix-socket or inside the namespace. To cut external but
> keep loopback, front the agent with a domain-allowlisting HTTP(S) proxy (`HTTPS_PROXY=...`).

---

## Jean-Zay (HPC) — test on a COMPUTE node, not the login node

```bash
# 1. grab a CPU compute node (enough for the probe; GPU not needed):
srun --account=imi@cpu --qos=qos_cpu-dev --time=00:30:00 --pty bash

# 2. on the node, see which backend you get (login had bwrap; compute often disables userns):
make debug D=sec_backend_check

# 3. verify (scratch on Lustre, not $HOME):
PYTHONPATH=$WORK/regact/src python -m regact.security.probe --sandbox --json
```

- If `detect() -> bwrap` → great, expect ALL DEFENDED (and no `.sif` needed).
- If `detect() -> apptainer` (because `$SINGULARITY_ALLOWED_DIR` is set) → the probe needs an image:
  `python -m regact.security.probe --sandbox --image $SINGULARITY_ALLOWED_DIR/regact.sif`
  (build the `.sif` off-node; the login node can't build it). Without `--image` the probe prints a
  clear message and exits.
- Compute nodes have **no external internet** → R5 is DEFENDED by the network for free. Run the model
  locally (SGLang on `127.0.0.1`) and allow only loopback.

GPU node instead of CPU: `srun --account=imi@h100 -C h100 --qos=qos_gpu_h100-dev --gres=gpu:1 --cpus-per-task=24 --time=00:30:00 --pty bash`.

---

## Adastra (HPC) — login nodes lack singularity; always grab a compute node

```bash
# do NOT pass --partition or --qos (rejected; auto-assigned):
srun --account=<proj> --constraint=GENOA --cpus-per-task=16 --time=00:30:00 --pty bash   # CPU; MI250 for GPU
make debug D=sec_backend_check
PYTHONPATH=$WORKDIR/regact/src python -m regact.security.probe --sandbox --json
```

Same backend order as Jean-Zay. Compute-node egress is undocumented → the probe's R5 verdict is the
real answer; treat external egress as blocked until R5 shows otherwise, and prefer a loopback LLM.

---

## Allowlist reference (what claude / codex actually need)

Build the deny-default allowlist from these (confirm per-machine with the discover scripts):

**Filesystem (read-only unless noted):**
- system + interpreter: `/usr /bin /sbin /lib /lib64 /etc` + `/etc/ssl` / CA bundle (TLS), the real
  `node` binary (`readlink -f $(command -v node)` — under `~/.nvm|.volta` if nvm/volta), and the npm
  global tree (`npm root -g`/@anthropic-ai/claude-code, /@openai/codex — the ~215 MB native binaries).
- agent config/auth (read-write): `~/.claude` + `~/.claude.json` (+ `.credentials.json`); `~/.codex`
  (sqlite WAL). The workdir is the only broad read-write path.
- macOS extras: `/System/Library`, `/private/var/db/dyld`, `~/Library/Caches`; canonicalize `/tmp ->
  /private/tmp`, `$TMPDIR -> /private/var/folders/...`.

**Network endpoints:**
- claude: `api.anthropic.com:443` (required). Optional block: `statsig.anthropic.com`, `*.sentry.io`
  (`DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 DISABLE_AUTOUPDATER=1`).
- codex (prefer API-key mode): `api.openai.com:443` only. ChatGPT-login mode also needs
  `auth.openai.com:443` + `chatgpt.com:443` + a one-time `127.0.0.1:1455` callback.
- HPC default: a local model on `127.0.0.1:<port>/v1` (loopback only) — no external endpoint needed.

---

## What to tell your Linux mate (copy-paste)

> Pull regact, then:
> ```bash
> unshare -Urm true && echo USERNS_OK || echo USERNS_BLOCKED   # bwrap needs this
> which bwrap || sudo apt-get install -y bubblewrap
> PYTHONPATH=src python -m regact.security.probe --sandbox --json   # expect ALL DEFENDED
> ```
> If `USERNS_BLOCKED`: apply the AppArmor fix above and re-test. Then, to learn the allowlist for the
> agents on your box: `PYTHONPATH=src python debug/sec_discover_paths.py claude` (and `codex`) — send
> us the path list + the connect() endpoints; that's what we bind read-only and allow, denying the rest.
> ⚠️ bwrap has never been run end-to-end here — if R1 fails or python crashes inside the sandbox, it's a
> missing bind; send us the error and we add the path.
