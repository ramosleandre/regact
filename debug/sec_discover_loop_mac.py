"""Iteratively LEARN a deny-default Seatbelt allowlist for an agent CLI (macOS).

Run:  make debug D=sec_discover_loop_mac ARGS=codex     (or ARGS=claude)

Runs the CLI under deny-default sandbox-exec, reads each Sandbox denial from
`log stream`, auto-appends an (allow ...) rule, and RE-RUNS — until the CLI exits 0
(fully ran) or no new denial appears. Prints the converged allowlist.

This is the macOS analogue of Linux `strace` discovery. macOS blocks the easy
"observe-without-blocking" tools (dtruss/fs_usage need SIP off and cannot trace
hardened binaries like claude.exe), so here we discover by block-then-allow instead.
codex (node) reveals many denials per run -> converges fast. claude.exe (Bun) aborts
on the FIRST denied op each run -> slow but still converges (early runs never reach
the network, so they make no API call; only the final converged run does).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

_ARGS = {"claude": ["-p", "say hi"], "codex": ["exec", "--skip-git-repo-check", "say hi"]}
_AGENT_PROCS = {"claude.exe", "claude", "node", "env", "codex", "bun"}
_SEED = [
    "(allow process*)",
    "(allow sysctl-read)",
    "(allow mach-lookup)",
    '(allow file-read* (literal "/") (subpath "/usr") (subpath "/System")'
    ' (subpath "/Library") (subpath "/private/var/db/dyld") (subpath "/dev") (subpath "/private/etc") (subpath "/opt"))',
]
_MAX_ITERS = 30
_LINE = re.compile(r"Sandbox: (\S+)\(\d+\) deny\(\d+\) (\S+) (.+)$")


def _rule_for(op: str, target: str) -> str | None:
    if op.startswith("file-write") or op == "file*":
        return f'(allow file* (subpath "{os.path.dirname(target) or target}"))'
    if op.startswith("file-read"):
        return f'(allow file-read* (subpath "{os.path.dirname(target) or target}"))'
    if op.startswith("network"):
        return "(allow network*)"
    return None  # mach-lookup / sysctl already broadly seeded


def _denials(logfile: str) -> list[tuple[str, str]]:
    out = []
    for ln in open(logfile, errors="replace").read().splitlines():
        m = _LINE.search(ln)
        if m and m.group(1) in _AGENT_PROCS:
            out.append((m.group(2), m.group(3).strip()))
    return out


def main() -> None:
    if sys.platform != "darwin":
        print("macOS only. On Linux use sec_discover_paths (strace observes without blocking).")
        return
    name = sys.argv[1] if len(sys.argv) > 1 else "codex"
    binary = shutil.which(name)
    if name not in _ARGS or not binary:
        print(f"need claude/codex on PATH; got {name!r} -> {binary}")
        return
    wd = tempfile.mkdtemp(prefix=f"sec-loop-{name}-")
    rules = [*_SEED, f'(allow file* (subpath "{wd}"))']
    seen: set[tuple[str, str]] = set()
    for i in range(_MAX_ITERS):
        profile = "(version 1)(deny default)(debug deny) " + " ".join(rules)
        logf = os.path.join(wd, f"d{i}.log")
        handle = open(logf, "w")
        streamer = subprocess.Popen(
            ["/usr/bin/log", "stream", "--style", "compact", "--predicate", 'sender == "Sandbox"'],
            stdout=handle, stderr=subprocess.STDOUT)
        time.sleep(0.8)
        r = subprocess.run(["sandbox-exec", "-p", profile, binary, *_ARGS[name]],
                           cwd=wd, stdin=subprocess.DEVNULL, capture_output=True)
        time.sleep(1.2)
        streamer.terminate()
        handle.close()
        new = []
        for op, target in _denials(logf):
            if (op, target) in seen:
                continue
            seen.add((op, target))
            rule = _rule_for(op, target)
            if rule and rule not in rules:
                rules.append(rule)
                new.append(f"{op} {target}")
        status = "EXIT 0 (ran fully)" if r.returncode == 0 else f"exit {r.returncode}"
        print(f"iter {i:2d}: {status}; +{len(new)} new" + (f" :: {', '.join(new[:5])}" if new else ""))
        # Stop as soon as the agent RAN (exit 0): the allowlist is now sufficient.
        # Remaining denials are non-fatal (the agent ignored them) and vary per run, so
        # chasing "zero denials" would loop forever. Keep iterating only while it FAILS.
        if r.returncode == 0:
            print(f"\n== CONVERGED (agent ran) — allowlist ({len(rules)} rules) ==\n" + "\n".join(rules))
            return
        if not new:
            print(f"\n== stuck: exit {r.returncode}, no new path denials (likely auth/network, "
                  f"not a path) — allowlist ({len(rules)} rules) ==\n" + "\n".join(rules))
            return
    print(f"\n== hit max iters — allowlist so far ({len(rules)} rules) ==\n" + "\n".join(rules))


if __name__ == "__main__":
    main()
