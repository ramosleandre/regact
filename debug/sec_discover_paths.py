"""Discover the filesystem paths a code-agent CLI touches (to build an allowlist).

Run:  make debug D=sec_discover_paths ARGS=claude     (or ARGS=codex)
Linux: traces the CLI with strace, prints the unique opened paths + connect() lines.
macOS: runs it under a deny-default sandbox-exec while capturing Sandbox denials.
Output = raw material for allow_read=[...] / the bwrap bind list. (Run authenticated.)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

_ARGS = {"claude": ["-p", "say hi"], "codex": ["exec", "--skip-git-repo-check", "say hi"]}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "claude"
    binary = shutil.which(name)
    if name not in _ARGS or binary is None:
        print(f"need an installed agent on PATH; got {name!r} -> {binary}")
        return
    wd = tempfile.mkdtemp(prefix=f"sec-{name}-")
    (_linux if sys.platform.startswith("linux") else _macos)(binary, name, wd)


def _linux(binary: str, name: str, wd: str) -> None:
    log = os.path.join(wd, "trace.log")
    cmd = ["strace", "-f", "-e", "trace=openat,open,connect,socket", "-yy", "-s", "256",
           "-o", log, binary, *_ARGS[name]]
    subprocess.run(cmd, cwd=wd, stdin=subprocess.DEVNULL, capture_output=True)
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    paths = sorted({ln.split('"')[1] for ln in text.splitlines()
                    if ("open(" in ln or "openat(" in ln) and "= -1" not in ln and '"' in ln})
    conns = sorted({ln for ln in text.splitlines() if "connect(" in ln and "inet" in ln})
    print(f"== opened paths ({len(paths)}) — ALLOW these ==\n" + "\n".join(paths))
    print(f"\n== connect() ({len(conns)}) — egress endpoints ==\n" + "\n".join(conns))


def _macos(binary: str, name: str, wd: str) -> None:
    denials = os.path.join(wd, "denials.log")
    # Seed the system layer so the Bun/node binary gets PAST its early aborts (it does
    # file-read-DATA on "/" and dyld and SIGABRTs otherwise); the REMAINING denials are
    # then the app-specific paths worth allowlisting (~/.claude, ~/.codex, caches, ...).
    profile = (
        '(version 1)(deny default)(debug deny)(allow process*)(allow sysctl-read)'
        ' (allow file-read* (literal "/") (subpath "/usr") (subpath "/System")'
        ' (subpath "/Library") (subpath "/private/var/db/dyld") (subpath "/opt/homebrew")'
        ' (subpath "/dev") (subpath "/private/etc"))'
    )
    handle = open(denials, "w")
    streamer = subprocess.Popen(
        ["/usr/bin/log", "stream", "--style", "compact", "--predicate", 'sender == "Sandbox"'],
        stdout=handle, stderr=subprocess.STDOUT)
    time.sleep(1)
    subprocess.run(["sandbox-exec", "-p", profile, binary, *_ARGS[name]],
                   cwd=wd, stdin=subprocess.DEVNULL, capture_output=True)
    time.sleep(2)
    streamer.terminate()
    handle.close()
    lines = sorted({ln for ln in open(denials, errors="replace").read().splitlines() if "deny(" in ln})
    print(f"== sandbox denials ({len(lines)}) — these paths/ops to ALLOW ==\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
