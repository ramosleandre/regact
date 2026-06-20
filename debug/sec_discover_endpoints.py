"""Discover the network endpoints a code-agent CLI contacts (to build an egress allowlist).

Run:  make debug D=sec_discover_endpoints ARGS=codex    (or ARGS=claude)
codex: greps its vendored binary for https URLs (version-proof ground truth).
claude: polls lsof for established TLS sockets while it runs one turn.
"""
import glob
import re
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "codex"
    if name == "codex":
        _codex()
    elif name == "claude":
        _claude()
    else:
        print(f"use claude or codex; got {name!r}")


def _npm_root() -> str:
    try:
        return subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _codex() -> None:
    hits = glob.glob(f"{_npm_root()}/@openai/codex/node_modules/@openai/codex-*/vendor/*/bin/codex")
    if not hits:
        print("codex vendored binary not found under npm root -g")
        return
    out = subprocess.run(["strings", "-n", "8", hits[0]], capture_output=True, text=True).stdout
    # extract just the HOST (not the messy concatenated URL strings in the binary)
    hosts = sorted({h for h in re.findall(r"https://([a-z0-9.\-]+)", out)
                    if re.search(r"openai|chatgpt|sentry|statsig", h)})
    print("== codex hosts (from the binary) — allow the API ones, block telemetry ==")
    print("\n".join(hosts))


def _claude() -> None:
    binary = shutil.which("claude")
    if binary is None:
        print("claude not on PATH")
        return
    wd = tempfile.mkdtemp(prefix="sec-claude-ep-")
    proc = subprocess.Popen([binary, "-p", "say hi"], cwd=wd,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seen: set[str] = set()
    for _ in range(8):
        out = subprocess.run(["lsof", "-nP", "-i", "-a", "-p", str(proc.pid)],
                             capture_output=True, text=True).stdout
        for ln in out.splitlines():
            if "ESTABLISHED" in ln:
                seen.add(ln.split("->")[-1].split(" ")[0])
    proc.wait()
    print("== claude established TLS endpoints (lsof) ==\n" + "\n".join(sorted(seen)))


if __name__ == "__main__":
    main()
