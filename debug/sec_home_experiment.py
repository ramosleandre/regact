"""Experiment: does HOME=<workdir> let an agent run under a DENY-DEFAULT macOS sandbox?

Point the agent's HOME into its (allowed) workdir + copy its auth there, then deny
everything else (incl. the real $HOME where the game lives). If the agent never reads
a *denied* path, the Bun binary should not abort -> deny-default could work on macOS.

Run:  make debug D=sec_home_experiment ARGS=claude     (or codex)
"""
import os
import shutil
import subprocess
import sys
import tempfile


def _prefixes():
    return sorted({os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)})


def _deny_default(workdir):
    ro = [p for p in ("/usr", "/System", "/Library", "/private/var/db/dyld",
                      "/private/etc", "/opt", "/bin", "/sbin", *_prefixes()) if os.path.exists(p)]
    return "".join([
        "(version 1)(deny default)(allow process*)(allow sysctl-read)(allow mach-lookup)",
        "(allow file-read-metadata)",
        '(allow file-read* (literal "/") ' + " ".join(f'(subpath "{p}")' for p in ro) + ")",
        f'(allow file* (subpath "{workdir}") (subpath "/dev"))',
        "(allow network*)",
    ])


def _seed(real_home, fake_home, agent):
    # copy only the auth + config FILES (robust: skips missing, avoids broken symlinks)
    items = {
        "claude": [".claude.json", ".claude/.credentials.json", ".claude/settings.json"],
        "codex": [".codex/auth.json", ".codex/config.toml"],
    }[agent]
    for rel in items:
        src, dst = os.path.join(real_home, rel), os.path.join(fake_home, rel)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def _run(agent, binary, env, profile=None):
    prompt = ["-p", "say hi"] if agent == "claude" else ["exec", "--skip-git-repo-check", "say hi"]
    cmd = (["sandbox-exec", "-p", profile] if profile else []) + [binary, *prompt]
    try:
        return subprocess.run(cmd, env=env, cwd=env["HOME"], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None


def main():
    agent = sys.argv[1] if len(sys.argv) > 1 else "claude"
    binary = shutil.which(agent)
    if not binary:
        print(f"{agent} not on PATH")
        return
    workdir = tempfile.mkdtemp(prefix=f"home-exp-{agent}-")
    os.makedirs(os.path.join(workdir, "tmp"), exist_ok=True)
    _seed(os.path.expanduser("~"), workdir, agent)
    env = {**os.environ, "HOME": workdir, "TMPDIR": os.path.join(workdir, "tmp")}
    env.pop("CLAUDE_CONFIG_DIR", None)
    print(f"workdir(fake HOME) = {workdir}")
    for label, profile in [
        ("A) HOME=workdir, NO sandbox (does auth survive the redirect?)", None),
        ("B) HOME=workdir, DENY-DEFAULT sandbox (does it survive?)", _deny_default(workdir)),
    ]:
        r = _run(agent, binary, env, profile)
        if r is None:
            print(f"== {label} -> TIMEOUT")
            continue
        print(f"== {label}")
        print(f"   rc={r.returncode} | out: {r.stdout.strip()[:200]!r} | err: {r.stderr.strip()[:160]!r}")


if __name__ == "__main__":
    main()
