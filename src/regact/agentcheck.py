"""``python -m regact.agentcheck`` — can this agent actually launch here?

Answers a question the conformance probe structurally cannot: the probe only ever
wraps the interpreter, which is always inside the sandbox's allowlist, so it reports
ALL DEFENDED while a backend's real executable (``claude`` under ``~/.local``,
``codex`` under ``~/.codex``) is absent from that allowlist and dies with
``execvp: No such file``. Here the *backend's own* argv is wrapped and run.

It sweeps a small matrix per agent — unwrapped, sandboxed, sandboxed + deny_egress —
so a failure names the knob that caused it instead of a whole run failing later with
a bare error. Agent-agnostic: each backend declares what to launch via
``CodeAgent.launch_probe_argv()``; nothing here is written per backend.

    python -m regact.agentcheck --all
    python -m regact.agentcheck --agent claude --agent codex --verbose
    python -m regact.agentcheck --all --json          # for a job log
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass

from regact.agent.base import build_agent
from regact.config.schema import AgentConfig, AgentName
from regact.security.runtime import SandboxRuntime, detect, wrap_argv

_LAUNCH_TIMEOUT_S = 60.0

# The matrix each agent is swept through: (label, wrap in the sandbox, deny egress).
_MODES: tuple[tuple[str, bool, bool], ...] = (
    ("bare", False, False),
    ("sandbox", True, False),
    ("sandbox+deny_egress", True, True),
)


@dataclass(frozen=True)
class LaunchResult:
    """One (agent, mode) attempt."""

    agent: str
    mode: str
    ok: bool
    detail: str
    argv: list[str]
    stderr: str = ""


def _resolve(argv: list[str]) -> tuple[str | None, str]:
    """The executable's real path (symlinks followed) and a human note about it."""
    if not argv:
        return None, "in-process backend: nothing to launch"
    found = shutil.which(argv[0])
    if found is None:
        return None, f"{argv[0]!r} is not on PATH"
    return os.path.realpath(found), os.path.realpath(found)


def _run(argv: list[str], *, cwd: str) -> tuple[bool, str, str]:
    """Run argv; return (succeeded, short detail, stderr tail)."""
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=_LAUNCH_TIMEOUT_S, check=False
        )
    except FileNotFoundError as exc:
        return False, f"not found: {exc.filename}", ""
    except subprocess.TimeoutExpired:
        return False, f"timed out after {_LAUNCH_TIMEOUT_S:.0f}s", ""
    except OSError as exc:
        return False, type(exc).__name__, ""
    if proc.returncode == 0:
        first = (proc.stdout or "").strip().splitlines()
        return True, first[0][:60] if first else "exit 0", ""
    return False, f"exit {proc.returncode}", (proc.stderr or "").strip()[-400:]


def check_agent(
    name: AgentName, *, workdir: str, modes: tuple[str, ...] | None = None
) -> list[LaunchResult]:
    """Sweep one backend through the matrix; empty for an in-process backend."""
    agent = build_agent(AgentConfig(name=name))
    argv = agent.launch_probe_argv()
    if not argv:
        return []

    results: list[LaunchResult] = []
    for label, sandboxed, deny_egress in _MODES:
        if modes is not None and label not in modes:
            continue
        launched = argv
        if sandboxed:
            launched = wrap_argv(
                detect(),
                argv,
                workdir=workdir,
                allow_read=agent.host_read_paths(),
                deny_egress=deny_egress,
                allow_write_prefixes=agent.host_write_prefixes(),
            )
        ok, detail, stderr = _run(launched, cwd=workdir)
        results.append(LaunchResult(str(name.value), label, ok, detail, list(launched), stderr))
    return results


def format_report(results: list[LaunchResult], *, verbose: bool) -> str:
    """The table a human reads; ``verbose`` adds the wrapped argv and stderr."""
    lines = [f"{'AGENT':<18} {'MODE':<20} VERDICT", "-" * 78]
    for r in results:
        lines.append(f"{r.agent:<18} {r.mode:<20} {'OK  ' if r.ok else 'FAIL'}  ({r.detail})")
        if verbose:
            lines.append(f"    argv: {' '.join(r.argv)}")
            if r.stderr:
                lines.append(f"    stderr: {r.stderr}")
    failed = [r for r in results if not r.ok]
    lines.append("-" * 78)
    lines.append("GLOBAL: " + ("ALL LAUNCHED" if not failed else f"{len(failed)} FAILURE(S)"))
    return "\n".join(lines)


def _selected(args: argparse.Namespace) -> list[AgentName]:
    if args.all or not args.agent:
        return list(AgentName)
    return [AgentName(a) for a in args.agent]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regact.agentcheck", description="Can the configured agents launch on this host?"
    )
    parser.add_argument(
        "--agent", action="append", choices=[a.value for a in AgentName], help="repeatable"
    )
    parser.add_argument("--all", action="store_true", help="check every known backend")
    parser.add_argument("--workdir", default=None, help="where to launch (default: a temp dir)")
    parser.add_argument("--verbose", action="store_true", help="print the argv and stderr")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    import tempfile

    workdir = args.workdir or tempfile.mkdtemp(prefix="regact_agentcheck_")
    print(f"sandbox backend: {detect().value}   workdir: {workdir}")
    if detect() is SandboxRuntime.NONE:
        print("note: no sandbox backend here — the sandboxed rows repeat the bare one.")

    results: list[LaunchResult] = []
    skipped: list[str] = []
    for name in _selected(args):
        rows = check_agent(name, workdir=workdir)
        results.extend(rows)
        if not rows:
            skipped.append(name.value)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(format_report(results, verbose=args.verbose))
        if skipped:
            print(f"skipped (in-process, nothing to launch): {', '.join(skipped)}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
