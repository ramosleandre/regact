"""``python -m regact.doctor`` — can this machine run regact?

The first thing to run on a new host. It reports one ``[ ok ]`` / ``[warn]`` /
``[fail]`` line per capability so a user learns what works BEFORE launching a run. It
never raises: a missing optional piece is a ``warn``; only a broken core (wrong Python,
package not importable) is a ``fail`` and sets the exit code.

For each agent backend it prints the executable's **resolved real path** and the version
it actually reports — not merely "present on PATH". Both matter: a sandbox allow-list is
built from real paths (a CLI living under ``~/.local/share/...`` is the classic reason a
confined run dies with ``execvp``), and a binary can be on PATH yet fail to execute.

Three diagnostics, one question each — this one is presence:
  * ``regact.doctor``          is everything installed, and where?
  * ``regact.agentcheck``      do the backends launch *under the sandbox*?
  * ``regact.security.probe``  does the sandbox honor the R1-R6 contract?

``--auth`` adds a per-backend authentication check (spends one trivial LLM call per CLI
agent), catching the "present but not logged in" case a plain ``--version`` misses.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

_COLOR = {
    OK: "\033[32m[ ok ]\033[0m",
    WARN: "\033[33m[warn]\033[0m",
    FAIL: "\033[31m[fail]\033[0m",
    SKIP: "[ -- ]",
}
_VERSION_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Row:
    """One reported capability."""

    section: str
    name: str
    status: str
    detail: str


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _core_rows() -> list[Row]:
    """Python version and the package itself — the only checks that can fail hard."""
    version = sys.version.split()[0]
    py_ok = sys.version_info[:2] in ((3, 11), (3, 12))
    rows = [Row("core", "python 3.11 / 3.12", OK if py_ok else FAIL, f"found {version}")]

    try:
        import regact

        rows.append(Row("core", "regact importable", OK, os.path.dirname(regact.__file__)))
    except ImportError:
        rows.append(
            Row("core", "regact importable", FAIL, 'run `pip install -e ".[dev]"` in the repo')
        )
    return rows


def _sandbox_rows() -> list[Row]:
    """Which OS sandbox backends exist here; none means runs are unconfined."""
    backends = (
        ("sandbox-exec", "macOS Seatbelt"),
        ("bwrap", "Linux bubblewrap"),
    )
    rows = [
        Row("sandbox", binary, OK, f"{label} — {shutil.which(binary)}")
        for binary, label in backends
        if shutil.which(binary)
    ]
    if not rows:
        rows.append(Row("sandbox", "no backend", WARN, "runs unconfined (sandbox=false only)"))
    try:
        from regact.security.runtime import SandboxRuntime, detect, userns_ok
    except ImportError:
        return rows

    chosen = detect()
    if chosen is not SandboxRuntime.NONE:
        rows.append(Row("sandbox", "auto resolves to", OK, chosen.value))
        return rows

    # An installed-but-unused backend is the confusing case: say why, or the report reads
    # "bwrap ok" next to "sandbox none" with no explanation and runs go unconfined.
    reason = "no usable backend on this host"
    if shutil.which("bwrap") and not userns_ok():
        reason = (
            "bwrap is installed but unprivileged user namespaces are blocked "
            "(Ubuntu 23.10+: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0)"
        )
    rows.append(Row("sandbox", "auto resolves to", WARN, f"none — {reason}"))
    return rows


def _version_of(argv: list[str]) -> tuple[str, str]:
    """Run the backend's own probe argv; return (status, detail)."""
    binary = shutil.which(argv[0]) or (argv[0] if os.path.isabs(argv[0]) else None)
    if binary is None:
        return WARN, f"{argv[0]!r} not on PATH"
    real = os.path.realpath(binary)
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_VERSION_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return WARN, f"{real} — cannot execute ({type(exc).__name__})"
    if proc.returncode != 0:
        return WARN, f"{real} — exited {proc.returncode}"
    first = (proc.stdout or "").strip().splitlines()
    return OK, f"{first[0][:40] if first else 'ok'} — {real}"


def _agent_rows(check_auth: bool = False) -> list[Row]:
    """Every registered backend, asked what it needs to launch (no hardcoded CLI list).

    Uses ``CodeAgent.launch_probe_argv``, so a newly added backend is covered here the
    day it declares one — nothing in this module names a specific agent. With
    ``check_auth``, each backend that offers an ``auth_check`` is also asked whether it
    can actually authenticate (present-but-not-logged-in is the failure a version check
    cannot see); this spends one trivial LLM call per CLI agent.
    """
    try:
        from regact.agent.base import build_agent
        from regact.config.schema import AgentConfig, AgentName
    except ImportError:
        return [Row("agent backends", "registry", FAIL, "regact is not importable")]

    rows = []
    for name in AgentName:
        agent = build_agent(AgentConfig(name=name))
        argv = agent.launch_probe_argv()
        if not argv:
            rows.append(Row("agent backends", name.value, SKIP, "in-process (nothing to launch)"))
            continue
        status, detail = _version_of(argv)
        rows.append(Row("agent backends", name.value, status, detail))
        if check_auth and status == OK and (auth := agent.auth_check()) is not None:
            rows.append(Row("agent backends", f"{name.value} auth", auth[0], auth[1]))
    return rows


def _extras_rows() -> list[Row]:
    """The optional game libraries, named by the extra that installs them."""
    extras = (("arc_agi", "arc"), ("minigrid", "minigrid"))
    return [
        Row(
            "game extras",
            module,
            OK if _module_present(module) else WARN,
            "installed" if _module_present(module) else f'pip install ".[{extra}]"',
        )
        for module, extra in extras
    ]


def _endpoint_row(url: str) -> Row:
    """Is a model endpoint actually reachable? (the usual HPC failure: it is not up yet)."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.create_connection((host, port), timeout=5).close()
    except OSError as exc:
        return Row("endpoint", url, WARN, f"unreachable ({type(exc).__name__})")
    return Row("endpoint", url, OK, f"reachable on {host}:{port}")


def collect(endpoint: str | None = None, check_auth: bool = False) -> list[Row]:
    """Every check, in report order."""
    rows = [*_core_rows(), *_sandbox_rows(), *_agent_rows(check_auth), *_extras_rows()]
    if endpoint:
        rows.append(_endpoint_row(endpoint))
    return rows


def format_report(rows: list[Row]) -> str:
    """Group the rows by section, one aligned line each."""
    lines: list[str] = ["regact doctor — machine readiness"]
    section = ""
    for row in rows:
        if row.section != section:
            section = row.section
            lines.append(f"\n{section}")
        lines.append(f"  {_COLOR[row.status]} {row.name:<18} {row.detail}")
    fails = sum(1 for r in rows if r.status == FAIL)
    lines.append("")
    lines.append(
        f"{_COLOR[FAIL]} {fails} core check(s) failed — fix these before running."
        if fails
        else f"{_COLOR[OK]} core is ready. Warnings are optional pieces, install as needed."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regact.doctor", description="Report what this machine can run."
    )
    parser.add_argument("--endpoint", default=None, help="also check a model URL is reachable")
    parser.add_argument(
        "--auth", action="store_true", help="also check each agent CAN log in (spends 1 call/agent)"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable (for a job log)")
    args = parser.parse_args(argv)

    rows = collect(args.endpoint, check_auth=args.auth)
    print(json.dumps([asdict(r) for r in rows], indent=2) if args.json else format_report(rows))
    return 1 if any(r.status == FAIL for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
