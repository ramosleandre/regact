"""Conformance probe: attempt the cheat catalog, check the contract (R1..R6).

Importable -- :func:`run_probe` returns structured results for the test suite --
and runnable -- ``python -m regact.security.probe`` -- so the same checks produce a
conformance table by hand on Jean-Zay / Adastra / Kaggle. Run it without a sandbox
(expected vulnerable) and inside a sandbox (expected defended) to verify the change.

Each check states an invariant and an ``intent``: ``deny`` checks run an *attack*
that must FAIL; ``allow`` checks run a *legitimate use* that must SUCCEED. A check
is ``defended`` when the outcome matches the intent.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass

from regact.security.contract import Invariant


@dataclass(frozen=True)
class CheckResult:
    invariant: str
    name: str
    intent: str  # "deny" (attack must fail) | "allow" (use must work)
    defended: bool
    detail: str


def _can_read(path: str) -> tuple[bool, str]:
    try:
        with open(path, "rb") as handle:
            return True, f"read {len(handle.read(64))}B"
    except OSError as exc:
        return False, type(exc).__name__


def _can_list(path: str) -> tuple[bool, str]:
    try:
        return True, f"{len(os.listdir(path))} entries"
    except OSError as exc:
        return False, type(exc).__name__


def _can_connect(host: str, port: int) -> tuple[bool, str]:
    try:
        socket.create_connection((host, port), timeout=3).close()
        return True, "connected"
    except OSError as exc:
        return False, type(exc).__name__


def _can_write(path: str) -> tuple[bool, str]:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
    except OSError as exc:
        return False, type(exc).__name__
    with contextlib.suppress(OSError):
        os.unlink(path)
    return True, "wrote"


def _child_can_read(path: str) -> tuple[bool, str]:
    """Spawn a child process that reads ``path`` — it must inherit the restriction."""
    code = "import sys; open(sys.argv[1]).read()"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, path], capture_output=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, type(exc).__name__
    return result.returncode == 0, "child read it" if result.returncode == 0 else "child denied"


def _check(inv: Invariant, name: str, intent: str, attacked: bool, detail: str) -> CheckResult:
    defended = (not attacked) if intent == "deny" else attacked
    return CheckResult(str(inv), name, intent, defended, detail)


def run_probe(
    *,
    workdir: str,
    secret_path: str,
    external: tuple[str, int] = ("example.com", 443),
    check_egress: bool = True,
) -> list[CheckResult]:
    """Attempt the catalog from THIS process; return one result per check."""
    own = os.path.join(workdir, "solution.py")
    if not os.path.exists(own):
        with contextlib.suppress(OSError), open(own, "w", encoding="utf-8") as handle:
            handle.write("# the agent's own work\n")

    game_dir = os.path.dirname(os.path.realpath(secret_path))
    home = os.path.expanduser("~")
    out: list[CheckResult] = []

    def add(inv: Invariant, name: str, intent: str, probe: tuple[bool, str]) -> None:
        out.append(_check(inv, name, intent, probe[0], probe[1]))

    # R1 (allow): the agent must be able to work in its own workdir.
    add(Invariant.R1_WORKDIR, "read own workdir", "allow", _can_read(own))
    # R2 (deny): read / enumerate the game, or escape to an unrelated location.
    add(Invariant.R2_SECRET, "A1 open(game secret)", "deny", _can_read(secret_path))
    add(Invariant.R2_SECRET, "A8 list the game directory", "deny", _can_list(game_dir))
    add(Invariant.R2_SECRET, "A8 reach the user home dir", "deny", _can_list(home))
    # R3 (deny): writing outside the workdir (here, into the game dir) must fail.
    wrote = _can_write(os.path.join(game_dir, ".probe_write"))
    add(Invariant.R3_WRITE, "R3 write outside the workdir", "deny", wrote)
    # R5 (deny): external egress must fail (scored runs).
    if check_egress:
        add(Invariant.R5_EGRESS, "E1 external internet egress", "deny", _can_connect(*external))
    # R6 (deny): a child process inherits the restriction.
    child = _child_can_read(secret_path)
    add(Invariant.R6_NO_ESCAPE, "G1 child reads the game secret", "deny", child)

    return out


def format_report(results: list[CheckResult]) -> str:
    lines = [f"{'INV':<4} {'CHECK':<44} {'WANT':<6} VERDICT", "-" * 80]
    breaches = 0
    for r in results:
        verdict = "DEFENDED" if r.defended else "*** VULNERABLE ***"
        breaches += 0 if r.defended else 1
        lines.append(f"{r.invariant:<4} {r.name:<44} {r.intent:<6} {verdict:<18} ({r.detail})")
    lines.append("-" * 80)
    lines.append("GLOBAL: " + ("ALL DEFENDED" if breaches == 0 else f"{breaches} BREACH(ES)"))
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regact sandbox conformance probe")
    # a fresh scratch workdir by default, so running the probe never pollutes cwd
    parser.add_argument("--workdir", default=tempfile.mkdtemp(prefix="regact_probe_"))
    parser.add_argument("--secret", default=os.environ.get("REGACT_PROBE_SECRET"))
    parser.add_argument("--no-egress", action="store_true", help="skip the external-egress check")
    parser.add_argument(
        "--sandbox", action="store_true", help="re-run this probe inside the detected OS sandbox"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    secret = args.secret
    if not secret:
        # the stand-in secret lives in its own dir, so --sandbox can forbid that dir
        # without touching the workdir or the rest of the temp tree.
        secret = os.path.join(tempfile.mkdtemp(prefix="regact_probe_secret_"), "game_secret.py")
        with open(secret, "w", encoding="utf-8") as handle:
            handle.write("WINNING = [3, 1, 2, 0]  # the game answer\n")

    if args.sandbox:
        return _rerun_sandboxed(args.workdir, secret, no_egress=args.no_egress, as_json=args.json)

    results = run_probe(workdir=args.workdir, secret_path=secret, check_egress=not args.no_egress)
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(format_report(results))
    return 0 if all(r.defended for r in results) else 1


def _rerun_sandboxed(workdir: str, secret: str, *, no_egress: bool, as_json: bool) -> int:
    """Re-exec this probe inside the auto-detected sandbox, forbidding the secret's dir."""
    import subprocess

    import regact
    from regact.security.runtime import detect, wrap_argv

    src = os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))
    child = [sys.executable, "-m", "regact.security.probe"]
    child += ["--workdir", workdir, "--secret", secret]
    if no_egress:
        child.append("--no-egress")
    if as_json:
        child.append("--json")
    argv = wrap_argv(
        detect(), child, workdir=workdir, allow_read=[src], forbid_read=[os.path.dirname(secret)]
    )
    return subprocess.run(argv, env={**os.environ, "PYTHONPATH": src}, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(_main())
