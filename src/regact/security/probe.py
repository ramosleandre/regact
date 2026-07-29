"""Conformance probe: attempt the cheat catalog, check the contract (R1..R6).

Importable -- :func:`run_probe` returns structured results for the test suite --
and runnable -- ``python -m regact.security.probe`` -- so the same checks produce a
conformance table by hand on Jean-Zay / Adastra / Kaggle. Run it without a sandbox
(expected vulnerable) and inside a sandbox (expected defended) to verify the change.

Each check states an invariant and an ``intent``: ``deny`` checks run an *attack*
that must FAIL; ``allow`` checks run a *legitimate use* that must SUCCEED. A check
is ``defended`` when the outcome matches the intent, and the report keeps the two
vocabularies apart: a conforming deny is ``DEFENDED`` (vs ``VULNERABLE``), a
conforming allow is ``WORKS`` (vs ``BLOCKED`` — an over-restrictive sandbox, not a
breach).
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
from collections.abc import Iterator
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


def _can_connect_unix(path: str) -> tuple[bool, str]:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            sock.connect(path)
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


@contextlib.contextmanager
def loopback_listener() -> Iterator[int]:
    """Hold a listening socket on an ephemeral localhost port; yield the port.

    Stands in for the two localhost services a real run needs to reach — the env
    server and a local model endpoint — so R4 can be checked without either being up.
    A pending connection completes from the kernel backlog, so nothing has to accept.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        yield int(sock.getsockname()[1])
    finally:
        sock.close()


@contextlib.contextmanager
def bridged_socket_listener() -> Iterator[str]:
    """Hold a pathname unix listener in its own short dir; yield the socket path.

    Stands in for a bridged host service (see :mod:`regact.security.netbridge`): a
    socket file crosses a network namespace because it is a filesystem object, so
    this is the sanctioned loopback path on a backend that severs host TCP.
    """
    import shutil

    socket_dir = tempfile.mkdtemp(prefix="regact-net-")
    path = os.path.join(socket_dir, "probe.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        sock.listen(1)
        yield path
    finally:
        sock.close()
        shutil.rmtree(socket_dir, ignore_errors=True)


def run_probe(
    *,
    workdir: str,
    secret_path: str,
    external: tuple[str, int] = ("example.com", 443),
    check_egress: bool = True,
    loopback_port: int | None = None,
    loopback_want: str = "allow",
    bridge_socket: str | None = None,
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
    if loopback_port is not None:
        add(
            Invariant.R4_LOOPBACK,
            "L1 host loopback, direct TCP",
            loopback_want,
            _can_connect("127.0.0.1", loopback_port),
        )
    if bridge_socket is not None:
        add(
            Invariant.R4_LOOPBACK,
            "L2 host service via bridged socket",
            "allow",
            _can_connect_unix(bridge_socket),
        )
    # R5 (deny): external egress must fail (scored runs).
    if check_egress:
        add(Invariant.R5_EGRESS, "E1 external internet egress", "deny", _can_connect(*external))
    # R6 (deny): a child process inherits the restriction.
    child = _child_can_read(secret_path)
    add(Invariant.R6_NO_ESCAPE, "G1 child reads the game secret", "deny", child)

    return out


def format_report(results: list[CheckResult]) -> str:
    """One line per check, with intent-specific verdicts.

    ``deny`` checks are security: a pass is ``DEFENDED``, a failure a ``VULNERABLE``
    breach. ``allow`` checks are liveness: a pass is ``WORKS``, a failure a
    ``BLOCKED`` sanctioned path (an unusable sandbox, not a breach).
    """
    lines = [f"{'INV':<4} {'CHECK':<44} {'WANT':<6} VERDICT", "-" * 80]
    breaches = blocked = 0
    for r in results:
        if r.defended:
            verdict = "DEFENDED" if r.intent == "deny" else "WORKS"
        elif r.intent == "deny":
            verdict = "*** VULNERABLE ***"
            breaches += 1
        else:
            verdict = "*** BLOCKED ***"
            blocked += 1
        lines.append(f"{r.invariant:<4} {r.name:<44} {r.intent:<6} {verdict:<18} ({r.detail})")
    lines.append("-" * 80)
    if not breaches and not blocked:
        lines.append("GLOBAL: CONTRACT HELD (attacks defended, sanctioned paths work)")
    else:
        parts = [f"{breaches} BREACH(ES)"] if breaches else []
        parts += [f"{blocked} BLOCKED PATH(S)"] if blocked else []
        lines.append("GLOBAL: " + ", ".join(parts))
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="regact sandbox conformance probe")
    # default=None then mkdtemp lazily: an eager default would call mkdtemp even when
    # --workdir is passed, which fails inside a deny-default sandbox (no temp dir).
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--secret", default=os.environ.get("REGACT_PROBE_SECRET"))
    parser.add_argument("--no-egress", action="store_true", help="skip the external-egress check")
    parser.add_argument(
        "--sandbox", action="store_true", help="re-run this probe inside the detected OS sandbox"
    )
    parser.add_argument("--image", default=None, help="apptainer/singularity .sif image (HPC)")
    parser.add_argument(
        "--loopback-port",
        type=int,
        default=None,
        help="check R4 by connecting to this localhost port (set by --sandbox)",
    )
    parser.add_argument(
        "--loopback-want",
        choices=("allow", "deny"),
        default="allow",
        help="whether direct host TCP must work or must fail (set by --sandbox per backend)",
    )
    parser.add_argument(
        "--bridge-socket",
        default=None,
        help="check R4's bridged path by connecting to this socket file (set by --sandbox)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    workdir = args.workdir or tempfile.mkdtemp(prefix="regact_probe_")
    secret = args.secret
    if not secret:
        # the stand-in secret lives in its own dir, so --sandbox can forbid that dir
        # without touching the workdir or the rest of the temp tree.
        secret = os.path.join(tempfile.mkdtemp(prefix="regact_probe_secret_"), "game_secret.py")
        with open(secret, "w", encoding="utf-8") as handle:
            handle.write("WINNING = [3, 1, 2, 0]  # the game answer\n")

    if args.sandbox:
        return _rerun_sandboxed(
            workdir, secret, no_egress=args.no_egress, as_json=args.json, image=args.image
        )

    results = run_probe(
        workdir=workdir,
        secret_path=secret,
        check_egress=not args.no_egress,
        loopback_port=args.loopback_port,
        loopback_want=args.loopback_want,
        bridge_socket=args.bridge_socket,
    )
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(format_report(results))
    return 0 if all(r.defended for r in results) else 1


def _rerun_sandboxed(
    workdir: str, secret: str, *, no_egress: bool, as_json: bool, image: str | None = None
) -> int:
    """Re-exec this probe inside the auto-detected sandbox, forbidding the secret's dir."""
    import subprocess

    import regact
    from regact.security.runtime import SandboxRuntime, detect, wrap_argv

    backend = detect()
    print(f"detect() -> {backend.value}")
    if backend is SandboxRuntime.APPTAINER and not image:
        print(
            "apptainer needs a .sif image: pass --image PATH (build it off-node), "
            "or run where bwrap/seatbelt is available."
        )
        return 2

    src = os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))
    child = [sys.executable, "-m", "regact.security.probe"]
    child += ["--workdir", workdir, "--secret", secret]
    if no_egress:
        child.append("--no-egress")
    if as_json:
        child.append("--json")
    deny_egress = not no_egress
    severs_loopback = backend is SandboxRuntime.BWRAP and deny_egress
    with (
        loopback_listener() as port,  # stands in for the env server / a local model
        bridged_socket_listener() as bridge,  # stands in for the same service, bridged
    ):
        child += ["--loopback-port", str(port)]
        child += ["--loopback-want", "deny" if severs_loopback else "allow"]
        child += ["--bridge-socket", bridge]
        argv = wrap_argv(
            backend,
            child,
            workdir=workdir,
            allow_read=[src],
            deny_egress=deny_egress,
            allow_rw=[bridge],
            image=image,
        )
        # TMPDIR inside the (allowed) workdir: scratch for the child without exposing /tmp.
        env = {**os.environ, "PYTHONPATH": src, "TMPDIR": workdir}
        return subprocess.run(argv, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(_main())
