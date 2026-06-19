"""Select and apply an OS sandbox around the agent's subprocess.

Defends against an agent that runs arbitrary code (e.g.
``python3 -c "open('/abs/path').read()"``) to read files outside its workdir. The
defense is a filesystem allowlist (paths outside the workdir are absent from the
agent's view) rather than a blocklist of techniques (which cannot enumerate every
trick). :func:`wrap_argv` is the single integration point; it prepends the right
per-platform launcher and the rest of regact does not depend on which backend runs.

Agnostic: this module takes only paths and an argv -- no agent, environment, or
feature types -- so it never couples to a backend or a problem.

Backends:
  ``none``       no sandbox: used when none is configured or available.
  ``seatbelt``   macOS ``sandbox-exec``: deny paths outside the workdir, allow the workdir.
  ``bwrap``      Linux bubblewrap mount namespace: bind only the allowed paths.
  ``apptainer``  HPC Singularity/Apptainer SIF (``--containall --no-home``).
  ``landlock``   Linux Landlock LSM: declared; needs a helper, not yet implemented.

``wrap_argv`` is pure (``argv -> argv``), so each backend's command is unit-testable
without running a sandbox; ``seatbelt`` is also exercised end-to-end by the probe on
macOS.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from enum import StrEnum

Argv = list[str]
Wrapper = Callable[[Argv], Argv]


class SandboxRuntime(StrEnum):
    AUTO = "auto"  # probe the host, pick the strongest backend available
    NONE = "none"  # no sandbox: when none is configured or available
    SEATBELT = "seatbelt"  # macOS sandbox-exec
    BWRAP = "bwrap"  # Linux bubblewrap (mount namespace)
    LANDLOCK = "landlock"  # Linux Landlock LSM (declared; helper not yet built)
    APPTAINER = "apptainer"  # HPC Singularity/Apptainer SIF


def detect() -> SandboxRuntime:
    """Pick the strongest sandbox actually available on this host."""
    if sys.platform == "darwin":
        return SandboxRuntime.SEATBELT if shutil.which("sandbox-exec") else SandboxRuntime.NONE
    if sys.platform.startswith("linux"):
        if os.environ.get("SINGULARITY_ALLOWED_DIR") and (
            shutil.which("apptainer") or shutil.which("singularity")
        ):
            return SandboxRuntime.APPTAINER
        if shutil.which("bwrap") and _userns_ok():
            return SandboxRuntime.BWRAP
    return SandboxRuntime.NONE


def _userns_ok() -> bool:
    """True iff unprivileged user namespaces work here (bwrap needs them)."""
    try:
        result = subprocess.run(
            ["unshare", "-Urm", "true"], capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve(runtime: SandboxRuntime) -> SandboxRuntime:
    """Resolve ``auto`` to a concrete backend; pass concrete values through."""
    return detect() if runtime is SandboxRuntime.AUTO else runtime


def make_wrapper(
    runtime: SandboxRuntime,
    *,
    workdir: str,
    allow_read: Sequence[str] = (),
    forbid_read: Sequence[str] = (),
    deny_egress: bool = False,
    image: str | None = None,
) -> Wrapper:
    """Return a pure ``argv -> argv`` wrapper that runs argv inside the sandbox.

    ``allow_read`` are extra read-only paths the agent legitimately needs (the regact
    source, a shared venv). ``forbid_read`` are the paths to hide; allow-by-default
    backends (seatbelt) deny them explicitly, while allowlist backends (bwrap,
    apptainer) simply never bind them. The Python prefix and the workdir are always
    granted.
    """
    resolved = resolve(runtime)

    def wrap(argv: Argv) -> Argv:
        return wrap_argv(
            resolved,
            argv,
            workdir=workdir,
            allow_read=allow_read,
            forbid_read=forbid_read,
            deny_egress=deny_egress,
            image=image,
        )

    return wrap


def wrap_argv(
    runtime: SandboxRuntime,
    argv: Sequence[str],
    *,
    workdir: str,
    allow_read: Sequence[str] = (),
    forbid_read: Sequence[str] = (),
    deny_egress: bool = False,
    image: str | None = None,
) -> Argv:
    """Prepend the per-platform launcher so ``argv`` runs inside the sandbox."""
    resolved = resolve(runtime)
    if resolved is SandboxRuntime.NONE:
        return list(argv)
    if resolved is SandboxRuntime.SEATBELT:
        return _seatbelt(argv, workdir, allow_read, forbid_read, deny_egress)
    if resolved is SandboxRuntime.BWRAP:
        return _bwrap(argv, workdir, allow_read, deny_egress)
    if resolved is SandboxRuntime.APPTAINER:
        return _apptainer(argv, workdir, allow_read, image)
    if resolved is SandboxRuntime.LANDLOCK:
        raise NotImplementedError("landlock backend needs a helper binary; not yet built")
    return list(argv)


def _python_prefixes() -> list[str]:
    """The interpreter dirs the agent always needs to start Python at all."""
    return sorted({os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)})


def _seatbelt(
    argv: Sequence[str],
    workdir: str,
    allow_read: Sequence[str],
    forbid_read: Sequence[str],
    deny_egress: bool,
) -> Argv:
    """macOS: allow by default, deny the forbidden paths, re-allow what the agent needs.

    A later ``allow`` overrides an earlier ``deny`` in SBPL, so the workdir, source,
    and interpreter stay readable even when nested under a forbidden tree. This denies
    named paths rather than denying everything by default, so it is weaker than the
    bwrap/apptainer allowlist; it suits a development machine, where breaking the
    agent's own toolchain (auth, interpreter) must be avoided.
    """
    allow = [os.path.realpath(workdir), *_python_prefixes()]
    allow += [os.path.realpath(p) for p in allow_read]
    forbid = [os.path.realpath(p) for p in forbid_read]
    rules = ["(version 1)", "(allow default)"]
    if forbid:
        rules.append("(deny file-read* " + " ".join(f'(subpath "{p}")' for p in forbid) + ")")
    rules.append("(allow file-read* " + " ".join(f'(subpath "{p}")' for p in allow) + ")")
    if deny_egress:
        # block external egress but keep loopback (the env server + a local LLM)
        rules.append("(deny network-outbound)")
        rules.append('(allow network-outbound (remote ip "localhost:*"))')
    return ["sandbox-exec", "-p", "".join(rules), *argv]


def _bwrap(
    argv: Sequence[str], workdir: str, allow_read: Sequence[str], deny_egress: bool
) -> Argv:
    """Linux: a mount namespace that contains ONLY the allowlist (deny-default).

    The games/repo are simply never bound, so they are absent from the agent's
    filesystem — an allowlist by construction, not a blocklist of techniques.
    """
    wd = os.path.realpath(workdir)
    cmd = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-uts",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for system_dir in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
        if os.path.isdir(system_dir):
            cmd += ["--ro-bind", system_dir, system_dir]
    for path in (*_python_prefixes(), *(os.path.realpath(p) for p in allow_read)):
        if os.path.isdir(path):
            cmd += ["--ro-bind", path, path]
    cmd += ["--bind", wd, wd, "--chdir", wd]
    if deny_egress:
        # NOTE: --unshare-net also severs loopback; only safe when the env server +
        # LLM are reached over a unix socket or are inside the namespace. Fine-grained
        # "deny external, keep loopback" needs a seccomp filter (a later backend).
        cmd += ["--unshare-net"]
    cmd += ["--", *argv]
    return cmd


def _apptainer(
    argv: Sequence[str], workdir: str, allow_read: Sequence[str], image: str | None
) -> Argv:
    """HPC: a Singularity/Apptainer SIF with ONLY the allowlist bound.

    ``--containall --no-home`` are mandatory: without them Apptainer auto-mounts
    ``$HOME``/``$PWD``/``$TMP`` and the game source stays readable.
    """
    if not image:
        raise ValueError("apptainer runtime needs an image (.sif); set runtime_opts['image']")
    wd = os.path.realpath(workdir)
    binary = "apptainer" if shutil.which("apptainer") else "singularity"
    cmd = [binary, "exec", "--containall", "--no-home", "--bind", f"{wd}:{wd}", "--pwd", wd]
    for path in (*_python_prefixes(), *(os.path.realpath(p) for p in allow_read)):
        cmd += ["--bind", f"{path}:{path}"]
    cmd += [image, *argv]
    return cmd
