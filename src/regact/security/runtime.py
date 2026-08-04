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

``wrap_argv`` is pure (``argv -> argv``), so each backend's command is unit-testable
without running a sandbox; ``seatbelt`` is also exercised end-to-end by the probe on
macOS.
"""

from __future__ import annotations

import functools
import glob
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from enum import StrEnum

logger = logging.getLogger(__name__)

Argv = list[str]
Wrapper = Callable[[Argv], Argv]


class SandboxRuntime(StrEnum):
    AUTO = "auto"  # probe the host, pick the strongest backend available
    NONE = "none"  # no sandbox: when none is configured or available
    SEATBELT = "seatbelt"  # macOS sandbox-exec
    BWRAP = "bwrap"  # Linux bubblewrap (mount namespace)


def detect() -> SandboxRuntime:
    """Pick the strongest sandbox actually available on this host."""
    if sys.platform == "darwin":
        return SandboxRuntime.SEATBELT if shutil.which("sandbox-exec") else SandboxRuntime.NONE
    if sys.platform.startswith("linux") and shutil.which("bwrap") and userns_ok():
        return SandboxRuntime.BWRAP
    return SandboxRuntime.NONE


def userns_ok() -> bool:
    """True iff unprivileged user namespaces work here (bwrap needs them).

    Public because a present ``bwrap`` is not a usable one: distributions such as
    Ubuntu 23.10+ ship AppArmor's ``kernel.apparmor_restrict_unprivileged_userns=1``,
    which makes ``detect()`` fall back to ``none``. Diagnostics report that reason
    instead of leaving "bwrap installed" and "sandbox: none" side by side.
    """
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


def _log_resolved(requested: SandboxRuntime, resolved: SandboxRuntime) -> None:
    """Record which backend a run actually got — the run is unconfined on ``none``.

    ``auto`` silently degrades to ``none`` on a host with no backend, so a run can be
    unsandboxed without asking for it. Log it (WARNING when unconfined, INFO otherwise)
    so the choice is visible in the run's log rather than assumed from the config.
    """
    if resolved is SandboxRuntime.NONE:
        auto = requested is SandboxRuntime.AUTO
        reason = "no backend detected on this host" if auto else "configured"
        logger.warning("sandbox: NOT confined (requested=%s, %s)", requested.value, reason)
    else:
        logger.info("sandbox: %s (requested=%s)", resolved.value, requested.value)


def make_wrapper(
    runtime: SandboxRuntime,
    *,
    workdir: str,
    allow_read: Sequence[str] = (),
    deny_egress: bool = False,
    deny_read: Sequence[str] = (),
    allow_write_prefixes: Sequence[str] = (),
    allow_rw: Sequence[str] = (),
) -> Wrapper:
    """Return a pure ``argv -> argv`` wrapper that runs argv inside the sandbox.

    Deny-by-default on every backend: only the workdir, the interpreter, and ``allow_read``
    (the regact source + the loaded agent's own host dirs) are reachable; everything else —
    every copy of the game, sibling experiments — is absent. We never enumerate the game's
    locations; we allow what the agent needs and deny the rest. ``deny_read`` carves specific
    subtrees (the game engine/data packages) back out of the allowed interpreter prefix.
    ``allow_rw`` grants full access to extra paths — including connecting to unix sockets
    there, which is how bridged host services stay reachable when egress is denied.
    """
    resolved = resolve(runtime)
    _log_resolved(runtime, resolved)

    def wrap(argv: Argv) -> Argv:
        return wrap_argv(
            resolved,
            argv,
            workdir=workdir,
            allow_read=allow_read,
            deny_egress=deny_egress,
            deny_read=deny_read,
            allow_write_prefixes=allow_write_prefixes,
            allow_rw=allow_rw,
        )

    return wrap


def wrap_argv(
    runtime: SandboxRuntime,
    argv: Sequence[str],
    *,
    workdir: str,
    allow_read: Sequence[str] = (),
    deny_egress: bool = False,
    deny_read: Sequence[str] = (),
    allow_write_prefixes: Sequence[str] = (),
    allow_rw: Sequence[str] = (),
) -> Argv:
    """Prepend the per-platform launcher so ``argv`` runs inside the sandbox (deny-by-default)."""
    resolved = resolve(runtime)
    if resolved is SandboxRuntime.NONE:
        return list(argv)
    if resolved is SandboxRuntime.SEATBELT:
        return _seatbelt(
            argv, workdir, allow_read, deny_egress, deny_read, allow_write_prefixes, allow_rw
        )
    if resolved is SandboxRuntime.BWRAP:
        return _bwrap(argv, workdir, allow_read, deny_egress, deny_read, allow_rw)
    return list(argv)


def symlink_chain_dirs(path: str) -> list[str]:
    """Parent dirs of every hop of ``path``'s symlink chain, unresolved.

    Binding only the final realpath is not enough for execvp: a chain like
    ``venv/bin/python -> <module alias>/bin/python -> <install tree>/bin/python``
    dies inside the namespace when the intermediate alias dir was never bound
    (typical on HPC, where module trees alias padded Spack paths).
    """
    dirs: list[str] = []
    current = path
    for _ in range(40):  # cycle guard
        if not os.path.islink(current):
            break
        target = os.readlink(current)
        current = os.path.normpath(os.path.join(os.path.dirname(current), target))
        dirs.append(os.path.dirname(current))
    return dirs


def interpreter_chain_prefixes(executable: str) -> list[str]:
    """Chain-hop dirs of ``executable``, each ``<prefix>/bin`` lifted to ``<prefix>``.

    The dynamic loader resolves RPATH (``$ORIGIN/../lib``) against the hop path it
    executed through, so binding a hop's ``bin`` alone leaves the sibling ``lib``
    tree missing inside the namespace.
    """
    return sorted(
        {
            os.path.dirname(d) if os.path.basename(d) == "bin" else d
            for d in symlink_chain_dirs(executable)
        }
    )


def _parse_ldd_lib_paths(output: str) -> set[str]:
    """Resolved library paths from ``ldd`` output (``name => /path (addr)`` lines)."""
    paths: set[str] = set()
    for line in output.splitlines():
        if " => " not in line:
            continue
        target = line.split(" => ", 1)[1].strip()
        if not target.startswith("/"):
            continue  # "not found" and vdso-style entries
        paths.add(target.split(" (", 1)[0].strip())
    return paths


@functools.lru_cache(maxsize=1)
def _interpreter_lib_dirs() -> tuple[str, ...]:
    """Dirs of every shared library the interpreter and its stdlib extensions load.

    A relocated interpreter (HPC module/Spack trees) resolves its RPATH to lib dirs
    scattered across many install prefixes; each must exist inside the namespace or
    startup dies on the first missing ``.so``. ``ldd`` yields the transitive closure
    up front instead of one ENOENT round-trip per library. Host dirs already in the
    system allowlist are skipped; empty where ``ldd`` is absent (macOS).
    """
    if shutil.which("ldd") is None:
        return ()
    targets = [os.path.realpath(sys.executable)]
    targets += glob.glob(
        os.path.join(os.path.realpath(sys.base_prefix), "lib", "python*", "lib-dynload", "*.so")
    )
    try:
        proc = subprocess.run(["ldd", *targets], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ()
    dirs: set[str] = set()
    for path in _parse_ldd_lib_paths(proc.stdout):
        for candidate in (os.path.normpath(path), os.path.realpath(path)):
            parent = os.path.dirname(candidate)
            if not parent.startswith(("/usr/", "/lib", "/bin", "/sbin", "/etc")):
                dirs.add(parent)
    return tuple(sorted(dirs))


def _python_prefixes() -> list[str]:
    """The interpreter dirs the agent always needs to start Python at all."""
    return sorted(
        {
            os.path.realpath(sys.prefix),
            os.path.realpath(sys.base_prefix),
            *interpreter_chain_prefixes(sys.executable),
            *_interpreter_lib_dirs(),
        }
    )


def _sbpl_target(path: str) -> str:
    """An SBPL path target: ``subpath`` for a directory, ``literal`` for a file."""
    return f'(subpath "{path}")' if os.path.isdir(path) else f'(literal "{path}")'


def _seatbelt(
    argv: Sequence[str],
    workdir: str,
    allow_read: Sequence[str],
    deny_egress: bool,
    deny_read: Sequence[str] = (),
    allow_write_prefixes: Sequence[str] = (),
    allow_rw: Sequence[str] = (),
) -> Argv:
    """macOS: deny-by-default; allow only the system layer, interpreter, and ``allow_read``.

    File *metadata* (stat) is allowed anywhere (harmless — it reveals only that a path
    exists, not its contents — and avoids chasing non-fatal stat denials); file-read *data*
    is allowed only on the system layer + interpreter + ``allow_read``. The workdir, /dev and
    the user cache dir are read-write. Everything else — the game (wherever its many copies
    live), sibling experiments, the shared temp dir — is absent. The agent's scratch is kept
    in its workdir via ``TMPDIR`` (set by the orchestrator).

    NOTE: codex (Node) and CPython run fine here, and so does ``claude.exe`` (a Bun binary).
    Bun additionally reads the ICU timezone DB and a POSIX shm region at startup and *SIGTRAPs*
    (not a graceful error) if either is denied, so the profile allows both below. Subscription
    auth keeps its token in the macOS Keychain, so claude's ``host_read_paths`` add
    ``~/Library/Keychains`` (API-key mode via ``ANTHROPIC_API_KEY`` needs neither). Verified on
    macOS: claude runs a real authed turn fully confined, and a read outside the allowlist stays
    denied (R2).
    """
    home = os.path.expanduser("~")
    system_ro = (
        "/usr",
        "/bin",
        "/sbin",
        "/System",
        "/Library",
        "/private/var/db/dyld",
        "/private/etc",
        "/opt",
        "/private/var/db/timezone",
    )
    read_only = [*_python_prefixes(), *(d for d in system_ro if os.path.exists(d))]
    read_write = [os.path.realpath(workdir), "/dev", os.path.join(home, "Library/Caches")]
    read_write += [os.path.realpath(p) for p in allow_read]
    read_write += [os.path.realpath(p) for p in allow_rw]
    rules = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow file-read-metadata)",
        "(allow ipc-posix-shm*)",
        '(allow file-read* (literal "/") ' + " ".join(_sbpl_target(p) for p in read_only) + ")",
        "(allow file* " + " ".join(_sbpl_target(p) for p in read_write) + ")",
    ]
    for prefix in allow_write_prefixes:
        # Files whose leaf name is random per call (e.g. Claude Code's /tmp/claude-<rand>-cwd);
        # a subpath rule can't name them, so allow the prefix. Prefixes must be plain paths.
        rules.append(f'(allow file* (regex #"^{prefix}"))')
    if deny_read:  # carve the game packages back out of the allowed venv (last match wins)
        targets = " ".join(_sbpl_target(os.path.realpath(p)) for p in deny_read)
        rules.append(f"(deny file-read* {targets})")
    if deny_egress:  # keep loopback (env server + local LLM), block external
        rules.append('(allow network* (local ip "localhost:*") (remote ip "localhost:*"))')
        # The loopback-ip filter above excludes unix sockets, so sockets in an
        # ``allow_rw`` path need their own connect rule to stay reachable.
        for path in (os.path.realpath(p) for p in allow_rw):
            rules.append(f"(allow network-outbound (remote unix-socket {_sbpl_target(path)}))")
    else:
        rules.append("(allow network*)")
    return ["sandbox-exec", "-p", "".join(rules), *argv]


def _bwrap(
    argv: Sequence[str],
    workdir: str,
    allow_read: Sequence[str],
    deny_egress: bool,
    deny_read: Sequence[str] = (),
    allow_rw: Sequence[str] = (),
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
    for path in _python_prefixes():
        if os.path.isdir(path):
            cmd += ["--ro-bind", path, path]
    for path in (os.path.realpath(p) for p in allow_read):
        cmd += ["--ro-bind-try", path, path]
    cmd += ["--bind", wd, wd, "--chdir", wd]
    for path in (os.path.realpath(p) for p in allow_rw):
        cmd += ["--bind", path, path]
    for path in (os.path.realpath(p) for p in deny_read):
        if os.path.isdir(path):
            cmd += ["--tmpfs", path]
    if deny_egress:
        # A fresh network namespace: only a private loopback exists, so every host
        # route (including the host's own 127.0.0.1) is gone by construction.
        # Sanctioned services come back through socket files in ``allow_rw``
        # (see :mod:`regact.security.netbridge`).
        cmd += ["--unshare-net"]
    cmd += ["--", *argv]
    return cmd
