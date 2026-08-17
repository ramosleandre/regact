"""The OS sandbox: the contract, the conformance probe, and the wrap_argv backends.

Pure-function tests run everywhere (they assert the launcher argv without running a
sandbox). One end-to-end test runs the probe under macOS ``sandbox-exec`` to show a
read of a forbidden path flips from succeeding to denied; it is skipped off macOS.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from regact.security.contract import CONTRACT, Invariant
from regact.security.probe import run_probe
from regact.security.runtime import (
    SandboxRuntime,
    _parse_ldd_lib_paths,
    detect,
    interpreter_chain_prefixes,
    make_wrapper,
    symlink_chain_dirs,
    wrap_argv,
)

_LDD_SAMPLE = """\
\tlinux-vdso.so.1 (0x00007fff0a1f2000)
\tlibpython3.12.so.1.0 => /opt/software/py/lib/libpython3.12.so.1.0 (0x00007f1a2c000000)
\tlibintl.so.8 => /opt/software/gettext/lib/libintl.so.8 (0x00007f1a2bfb0000)
\tlibmissing.so => not found
\tlibc.so.6 => /lib64/libc.so.6 (0x00007f1a2bc00000)
\t/lib64/ld-linux-x86-64.so.2 (0x00007f1a2c2ae000)
"""


def test_parse_ldd_lib_paths() -> None:
    paths = _parse_ldd_lib_paths(_LDD_SAMPLE)
    assert "/opt/software/py/lib/libpython3.12.so.1.0" in paths
    assert "/opt/software/gettext/lib/libintl.so.8" in paths
    assert "/lib64/libc.so.6" in paths
    assert not any("not found" in p or "vdso" in p for p in paths)


def test_symlink_chain_dirs_covers_intermediate_hops(tmp_path: Path) -> None:
    # HPC pattern: venv/bin/python -> <module alias>/bin/python -> <store>/bin/python.
    # Binding only the realpath leaves the alias hop missing inside the namespace.
    store = tmp_path / "store" / "bin"
    alias = tmp_path / "alias" / "bin"
    venv = tmp_path / "venv" / "bin"
    for d in (store, alias, venv):
        d.mkdir(parents=True)
    (store / "python").write_text("")
    (alias / "python").symlink_to(store / "python")
    (venv / "python").symlink_to(alias / "python")
    dirs = symlink_chain_dirs(str(venv / "python"))
    assert str(alias) in dirs and str(store) in dirs


def test_interpreter_chain_prefixes_lift_bin_to_prefix(tmp_path: Path) -> None:
    # The loader resolves $ORIGIN/../lib against the hop path, so <prefix>/bin
    # hops must be bound as the whole <prefix> (bin + lib), not bin alone.
    store = tmp_path / "store" / "bin"
    alias = tmp_path / "alias" / "bin"
    venv = tmp_path / "venv" / "bin"
    for d in (store, alias, venv):
        d.mkdir(parents=True)
    (store / "python").write_text("")
    (alias / "python").symlink_to(store / "python")
    (venv / "python").symlink_to(alias / "python")
    prefixes = interpreter_chain_prefixes(str(venv / "python"))
    assert str(tmp_path / "alias") in prefixes and str(tmp_path / "store") in prefixes
    assert str(alias) not in prefixes  # lifted, not the bare bin dir


def test_symlink_chain_dirs_plain_file_is_empty(tmp_path: Path) -> None:
    plain = tmp_path / "python"
    plain.write_text("")
    assert symlink_chain_dirs(str(plain)) == []


def test_contract_covers_the_six_invariants() -> None:
    assert {spec.invariant for spec in CONTRACT} == set(Invariant)


def test_probe_detects_the_hole_when_bare(tmp_path: Path) -> None:
    """With no room, the probe must report the read-the-secret invariant breached."""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    secret = tmp_path / "games" / "ar25.py"
    secret.parent.mkdir()
    secret.write_text("WINNING = [3, 1, 2, 0]\n")

    results = run_probe(workdir=str(workdir), secret_path=str(secret), check_egress=False)
    by_name = {r.name: r for r in results}
    assert by_name["read own workdir"].defended  # R1: the agent can still work
    assert not by_name["A1 open(game secret)"].defended  # R2: bare => the secret is readable


def test_wrap_argv_none_is_passthrough() -> None:
    argv = ["claude", "-p", "hi"]
    assert wrap_argv(SandboxRuntime.NONE, argv, workdir="/w") == argv


def test_make_wrapper_none_returns_pure_passthrough() -> None:
    wrap = make_wrapper(SandboxRuntime.NONE, workdir="/w")
    assert wrap(["echo", "hi"]) == ["echo", "hi"]


def test_wrap_argv_seatbelt_is_deny_default() -> None:
    out = wrap_argv(
        SandboxRuntime.SEATBELT, ["python3", "x.py"], workdir="/tmp/wd", allow_read=["/repo/src"]
    )
    assert out[0] == "sandbox-exec" and out[-2:] == ["python3", "x.py"]
    profile = out[2]
    assert "(deny default)" in profile  # deny-by-default
    assert "(allow file*" in profile  # the workdir + allow_read are read-write
    assert "/repo/environnement" not in profile  # a non-allowed path (the game) never appears


def test_wrap_argv_bwrap_binds_workdir_but_not_the_repo() -> None:
    workdir = os.path.realpath("/tmp/wd")  # _bwrap canonicalizes (/tmp -> /private/tmp on macOS)
    out = wrap_argv(SandboxRuntime.BWRAP, ["claude"], workdir=workdir, allow_read=["/repo/src"])
    assert out[0] == "bwrap" and out[-1] == "claude"
    joined = " ".join(out)
    assert f"--bind {workdir} {workdir}" in joined  # the workdir is bound (writable)
    assert "--ro-bind /repo /repo" not in joined  # the repo root is NOT bound => absent


def test_wrap_argv_allows_a_file_path(tmp_path: Path) -> None:
    """A FILE in allow_read (e.g. ~/.claude.json) is handled, not silently dropped."""
    config = tmp_path / "config.json"
    config.write_text("{}")
    real = os.path.realpath(str(config))
    sb = wrap_argv(SandboxRuntime.SEATBELT, ["x"], workdir=str(tmp_path), allow_read=[str(config)])
    assert f'(literal "{real}")' in sb[2]  # a file -> literal, not subpath
    bwrap = wrap_argv(SandboxRuntime.BWRAP, ["x"], workdir=str(tmp_path), allow_read=[str(config)])
    assert f"--ro-bind-try {real} {real}" in " ".join(bwrap)  # binds files + tolerates absent


def test_deny_read_carves_game_packages_out_of_the_allowed_venv() -> None:
    """deny_read hides packages that live INSIDE the allowed interpreter prefix (the venv)."""
    sb = wrap_argv(
        SandboxRuntime.SEATBELT,
        ["x"],
        workdir="/tmp/wd",
        allow_read=["/repo/src"],
        deny_read=["/repo/.venv/lib/python3.12/site-packages/arcengine"],
    )
    assert "(deny file-read*" in sb[2]  # a deny rule overrides the venv allow (last match wins)
    # bwrap mounts an empty tmpfs over each existing deny path (/usr exists here).
    bw = wrap_argv(SandboxRuntime.BWRAP, ["x"], workdir="/tmp/wd", deny_read=["/usr"])
    assert "--tmpfs /usr" in " ".join(bw)


@pytest.mark.skipif(
    detect() is not SandboxRuntime.BWRAP, reason="minigrid deny-read e2e needs a real bwrap host"
)
def test_minigrid_engine_is_unreadable_under_the_sandbox(tmp_path: Path) -> None:
    """End-to-end: with ``minigrid`` in deny_read (as MiniGridProblem.secret_modules() sets), an
    agent under the sandbox cannot read its encodings, env-class docs, or gymnasium.make() the env
    - closing the reconstruct-the-env-in-process exploit at the OS level, not just via prose. A
    generic dep (gymnasium) still imports, so the venv is not broadly broken.
    """
    pytest.importorskip("minigrid")
    import regact
    from regact.orchestration.task import _secret_module_paths
    from regact.problems.minigrid.problem import MiniGridProblem

    src = os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))
    deny = _secret_module_paths(MiniGridProblem().secret_modules())
    assert deny, "minigrid installed but its path did not resolve"

    def blocked(code: str) -> bool:
        argv = wrap_argv(
            SandboxRuntime.BWRAP,
            [sys.executable, "-c", code],
            workdir=str(tmp_path),
            allow_read=[src],
            deny_read=deny,
        )
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).returncode != 0

    assert blocked("from minigrid.core.constants import OBJECT_TO_IDX")  # encodings hidden
    assert blocked("from minigrid.envs import DoorKeyEnv")  # task rules (docstrings) hidden
    make = "import minigrid, gymnasium; gymnasium.make('MiniGrid-DoorKey-8x8-v0')"
    assert blocked(make)  # cannot reconstruct the env in-process
    assert not blocked("import gymnasium")  # a generic dependency is still importable


@pytest.mark.skipif(
    detect() is not SandboxRuntime.BWRAP, reason="regact.problems deny-read e2e needs bwrap"
)
def test_regact_problems_hidden_while_the_rest_of_regact_imports(tmp_path: Path) -> None:
    """The agent needs regact.envclient/controllers to run, but never regact.problems (the
    game wrappers, which leak the game + obs format). deny_read on that one subpackage hides
    it while the rest of regact - on allow_read - stays importable. (task.py wires this.)"""
    import regact

    src = os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))
    problems = os.path.realpath(os.path.join(src, "regact", "problems"))

    def rc(code: str) -> int:
        argv = wrap_argv(
            SandboxRuntime.BWRAP,
            [sys.executable, "-c", code],
            workdir=str(tmp_path),
            allow_read=[src],
            deny_read=[problems],
        )
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).returncode

    assert rc("from regact.envclient.client import EnvClient") == 0  # agent still runs
    assert rc("from regact.controllers.executor import run_episodes_raw") == 0  # eval still runs
    assert rc("import regact.problems.minigrid.problem") != 0  # the game wrapper is hidden


def test_detect_returns_a_known_runtime() -> None:
    assert detect() in set(SandboxRuntime)


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="seatbelt end-to-end runs on macOS only",
)
def test_seatbelt_blocks_the_secret_end_to_end() -> None:
    """Deny-default: a path outside the allowlist (the game) is denied; the workdir is not."""
    workdir = tempfile.mkdtemp(prefix="agent_wd_")
    game_dir = tempfile.mkdtemp(prefix="game_")  # NOT in the allowlist => denied
    secret = os.path.join(game_dir, "ar25.py")
    Path(secret).write_text("WINNING = [3, 1, 2, 0]\n")
    read = "import sys; open(sys.argv[1]).read()"

    blocked = subprocess.run(
        wrap_argv(SandboxRuntime.SEATBELT, [sys.executable, "-c", read, secret], workdir=workdir),
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0 and "PermissionError" in blocked.stderr

    # control: reading its own workdir still works (R1 not broken).
    own = os.path.join(workdir, "note.txt")
    Path(own).write_text("ok")
    ok = subprocess.run(
        wrap_argv(SandboxRuntime.SEATBELT, [sys.executable, "-c", read, own], workdir=workdir),
        capture_output=True,
    )
    assert ok.returncode == 0


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="seatbelt end-to-end runs on macOS only",
)
def test_task_style_wiring_hides_game_but_keeps_workdir(tmp_path: Path) -> None:
    """Mirror task.py's deny-default wiring: only the workdir + src are allowed, so a game dir
    (a sibling of src, never in the allowlist) is denied while the workdir — nested under
    experiments/ — stays readable.
    """
    (tmp_path / "src").mkdir()
    games = tmp_path / "environnement"
    games.mkdir()
    (games / "g.py").write_text("ANSWER = 1\n")
    wd = tmp_path / "experiments" / "run" / "workdir"
    wd.mkdir(parents=True)
    (wd / "solution.py").write_text("x = 1\n")

    allow = [str(tmp_path / "src")]  # what task.py passes: src + the agent's host dirs
    read = "import sys; open(sys.argv[1]).read()"

    blocked = subprocess.run(
        wrap_argv(
            SandboxRuntime.SEATBELT,
            [sys.executable, "-c", read, str(games / "g.py")],
            workdir=str(wd),
            allow_read=allow,
        ),
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0 and "PermissionError" in blocked.stderr  # game hidden

    ok = subprocess.run(
        wrap_argv(
            SandboxRuntime.SEATBELT,
            [sys.executable, "-c", read, str(wd / "solution.py")],
            workdir=str(wd),
            allow_read=allow,
        ),
        capture_output=True,
    )
    assert ok.returncode == 0  # the workdir (under experiments/) stays readable (R1)


def test_wrap_argv_bwrap_binds_allow_rw_paths(tmp_path: Path) -> None:
    sock = tmp_path / "p8000.sock"
    sock.touch()
    argv = wrap_argv(SandboxRuntime.BWRAP, ["true"], workdir=str(tmp_path), allow_rw=[str(sock)])
    real = os.path.realpath(str(sock))
    assert f"--bind {real} {real}" in " ".join(argv)


def test_wrap_argv_seatbelt_allows_unix_socket_connect_under_deny_egress(tmp_path: Path) -> None:
    sock = tmp_path / "p8000.sock"
    sock.touch()
    profile = wrap_argv(
        SandboxRuntime.SEATBELT,
        ["true"],
        workdir=str(tmp_path),
        deny_egress=True,
        allow_rw=[str(sock)],
    )[2]
    assert "remote unix-socket" in profile
    assert os.path.realpath(str(sock)) in profile


def test_home_enumeration_tolerates_only_the_mount_skeleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dir names on the path to an allowed prefix are metadata; anything else is a breach."""
    from regact.security.probe import _can_enumerate_home

    home = tmp_path / "home"
    venv = home / "work" / "repo" / ".venv"
    venv.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(venv))

    attacked, detail = _can_enumerate_home(str(home))
    assert not attacked and "skeleton only" in detail  # only the path to the venv shows

    (home / "private.txt").write_text("x")
    attacked, detail = _can_enumerate_home(str(home))
    assert attacked and "beyond" in detail  # a real entry outside the skeleton IS a breach


def test_probe_report_separates_security_and_liveness_verdicts() -> None:
    """A conforming deny is DEFENDED, a conforming allow WORKS; failures split into
    VULNERABLE (breach) vs BLOCKED (over-restrictive sandbox)."""
    from regact.security.probe import CheckResult, format_report

    report = format_report(
        [
            CheckResult("R2", "attack held off", "deny", True, "x"),
            CheckResult("R4", "sanctioned path up", "allow", True, "x"),
            CheckResult("R5", "attack got through", "deny", False, "x"),
            CheckResult("R4", "sanctioned path down", "allow", False, "x"),
        ]
    )
    assert "DEFENDED" in report and "WORKS" in report
    assert "VULNERABLE" in report and "BLOCKED" in report
    assert "1 BREACH(ES), 1 BLOCKED PATH(S)" in report


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="seatbelt end-to-end runs on macOS only",
)
def test_seatbelt_bridged_socket_stays_connectable_under_deny_egress(tmp_path: Path) -> None:
    """R4/L2 end to end: with egress denied, a socket file in ``allow_rw`` must still
    accept a connection from inside the sandbox."""
    from regact.security.probe import bridged_socket_listener

    connect = "import socket, sys; socket.socket(socket.AF_UNIX).connect(sys.argv[1])"
    with bridged_socket_listener() as sock_path:
        result = subprocess.run(
            wrap_argv(
                SandboxRuntime.SEATBELT,
                [sys.executable, "-c", connect, sock_path],
                workdir=str(tmp_path),
                deny_egress=True,
                allow_rw=[sock_path],
            ),
            cwd=str(tmp_path),  # like a real run: the child starts in its workdir
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stderr
