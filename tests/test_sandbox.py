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
    detect,
    make_wrapper,
    wrap_argv,
)


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


def test_wrap_argv_apptainer_is_containall_and_needs_an_image() -> None:
    out = wrap_argv(
        SandboxRuntime.APPTAINER,
        ["python3", "x.py"],
        workdir="/tmp/wd",
        image="/sif/agent.sif",
    )
    assert "--containall" in out and "--no-home" in out
    assert out[-3:] == ["/sif/agent.sif", "python3", "x.py"]

    with pytest.raises(ValueError):
        wrap_argv(SandboxRuntime.APPTAINER, ["x"], workdir="/tmp/wd", image=None)


def test_wrap_argv_allows_a_file_path(tmp_path: Path) -> None:
    """A FILE in allow_read (e.g. ~/.claude.json) is handled, not silently dropped."""
    config = tmp_path / "config.json"
    config.write_text("{}")
    real = os.path.realpath(str(config))
    sb = wrap_argv(SandboxRuntime.SEATBELT, ["x"], workdir=str(tmp_path), allow_read=[str(config)])
    assert f'(literal "{real}")' in sb[2]  # a file -> literal, not subpath
    bwrap = wrap_argv(SandboxRuntime.BWRAP, ["x"], workdir=str(tmp_path), allow_read=[str(config)])
    assert f"--ro-bind-try {real} {real}" in " ".join(bwrap)  # binds files + tolerates absent


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
        wrap_argv(SandboxRuntime.SEATBELT, [sys.executable, "-c", read, str(games / "g.py")],
                  workdir=str(wd), allow_read=allow),
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0 and "PermissionError" in blocked.stderr  # game hidden

    ok = subprocess.run(
        wrap_argv(SandboxRuntime.SEATBELT, [sys.executable, "-c", read, str(wd / "solution.py")],
                  workdir=str(wd), allow_read=allow),
        capture_output=True,
    )
    assert ok.returncode == 0  # the workdir (under experiments/) stays readable (R1)
