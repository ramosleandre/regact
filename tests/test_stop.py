"""Stopping a running launch from outside: the STOP file and its command."""

import pytest

from regact.orchestration.signals import FileStop, StopSignal, clear_stop
from regact.stop import main


def test_a_stop_file_in_the_run_or_task_directory_stops_that_task(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("regact.orchestration.signals._FILE_CHECK_S", 0.0)
    run = tmp_path / "run"
    (run / "ls20").mkdir(parents=True)
    (run / "vc33").mkdir()
    launch = StopSignal()
    ls20 = FileStop(launch, [str(run), str(run / "ls20")])
    vc33 = FileStop(launch, [str(run), str(run / "vc33")])
    assert not ls20.is_set() and not vc33.is_set()

    assert main([str(run), "--task", "ls20"]) == 0
    assert ls20.is_set() and not ls20.force_requested() and not vc33.is_set()

    assert main([str(run), "--force"]) == 0
    assert vc33.is_set() and vc33.force_requested() and ls20.force_requested()

    clear_stop(str(run))
    clear_stop(str(run / "ls20"))
    clear_stop(str(run / "ls20"))  # already gone
    assert not FileStop(launch, [str(run), str(run / "ls20")]).is_set()
    launch.set()  # Ctrl+C on the launch still reaches every task
    assert FileStop(launch, [str(run)]).is_set()


def test_the_command_refuses_a_directory_that_does_not_exist(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path / "missing")])
