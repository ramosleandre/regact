"""Submission serialization, including caller cancellation and partial failures."""

import asyncio
import threading
from pathlib import Path

import pytest

from regact.config.schema import Lifecycle
from regact.obs.result import EvalResult
from regact.session.state import ExperimentState
from regact.tools.base import ToolContext
from regact.tools.submit_solution import SubmitSolution


class BlockingExecutor:
    def __init__(self, fail=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.paths = []
        self.fail = fail

    def run(self, **kwargs):
        path = Path(kwargs["output_path"])
        self.paths.append(path)
        if len(self.paths) == 1:
            path.write_text("partial first submission")
            self.started.set()
            if not self.release.wait(5):
                raise TimeoutError("test did not release executor")
            if self.fail:
                raise RuntimeError("evaluation failed")
        return EvalResult(task="g", aggregate={"marker": path.parent.name})


def tool(tmp_path, executor):
    state = ExperimentState(problem_name="test", task_name="g")
    return SubmitSolution(
        state,
        executor,
        solution_path=str(tmp_path / "solution.py"),
        submissions_dir=str(tmp_path / "submissions"),
        task_name="g",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    ), state


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_submissions_serialize_even_after_cancellation_or_failure(tmp_path, cancel, fail):
    executor = BlockingExecutor(fail=fail)
    submit, state = tool(tmp_path, executor)
    context = ToolContext(cwd=str(tmp_path))
    first = asyncio.create_task(submit.call({}, context))
    second = None
    try:
        assert await asyncio.to_thread(executor.started.wait, 3)
        second = asyncio.create_task(submit.call({}, context))
        await asyncio.sleep(0)  # let the second request reach the lock
        if cancel:
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()  # repeated cancellation must also retain the lock
            await asyncio.sleep(0)
        assert len(executor.paths) == 1
        assert state.submission_count == 1
        assert not second.done()
        executor.release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await first
        elif fail:
            with pytest.raises(RuntimeError, match="evaluation failed"):
                await first
        else:
            assert (await first).data["submission"] == 0
        result = await asyncio.wait_for(second, 3)
        assert result.data["submission"] == 1
        assert [p.parent.name for p in executor.paths] == ["0", "1"]
        assert executor.paths[0].read_text() == "partial first submission"
        assert state.submission_count == 2
        assert state.last_submission_results["aggregate"]["marker"] == "1"
    finally:
        executor.release.set()
        await asyncio.gather(*[t for t in (first, second) if t], return_exceptions=True)


async def test_cancelling_queued_submission_does_not_run_it(tmp_path):
    executor = BlockingExecutor()
    submit, state = tool(tmp_path, executor)
    context = ToolContext(cwd=str(tmp_path))
    first = asyncio.create_task(submit.call({}, context))
    try:
        assert await asyncio.to_thread(executor.started.wait, 3)
        queued = asyncio.create_task(submit.call({}, context))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        executor.release.set()
        await first
        assert len(executor.paths) == 1
        assert state.submission_count == 1
    finally:
        executor.release.set()
        await asyncio.gather(first, return_exceptions=True)


async def test_different_tasks_do_not_share_submission_lock(tmp_path):
    blocked = BlockingExecutor()
    other = BlockingExecutor()
    other.release.set()
    a, _ = tool(tmp_path / "a", blocked)
    b, _ = tool(tmp_path / "b", other)
    context = ToolContext(cwd=str(tmp_path))
    first = asyncio.create_task(a.call({}, context))
    try:
        assert await asyncio.to_thread(blocked.started.wait, 3)
        result = await asyncio.wait_for(b.call({}, context), 3)
        assert result.data["submission"] == 0
        assert not first.done()
    finally:
        blocked.release.set()
        await asyncio.gather(first, return_exceptions=True)
