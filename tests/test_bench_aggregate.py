"""The benchmark aggregator's load-bearing piece: controller classification.

``scripts/`` is not importable as a package, so load the module by path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_aggregate.py"
_spec = importlib.util.spec_from_file_location("bench_aggregate", _SCRIPT)
assert _spec and _spec.loader
bench_aggregate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_aggregate)


def _ctrl(body: str) -> str:
    return f"class C:\n    def act(self, obs):\n        {body}\n"


_STUB = "class C:\n    def act(self, obs):\n        raise NotImplementedError\n"
_TRIVIAL_FIRST = _ctrl("return obs.available_actions[0]")
_TRIVIAL_CONST = _ctrl("return 1")
_TRIVIAL_RANDOM = "import random\n" + _ctrl("return random.choice(obs.available_actions)")
_REASONED_FRAME = _ctrl("return 2 if obs.frame[0][0] == 5 else 1")
_REASONED_STATE = _ctrl(
    "self.t = getattr(self, 't', 0) + 1\n        return obs.available_actions[self.t % 2]"
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_STUB, "stub"),
        (_TRIVIAL_FIRST, "trivial"),
        (_TRIVIAL_CONST, "trivial"),
        (_TRIVIAL_RANDOM, "trivial"),
        (_REASONED_FRAME, "reasoned"),
        (_REASONED_STATE, "reasoned"),
    ],
)
def test_classify_controller(tmp_path: Path, source: str, expected: str) -> None:
    path = tmp_path / "solution.py"
    path.write_text(source)
    assert bench_aggregate._classify_controller(path) == expected


def test_classify_controller_missing_and_unparsable(tmp_path: Path) -> None:
    assert bench_aggregate._classify_controller(tmp_path / "nope.py") == "missing"
    bad = tmp_path / "bad.py"
    bad.write_text("class C:\n    def act(self, obs)\n        return 1\n")  # syntax error
    assert bench_aggregate._classify_controller(bad) == "unparsable"
    no_act = tmp_path / "no_act.py"
    no_act.write_text("x = 1\n")
    assert bench_aggregate._classify_controller(no_act) == "unparsable"


def test_classify_controller_follows_factored_code_library(tmp_path: Path) -> None:
    """A thin ``solution.py`` subclass of an agent-written ``code_library`` controller is
    classified by the real ``act`` in that module - not reported ``unparsable``."""
    lib = tmp_path / "code_library"
    lib.mkdir()
    (lib / "__init__.py").write_text("")
    (lib / "smart_controller.py").write_text(
        "class SmartController:\n"
        "    def act(self, obs):\n"
        "        for a in obs.available_actions:\n"  # a loop over the obs = reasoned
        "            return a\n"
    )
    sol = tmp_path / "solution.py"
    sol.write_text(
        "from code_library.smart_controller import SmartController\n"
        "class Controller(SmartController):\n    pass\n"
        "def get_controller():\n    return Controller()\n"
    )
    assert bench_aggregate._classify_controller(sol) == "reasoned"


def test_classify_controller_follows_factory_instantiation(tmp_path: Path) -> None:
    """No subclass at all - solution.py's get_controller returns a controller imported
    from an agent-written module. The import is still followed to the real act."""
    lib = tmp_path / "code_library"
    lib.mkdir()
    (lib / "__init__.py").write_text("")
    (lib / "nav_controller.py").write_text(
        "class NavController:\n"
        "    def act(self, obs):\n"
        "        if obs.frame[0] == 5:\n            return 2\n        return 1\n"
    )
    sol = tmp_path / "solution.py"
    sol.write_text(
        "from code_library.nav_controller import NavController\n"
        "def get_controller():\n    return NavController()\n"
    )
    assert bench_aggregate._classify_controller(sol) == "reasoned"


def test_classify_controller_follows_transitive_base_and_relative_import(tmp_path: Path) -> None:
    """The follow chains through a middle module and handles a relative import."""
    lib = tmp_path / "code_library"
    lib.mkdir()
    (lib / "__init__.py").write_text("")
    (lib / "base_controller.py").write_text(
        "class BaseController:\n    def act(self, obs):\n        return 1\n"  # constant = trivial
    )
    (lib / "smart_controller.py").write_text(
        "from .base_controller import BaseController\n"
        "class SmartController(BaseController):\n    pass\n"
    )
    sol = tmp_path / "solution.py"
    sol.write_text(
        "from code_library.smart_controller import SmartController\n"
        "class Controller(SmartController):\n    pass\n"
    )
    assert bench_aggregate._classify_controller(sol) == "trivial"


def test_classify_controller_thin_subclass_without_modules_is_unparsable(tmp_path: Path) -> None:
    """No local ``act`` and the imported module isn't present -> graceful ``unparsable``."""
    sol = tmp_path / "solution.py"
    sol.write_text(
        "from code_library.smart_controller import SmartController\n"
        "class Controller(SmartController):\n    pass\n"
    )
    assert bench_aggregate._classify_controller(sol) == "unparsable"


@pytest.mark.parametrize(
    ("success_rate", "exit_reason", "expected"),
    [
        (1.0, "agent_exit", "solve"),
        (0.6, "agent_exit", "solve"),
        (0.4, "agent_exit", "genuine-fail"),  # a clean low score is real data
        (0.0, "agent_exit", "genuine-fail"),
        (0.0, "agent_api", "harness-killed"),  # empty_response wall - unreliable
        (None, "agent_api", "harness-killed"),  # killed before a final too
        (0.0, "walltime_limit", "walltime"),
        (None, None, "no-final"),  # still running / killed before teardown
        (None, "loop_crash", "loop_crash"),  # any other terminal state passes through
    ],
)
def test_classify_outcome(success_rate, exit_reason, expected) -> None:
    assert bench_aggregate._classify_outcome(success_rate, exit_reason) == expected


def _mk_run(task_dir: Path, *, model: str, success: float) -> None:
    """A minimal on-disk run dir: config.json + logs/ + a submitted solution + final result."""
    task_dir.mkdir(parents=True)
    (task_dir / "config.json").write_text(
        json.dumps({"agent": {"name": "alan", "model": f"openai/{model}"}, "problem": {"seed": 0}})
    )
    (task_dir / "logs").mkdir()
    wd = task_dir / "workdir"
    wd.mkdir()
    (wd / "solution.py").write_text(_TRIVIAL_CONST)
    final = wd / "submissions" / "final"
    final.mkdir(parents=True)
    (final / "results.json").write_text(json.dumps({"aggregate": {"success_rate": success}}))


def test_collect_runs_flat_and_model_grouped_layouts(tmp_path: Path) -> None:
    """Runs are discovered at any depth: a flat ``exp/stamp/task`` and a
    model-grouped ``model/exp/stamp/task`` tree both yield one row per run."""
    # Flat: root/exp/stamp/task
    _mk_run(tmp_path / "exp_A" / "2026-01-01_00-00-00" / "TaskX", model="Flat-7B", success=1.0)
    # Model-grouped: root/model/exp/stamp/task (one level deeper)
    _mk_run(
        tmp_path / "ModelM" / "exp_B" / "2026-01-01_00-00-00" / "TaskY",
        model="Grouped-70B",
        success=0.0,
    )
    rows = bench_aggregate.collect_runs(tmp_path, all_stamps=False)
    by_model = {r["model"]: r for r in rows}
    assert set(by_model) == {"Flat-7B", "Grouped-70B"}
    assert by_model["Flat-7B"]["task"] == "TaskX"
    assert by_model["Flat-7B"]["success_rate"] == 1.0
    assert by_model["Grouped-70B"]["task"] == "TaskY"


def test_repeats_without_an_attempt_marker_are_all_kept(tmp_path: Path) -> None:
    """Two timestamped runs of one (experiment, task) and NO attempt_N dir are indistinguishable on
    disk: a fan-out launching one job per (task, attempt) writes exactly this, and so does a rerun.
    Keep both. Dropping one is invisible and silently discards attempts; counting a rerun twice
    shows up as a raised n and is warned about on stderr."""
    exp = tmp_path / "exp_A"
    _mk_run(exp / "2026-01-01_00-00-00" / "TaskX", model="M", success=0.0)
    _mk_run(exp / "2026-01-02_00-00-00" / "TaskX", model="M", success=1.0)

    kept = bench_aggregate.collect_runs(tmp_path, all_stamps=False)
    assert len(kept) == 2
    assert {row["success_rate"] for row in kept} == {0.0, 1.0}

    both = bench_aggregate.collect_runs(tmp_path, all_stamps=True)
    assert len(both) == 2


def test_latest_stamp_still_wins_for_a_rerun_of_the_same_attempt(tmp_path: Path) -> None:
    """With an attempt_N marker the repeat IS classifiable - it is the same attempt run twice - so
    the newest stamp wins and the superseded one does not inflate the cell."""
    exp = tmp_path / "exp_A"
    _mk_run(exp / "2026-01-01_00-00-00" / "TaskX" / "attempt_0", model="M", success=0.0)
    _mk_run(exp / "2026-01-02_00-00-00" / "TaskX" / "attempt_0", model="M", success=1.0)

    kept = bench_aggregate.collect_runs(tmp_path, all_stamps=False)
    assert len(kept) == 1
    assert kept[0]["success_rate"] == 1.0


@pytest.mark.parametrize(
    ("exit_reason", "success_rate", "submissions", "expected"),
    [
        ("walltime_limit", 0.4, 12, "capped"),  # iterated, earned its score
        ("walltime_limit", 0.0, 1, "capped"),  # a scored zero is still the model's own
        ("walltime_limit", 0.0, 0, "teardown"),  # scored only by FinalizeControllerHook
        ("walltime_limit", 0.0, None, "teardown"),  # missing count reads as no submission
        ("walltime_limit", None, 0, "starved"),
        ("walltime_limit", None, 5, "starved"),  # submitted, but nothing scoreable came back
        ("agent_exit", 0.4, 3, None),  # not a walltime cut at all
        ("solved", 1.0, 2, None),
    ],
)
def test_walltime_bucket(exit_reason, success_rate, submissions, expected) -> None:
    row = {
        "exit_reason": exit_reason,
        "success_rate": success_rate,
        "submissions": submissions,
    }
    assert bench_aggregate._walltime_bucket(row) == expected


def test_walltime_buckets_partition_every_walltime_run() -> None:
    """No walltime run may fall outside the three columns - a dropped run is invisible."""
    rows = [
        {"exit_reason": "walltime_limit", "success_rate": s, "submissions": n}
        for s in (None, 0.0, 0.5)
        for n in (None, 0, 7)
    ]
    buckets = [bench_aggregate._walltime_bucket(row) for row in rows]
    assert None not in buckets
    assert set(buckets) == {"capped", "teardown", "starved"}
    assert len(buckets) == len(rows)


def _sub(task_dir: Path, index: int, score: float) -> None:
    d = task_dir / "workdir" / "submissions" / f"{index:03d}"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"aggregate": {"success_rate": score}}))


def test_tail_mean_averages_the_most_recent_submissions(tmp_path: Path) -> None:
    """Ordering is numeric, not lexicographic: submission 100 must not sort before 99."""
    task = tmp_path / "TaskX"
    for i, score in enumerate([1.0] * 99 + [0.0]):  # the LAST one is the 100th
        _sub(task, i + 1, score)
    mean, n = bench_aggregate._tail_mean(task, k=2)
    assert n == 2
    assert mean == pytest.approx(0.5)  # submissions 99 (1.0) and 100 (0.0)


def test_tail_mean_missing_submissions(tmp_path: Path) -> None:
    assert bench_aggregate._tail_mean(tmp_path / "nope") == (None, 0)


def test_stability_does_not_flag_a_clean_evaluation_that_beat_its_own_tail() -> None:
    """A model that iterates ends BETTER than its earlier submissions, so a full-length
    error-free evaluation far above its tail is improvement, not luck. The real 480B DoorKey
    cell: 1.00 from ten clean episodes against a 0.28 tail - if 0.28 were true that is 3e-6."""
    improved = {
        "model": "M",
        "task": "DoorKey",
        "success_rate": 1.0,
        "tail_mean": 0.28,
        "n_episodes": 10,
        "n_errors": 0,
        "episodes_asked": 10,
    }
    agreeing = {
        "model": "M",
        "task": "MemoryS17",
        "success_rate": 0.7,
        "tail_mean": 0.69,
        "n_episodes": 10,
        "n_errors": 0,
        "episodes_asked": 10,
    }
    out = bench_aggregate.stability_markdown([improved, agreeing])
    assert "DoorKey" not in out
    assert "MemoryS17" not in out
    assert "full evaluation" in out


def test_zero_episode_evaluation_is_not_a_score() -> None:
    """A final that ran no episodes measured nothing; its 0.0 is a default. Reporting it as a
    score puts a false zero in the table, reading exactly like a model that tried and failed."""
    assert bench_aggregate._primary_score({"n_episodes": 0, "success_rate": 0.0}) is None
    assert bench_aggregate._primary_score({"n_episodes": 10, "success_rate": 0.0}) == 0.0


def test_stability_flags_a_short_evaluation_even_when_it_agrees() -> None:
    """A 1.00 over 3 of 10 episodes is a coincidence, not a solve - flag it on episode count
    alone, since a short run can agree with its own tail and still be weak evidence."""
    short = {
        "model": "M",
        "task": "FourRooms",
        "success_rate": 1.0,
        "tail_mean": 0.95,
        "n_episodes": 3,
        "episodes_asked": 10,
    }
    out = bench_aggregate.stability_markdown([short])
    assert "FourRooms" in out
    assert "3 of 10" in out


def test_episodes_shortfall() -> None:
    assert bench_aggregate._episodes_shortfall({"n_episodes": 3}, 10) == 7
    assert bench_aggregate._episodes_shortfall({"n_episodes": 10}, 10) is None
    assert bench_aggregate._episodes_shortfall({"n_episodes": 12}, 10) is None
    assert bench_aggregate._episodes_shortfall({"n_episodes": None}, 10) is None
    assert bench_aggregate._episodes_shortfall({"n_episodes": 3}, None) is None


def test_stability_shows_the_score_with_errored_episodes_counted() -> None:
    """success_rate is computed over episodes that RAN, so a controller crashing on 7 of 10 and
    succeeding on the 3 it survives reports 1.00. The real 480B FourRooms cell: 1.00 -> 0.30."""
    row = {
        "model": "Qwen3-Coder-480B",
        "task": "FourRooms",
        "success_rate": 1.0,
        "tail_mean": 0.51,
        "n_episodes": 3,
        "n_errors": 7,
        "episodes_asked": 10,
    }
    out = bench_aggregate.stability_markdown([row])
    assert "1.00" in out
    assert "0.30" in out  # 3 successes over 10 attempted
    assert "3 of 10" in out


def test_all_episodes_crashing_is_not_a_zero_score() -> None:
    """A controller that raised on every episode never ran. The table otherwise renders it
    identically to a policy that played and lost, and only the second is about ability."""
    assert (
        bench_aggregate._classify_outcome(None, "agent_exit", controller_crashed=True)
        == "controller-crashed"
    )
    # A genuine 0.0 from episodes that actually ran keeps its meaning.
    assert bench_aggregate._classify_outcome(0.0, "agent_exit") == "genuine-fail"
    # Crashing outranks a solve threshold: a score computed off the survivors is not a solve.
    assert (
        bench_aggregate._classify_outcome(1.0, "solved", controller_crashed=True)
        == "controller-crashed"
    )


def _transcript(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for command, output in pairs:
        lines.append(json.dumps({"type": "ToolCall", "input": {"command": command}}))
        lines.append(json.dumps({"type": "ToolResult", "output": output}))
    path.write_text("\n".join(lines))


def test_uninformative_rate_counts_empty_and_repeated_results(tmp_path: Path) -> None:
    """Both halves matter: the silent heredoc AND informative-looking repetition. A GLM run
    issuing `ls -la` a hundred times for the same listing is not exploring."""
    t = tmp_path / "logs" / "transcript.jsonl"

    _transcript(t, [("ls", "a"), ("cat x", "b"), ("wc y", "c")])
    assert bench_aggregate._uninformative_rate(t) == 0.0  # all novel

    _transcript(t, [("w", "(no output)")] * 4)
    assert bench_aggregate._uninformative_rate(t) == 1.0  # silent writes

    _transcript(t, [("ls -la", "same listing")] * 5)
    assert bench_aggregate._uninformative_rate(t) == pytest.approx(0.8)  # 1st is news, 4 repeats

    # A repeated COMMAND whose output changes is informative - the run learned something.
    _transcript(t, [("ls", "one"), ("ls", "two"), ("ls", "three")])
    assert bench_aggregate._uninformative_rate(t) == 0.0

    assert bench_aggregate._uninformative_rate(tmp_path / "nope.jsonl") is None
