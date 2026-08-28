"""The CWM feature: recording, the verify.py scaffold, params, and end-to-end.

verify.py is exercised as the agent runs it: a plain subprocess on a scaffolded
workdir, with PYTHONPATH stripped so its stdlib-only claim is enforced.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from regact.agent.events import TextDelta, ToolCall, TurnComplete
from regact.agent.scripted_agent import ScriptedAgent
from regact.config.loader import run_config_from_mapping
from regact.config.schema import (
    AgentConfig,
    AgentName,
    ControllerConfig,
    LimitsConfig,
    ProblemConfig,
    RunConfig,
)
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.session import EnvSession
from regact.envclient.obs import Obs
from regact.features.base import FeatureContext, build_features
from regact.features.cwm import CwmFeature, RecordingEnvWrapper, TransitionRecorder
from regact.orchestration.task import run_task
from regact.problems.base import BaseProblem
from regact.testing.fakes import FakeNativeEnv

_IDENTITY_PARSER = "def parse(obs):\n    return obs\n"
_IDENTITY_RENDER = "def render(state):\n    return state\n"
_NO_DEPS: Any = None  # CwmFeature.submission_metrics ignores its deps arg


def _ctx(workdir: Path) -> FeatureContext:
    return FeatureContext(problem_name="fake", task_name="corridor", workdir=str(workdir))


def _cwm_session(output_dir: Path, *, goal: int = 2) -> EnvSession:
    """A session recording into ``output_dir``: canonical in ``output_dir/cwm/``,
    the agent mirror in ``output_dir/workdir/data/``."""
    ctx = FeatureContext(
        problem_name="fake",
        task_name="corridor",
        workdir=str(output_dir / "workdir"),
        output_dir=str(output_dir),
    )
    wrapper = CwmFeature().env_wrapper(ctx)
    assert wrapper is not None
    return EnvSession(
        make_native=lambda: FakeNativeEnv(goal=goal),
        key="corridor",
        renderer=RawRenderer(),
        lifecycle=MultiInstancePolicy(),
        wrappers=[wrapper],
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _lines(output_dir: Path) -> list[dict[str, Any]]:
    """The agent-facing mirror."""
    return _read_jsonl(output_dir / "workdir" / "data" / "transitions.jsonl")


def _canonical_lines(output_dir: Path) -> list[dict[str, Any]]:
    """The trusted canonical log (outside the agent workdir)."""
    return _read_jsonl(output_dir / "cwm" / "transitions.jsonl")


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def test_steps_recorded_verbatim_resets_not(tmp_path: Path) -> None:
    session = _cwm_session(tmp_path)
    env = session.make()
    env.reset()
    assert _lines(tmp_path) == []  # a reset is not a transition
    env.step(1)
    env.step(1)  # reaches the goal
    first, second = _lines(tmp_path)
    assert first["o"]["frame"] == {"pos": 0, "grid": [1, 0, 0]}
    assert first["a"] == 1 and first["r"] == 0.0 and first["done"] is False
    assert first["o2"]["frame"] == {"pos": 1, "grid": [0, 1, 0]}
    assert second["done"] is True and second["r"] == 1.0
    # The canonical (trusted, outside the workdir) mirrors the agent copy.
    assert _canonical_lines(tmp_path) == _lines(tmp_path)


def test_two_episodes_have_no_phantom_transition(tmp_path: Path) -> None:
    """A flat file across an episode boundary must not fabricate a transition:
    resets are not recorded, and ``done`` delimits episodes (Leandre's review)."""
    session = _cwm_session(tmp_path, goal=3)
    for _ in range(2):  # two episodes of three steps each
        env = session.make()
        env.reset()
        for _ in range(3):
            env.step(1)
    lines = _lines(tmp_path)
    assert len(lines) == 6  # 2 x 3 steps, NOT 7 (no last-of-N / first-of-N+1 pair)
    assert [line["done"] for line in lines] == [False, False, True, False, False, True]
    assert [(line["o"]["frame"]["pos"], line["o2"]["frame"]["pos"]) for line in lines] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_recording_accumulates_across_instance_swaps(tmp_path: Path) -> None:
    session = _cwm_session(tmp_path)
    env = session.make()
    env.reset()
    env.step(1)
    env = session.make()  # multi-instance: a fresh (re-wrapped) env
    env.reset()
    env.step(1)
    assert len(_lines(tmp_path)) == 2


def test_mirror_fault_never_breaks_step(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A fault on the agent mirror is best-effort (logged, never raised); with no
    canonical set, the step still succeeds."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    recorder = TransitionRecorder(str(blocker / "data" / "transitions.jsonl"))  # mirror only
    session = EnvSession(
        make_native=lambda: FakeNativeEnv(goal=2),
        key="corridor",
        renderer=RawRenderer(),
        lifecycle=MultiInstancePolicy(),
        wrappers=[lambda env: RecordingEnvWrapper(env, recorder)],
    )
    env = session.make()
    env.reset()
    with caplog.at_level(logging.WARNING):
        obs = env.step(1)
    assert obs.frame["pos"] == 1  # the step succeeded
    assert any("mirror line not written" in record.message for record in caplog.records)


def test_mirror_refuses_symlinked_paths(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A symlink swap must not turn the trusted writer into an out-of-workdir write."""
    outside = tmp_path / "outside"
    outside.mkdir()
    obs = Obs(frame=[0])

    workdir_a = tmp_path / "wd_a"  # data/ itself is a symlink out of the workdir
    (workdir_a / "data").parent.mkdir(parents=True)
    (workdir_a / "data").symlink_to(outside)
    recorder = TransitionRecorder(
        str(workdir_a / "data" / "transitions.jsonl"), mirror_root=str(workdir_a)
    )
    with caplog.at_level(logging.WARNING):
        recorder.record(obs, 1, obs)

    workdir_b = tmp_path / "wd_b"  # the file itself is the symlink
    (workdir_b / "data").mkdir(parents=True)
    (workdir_b / "data" / "transitions.jsonl").symlink_to(outside / "target.jsonl")
    recorder = TransitionRecorder(
        str(workdir_b / "data" / "transitions.jsonl"), mirror_root=str(workdir_b)
    )
    with caplog.at_level(logging.WARNING):
        recorder.record(obs, 1, obs)

    assert list(outside.iterdir()) == []  # nothing was ever written outside
    assert sum("mirror line skipped" in record.message for record in caplog.records) == 2


# --------------------------------------------------------------------------- #
# Params + templates
# --------------------------------------------------------------------------- #
def test_feature_knobs_bake_into_verify(tmp_path: Path) -> None:
    features = build_features(
        {
            "cwm": {
                "max_tested_transitions_per_verify": 123,
                "max_printed_incoherence_transitions_per_verify": 4,
            }
        }
    )
    (feature,) = features
    templates = {t.relpath: t.content for t in feature.templates(_ctx(tmp_path))}
    assert set(templates) == {
        "world_model/model_state.py",
        "world_model/model_parser.py",
        "world_model/model_render.py",
        "world_model/model_transition.py",
        "world_model/model_notes.py",
        "world_model/verify.py",
    }
    verify = templates["world_model/verify.py"]
    assert "MAX_UNIQUE_USED_DEFAULT = 123" in verify
    assert "MAX_INCOHERENCES_DEFAULT = 4" in verify
    assert "__MAX_UNIQUE_USED__" not in verify


def test_unknown_feature_param_fails_loudly() -> None:
    with pytest.raises(TypeError):
        build_features({"cwm": {"nope": 1}})


def test_loader_carries_feature_knobs() -> None:
    config = run_config_from_mapping(
        {
            "agent": {"name": "scripted"},
            "problem": {"name": "fake"},
            "features": {"cwm": {"max_tested_transitions_per_verify": 5}},
        }
    )
    assert config.features == {"cwm": {"max_tested_transitions_per_verify": 5}}


# --------------------------------------------------------------------------- #
# Data-integrity metric (computed from the trusted canonical, model-independent)
# --------------------------------------------------------------------------- #
def test_data_integrity_counts_conflicts(tmp_path: Path) -> None:
    from regact.features.cwm import _data_integrity

    path = tmp_path / "transitions.jsonl"
    a = {"o": _obs(0), "a": 1, "r": 0.0, "o2": _obs(1), "done": False}
    conflict = {"o": _obs(0), "a": 1, "r": 0.0, "o2": _obs(2), "done": False}  # same (o,a), diff o2
    path.write_text("\n".join(json.dumps(t) for t in (a, dict(a), conflict)) + "\n")
    assert _data_integrity(str(path)) == {
        "n_transitions": 3,  # the identical dup still counts on disk
        "n_distinct_transitions": 2,  # but collapses to 2 distinct
        "n_conflicting_transitions": 1,
    }


def test_submission_metrics_reports_data_integrity(tmp_path: Path) -> None:
    ctx = FeatureContext(
        problem_name="fake",
        task_name="corridor",
        workdir=str(tmp_path / "workdir"),
        output_dir=str(tmp_path),
    )
    feature = CwmFeature()
    assert feature.submission_metrics(_NO_DEPS) == {}  # no canonical written yet
    feature.env_wrapper(ctx)  # sets the canonical path on the feature
    canonical = tmp_path / "cwm" / "transitions.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps({"o": _obs(0), "a": 1, "r": 0.0, "o2": _obs(1), "done": False}) + "\n"
    )
    assert feature.submission_metrics(_NO_DEPS)["n_transitions"] == 1


# --------------------------------------------------------------------------- #
# verify.py (run exactly as the agent runs it: a subprocess, stdlib only)
# --------------------------------------------------------------------------- #
def _obs(pos: int, *, done: bool = False) -> dict[str, Any]:
    return {
        "frame": {"pos": pos, "cells": [0] * pos},
        "reward": 1.0 if done else 0.0,
        "is_done": done,
        "available_actions": [1],
        "info": {},
    }


def _scaffold(
    workdir: Path,
    *,
    parser_src: str = _IDENTITY_PARSER,
    render_src: str = _IDENTITY_RENDER,
    step_src: str | None = None,
    lines: list[dict[str, Any]] | None = None,
    n_transitions: int = 3,
) -> Path:
    for template in CwmFeature().templates(_ctx(workdir)):
        path = workdir / template.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.content)
    world = workdir / "world_model"
    (world / "model_parser.py").write_text(parser_src)
    (world / "model_render.py").write_text(render_src)
    if step_src is not None:  # else keep the NotImplementedError stub
        (world / "model_transition.py").write_text(step_src)
    if lines is None:
        lines = [
            {
                "o": _obs(i),
                "a": 1,
                "r": 0.0,
                "o2": _obs(i + 1, done=(i == n_transitions - 1)),
                "done": i == n_transitions - 1,
            }
            for i in range(n_transitions)
        ]
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "transitions.jsonl").write_text("\n".join(json.dumps(t) for t in lines) + "\n")
    return world / "verify.py"


def _run_verify(script: Path, *args: str) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # stdlib-only proof
    proc = subprocess.run(
        [sys.executable, str(script), "--json", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def _incoherences(report: dict[str, Any], check: str) -> list[dict[str, Any]]:
    return [f for f in report["incoherences"] if f.get("check") == check]


def test_verify_identity_model_is_fully_coherent(tmp_path: Path) -> None:
    report = _run_verify(_scaffold(tmp_path))
    assert report["coverage"]["n_unique_obs"] == 4  # 3 distinct o's + the terminal o2
    assert report["parser_injectivity"] == 1.0
    assert report["representation_coherence"] == 1.0
    assert report["transition_accuracy"] is None  # step is still the stub
    assert report["coverage"]["n_transitions_total"] == 3
    assert report["incoherences"] == []
    assert report["valid"] is False  # no rule of motion yet
    assert report["next_priority"] == "implementing the rule of motion"


def test_verify_reports_mismatch_with_pointed_diff(tmp_path: Path) -> None:
    render = (
        "def render(state):\n    s = dict(state)\n"
        "    s['frame'] = dict(s['frame'], pos=99)\n    return s\n"
    )
    report = _run_verify(_scaffold(tmp_path, render_src=render))
    assert report["representation_coherence"] == 0.0
    assert report["next_priority"] == "fixing representation coherence"
    failure = _incoherences(report, "representation")[0]
    assert failure["kind"] == "mismatch" and failure["part"] == "frame"
    assert failure["detail"].startswith("obs.frame.pos: 99 != ")


def test_verify_reports_shape_mismatch(tmp_path: Path) -> None:
    render = (
        "def render(state):\n    s = dict(state)\n"
        "    s['frame'] = dict(s['frame'], cells=[])\n    return s\n"
    )
    report = _run_verify(_scaffold(tmp_path, render_src=render))
    failures = [f for f in report["incoherences"] if "length" in f.get("detail", "")]
    assert failures and failures[0]["kind"] == "mismatch"


def test_verify_a_raising_parse_fails_injectivity(tmp_path: Path) -> None:
    raising = "def parse(obs):\n    raise KeyError('x')\n"
    report = _run_verify(_scaffold(tmp_path, parser_src=raising))
    assert report["parser_injectivity"] == 0.0  # no obs produced a state
    assert report["next_priority"] == "fixing parser injectivity"
    failure = _incoherences(report, "injectivity")[0]
    assert failure["kind"] == "error" and failure["part"] == "parse"
    assert "KeyError" in failure["detail"]


def test_verify_stub_parser_reports_not_implemented(tmp_path: Path) -> None:
    raising = "def parse(obs):\n    raise NotImplementedError\n"
    report = _run_verify(_scaffold(tmp_path, parser_src=raising))
    assert report["parser_injectivity"] is None
    assert report["next_priority"] == "implementing the parser"


def test_verify_cap_limits_unique_used(tmp_path: Path) -> None:
    report = _run_verify(_scaffold(tmp_path), "--max-used", "2")
    assert report["coverage"]["n_unique_obs"] == 2
    assert report["coverage"]["n_obs_total"] == 4  # 3 o's + 1 terminal o2


def test_verify_complexity_is_per_module_and_ignores_notes(tmp_path: Path) -> None:
    script = _scaffold(tmp_path)
    comp = _run_verify(script)["complexity"]
    assert {"parser", "render", "transition", "state", "total"} <= set(comp)
    before = comp["total"]
    notes = tmp_path / "world_model" / "model_notes.py"
    notes.write_text(notes.read_text() + "\n" + "JUNK = [0]\n" * 30)
    assert _run_verify(script)["complexity"]["total"] == before  # notes excluded


def test_verify_state_size_vs_obs(tmp_path: Path) -> None:
    report = _run_verify(_scaffold(tmp_path))  # identity: state == obs
    size = report["state_size"]
    assert size["avg"] > 0 and size["avg_obs"] > 0 and size["ratio"] is not None


def test_verify_injectivity_flags_a_collapsing_parser(tmp_path: Path) -> None:
    collapse = "def parse(obs):\n    return {'x': 1}\n"  # every obs -> one state
    report = _run_verify(_scaffold(tmp_path, parser_src=collapse))
    assert report["parser_injectivity"] < 1.0
    assert report["next_priority"] == "fixing parser injectivity"
    assert report["injectivity_collisions"] == {"n_observations": 4, "n_states": 1}
    collision = _incoherences(report, "injectivity")[0]
    assert len(collision["transitions"]) > 1


def test_verify_shows_all_numbers_even_when_gated(tmp_path: Path) -> None:
    # A non-injective parser still gets a representation number (not hidden behind "fix first").
    collapse = "def parse(obs):\n    return {'x': 1}\n"
    report = _run_verify(_scaffold(tmp_path, parser_src=collapse))
    assert report["parser_injectivity"] < 1.0
    assert isinstance(report["representation_coherence"], float)  # computed + shown, not None


_GOAL3_STEP = (
    "def step(state, action):\n"
    "    pos = state['frame']['pos'] + 1\n"
    "    done = pos == 3\n"
    "    return {'frame': {'pos': pos, 'cells': [0] * pos},\n"
    "            'reward': 1.0 if done else 0.0, 'is_done': done,\n"
    "            'available_actions': [1], 'info': {}}\n"
)


def test_verify_valid_when_representation_and_transition_perfect(tmp_path: Path) -> None:
    report = _run_verify(_scaffold(tmp_path, step_src=_GOAL3_STEP))
    assert report["transition_accuracy"] == 1.0
    assert report["valid"] is True and report["next_priority"] is None
    assert report["coverage"]["n_unique_transitions"] == 3


def test_verify_transition_mismatch_is_pointed(tmp_path: Path) -> None:
    stay = "def step(state, action):\n    return dict(state)\n"  # never advances
    report = _run_verify(_scaffold(tmp_path, step_src=stay))
    assert report["transition_accuracy"] == 0.0
    assert report["next_priority"] == "fixing the rule of motion"
    failure = _incoherences(report, "transition")[0]
    assert failure["kind"] == "mismatch"


def test_verify_flags_conflicting_transitions(tmp_path: Path) -> None:
    lines = [
        {"o": _obs(0), "a": 1, "r": 0.0, "o2": _obs(1), "done": False},
        {"o": _obs(0), "a": 1, "r": 0.0, "o2": _obs(2), "done": False},  # same (o,a), diff o2
    ]
    report = _run_verify(_scaffold(tmp_path, lines=lines))
    assert report["n_conflicting_transitions"] == 1


def test_verify_handles_none_reward_at_episode_start(tmp_path: Path) -> None:
    start = {
        "frame": {"pos": 0, "cells": []},
        "reward": None,
        "is_done": False,
        "available_actions": [1],
        "info": {},
    }
    lines = [{"o": start, "a": 1, "r": 0.0, "o2": _obs(1), "done": False}]
    report = _run_verify(_scaffold(tmp_path, lines=lines))  # identity model
    assert report["representation_coherence"] == 1.0  # None handled, not a crash


# --------------------------------------------------------------------------- #
# End-to-end through run_task (scripted agent + fake env)
# --------------------------------------------------------------------------- #
_FORWARD = """\
class Controller:
    def act(self, obs):
        return 1


def get_controller():
    return Controller()
"""


class _FakeProblem(BaseProblem):
    name = "fake"

    def make_env(self, task_name: str) -> Any:
        return FakeNativeEnv(goal=3)

    def get_task_names(self) -> list[str]:
        return ["corridor"]

    def obs_renderer(self, task_name: str, *, mode: Any) -> RawRenderer:
        return RawRenderer()

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        return {"success": final_obs.is_done, "steps": steps}

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(episodes) or 1
        return {"success_rate": sum(bool(e.get("success")) for e in episodes) / n}

    def build_prompt(self, task_name: str, *, info_mode: Any) -> str:
        return "# Game: fake\nReach the goal."

    def config_kwargs(self) -> dict[str, Any]:
        return {}


class _WritingAgent(ScriptedAgent):
    async def start(self, *, cwd: str, **kwargs: Any) -> None:
        await super().start(cwd=cwd, **kwargs)
        Path(cwd, "solution.py").write_text(_FORWARD)


@pytest.mark.integration
async def test_run_task_with_cwm_records_and_verifies(tmp_path: Path) -> None:
    config = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake"),
        controller=ControllerConfig(max_moves=10),
        features={"cwm": {"max_tested_transitions_per_verify": 50}},
        limits=LimitsConfig(max_turns=10),
    )
    agent = _WritingAgent(
        [
            [TextDelta("Submitting."), ToolCall("c1", "SubmitSolution", {}), TurnComplete()],
            [ToolCall("c2", "ExitTask", {}), TurnComplete()],
        ]
    )
    reason = await run_task(
        config, _FakeProblem(), "corridor", output_dir=str(tmp_path), agent=agent
    )
    assert reason == "agent_exit"

    workdir = tmp_path / "workdir"
    assert (workdir / "world_model" / "verify.py").exists()
    transcript = (tmp_path / "logs" / "transcript.jsonl").read_text()
    assert "Code World Model" in transcript  # the prompt fragment reached the system prompt

    # The submission + final evals stepped the env through HTTP: all recorded,
    # into both the agent mirror and the trusted canonical (outside the workdir).
    lines = _lines(tmp_path)
    assert len(lines) >= 3
    assert all({"o", "a", "r", "o2", "done"} <= set(line) for line in lines)
    assert any(line["done"] for line in lines)
    assert _canonical_lines(tmp_path) == lines

    # Full circle: an identity model verifies at coherence 1.0 on the recorded data.
    (workdir / "world_model" / "model_parser.py").write_text(_IDENTITY_PARSER)
    (workdir / "world_model" / "model_render.py").write_text(_IDENTITY_RENDER)
    report = _run_verify(workdir / "world_model" / "verify.py")
    assert report["representation_coherence"] == 1.0
    assert report["coverage"]["n_transitions_total"] == len(lines)
