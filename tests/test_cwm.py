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
from regact.config.schema import AgentConfig, AgentName, LimitsConfig, ProblemConfig, RunConfig
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
        "world_model/model_notes.py",
        "world_model/verify.py",
    }
    verify = templates["world_model/verify.py"]
    assert "MAX_OBS_DEFAULT = 123" in verify
    assert "MAX_FAILURES_DEFAULT = 4" in verify
    assert "__MAX_OBS__" not in verify


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
    n_transitions: int = 3,
) -> Path:
    for template in CwmFeature().templates(_ctx(workdir)):
        path = workdir / template.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.content)
    world = workdir / "world_model"
    (world / "model_parser.py").write_text(parser_src)
    (world / "model_render.py").write_text(render_src)
    lines = []
    for i in range(n_transitions):
        done = i == n_transitions - 1
        lines.append(
            json.dumps({"o": _obs(i), "a": 1, "r": 0.0, "o2": _obs(i + 1, done=done), "done": done})
        )
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "transitions.jsonl").write_text("\n".join(lines) + "\n")
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


def test_verify_identity_model_is_fully_coherent(tmp_path: Path) -> None:
    report = _run_verify(_scaffold(tmp_path))
    # 3 distinct o's + the terminal o2.
    assert report["n_obs_tested"] == 4
    assert report["coherence"] == 1.0
    assert report["n_transitions"] == 3
    assert report["failures"] == []


def test_verify_reports_mismatch_with_pointed_diff(tmp_path: Path) -> None:
    render = (
        "def render(state):\n    s = dict(state)\n"
        "    s['frame'] = dict(s['frame'], pos=99)\n    return s\n"
    )
    report = _run_verify(_scaffold(tmp_path, render_src=render))
    assert report["coherence"] == 0.0
    failure = report["failures"][0]
    assert failure["kind"] == "mismatch"
    assert failure["detail"].startswith("obs.frame.pos: 99 != ")


def test_verify_reports_shape_mismatch(tmp_path: Path) -> None:
    render = (
        "def render(state):\n    s = dict(state)\n"
        "    s['frame'] = dict(s['frame'], cells=[])\n    return s\n"
    )
    report = _run_verify(_scaffold(tmp_path, render_src=render))
    failures = [f for f in report["failures"] if "length" in f["detail"]]
    assert failures and failures[0]["kind"] == "mismatch"


def test_verify_counts_a_raising_parse_as_incoherent(tmp_path: Path) -> None:
    raising = "def parse(obs):\n    raise KeyError('x')\n"
    report = _run_verify(_scaffold(tmp_path, parser_src=raising))
    assert report["coherence"] == 0.0
    assert report["failures"][0]["kind"] == "error"
    assert "KeyError" in report["failures"][0]["detail"]


def test_verify_cap_reports_skipped(tmp_path: Path) -> None:
    report = _run_verify(_scaffold(tmp_path), "--max-obs", "2")
    assert report["n_obs_tested"] == 2
    assert report["n_obs_skipped"] == 2


def test_verify_complexity_ignores_notes_and_itself(tmp_path: Path) -> None:
    script = _scaffold(tmp_path)
    before = _run_verify(script)["complexity"]["ast_nodes"]
    notes = tmp_path / "world_model" / "model_notes.py"
    notes.write_text(notes.read_text() + "\n" + "JUNK = [0]\n" * 30)
    assert _run_verify(script)["complexity"]["ast_nodes"] == before


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
        features={
            "controller": {"max_moves": 10},
            "cwm": {"max_tested_transitions_per_verify": 50},
        },
        limits=LimitsConfig(keep_alive=10),
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
    assert report["coherence"] == 1.0
    assert report["n_transitions"] == len(lines)
