"""Nullable budget behavior, including genuine isolated CWM execution."""

import math
from dataclasses import asdict

import pytest

from regact.config.loader import _controller_from, _limits_from
from regact.config.schema import LimitsConfig
from regact.orchestration.loop import _decide_stop
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.ids import expand_id_ranges
from regact.protocols.cwm.limits import truncate_error
from regact.protocols.cwm.validation import differences
from test_cwm_protocol import exploration, model
from test_cwm_protocol import rig as rig


def no_caps(mapping):
    return {
        k: no_caps(v) if isinstance(v, dict) else None if k.startswith("max_") else v
        for k, v in mapping.items()
    }


def test_every_protocol_max_parameter_accepts_null():
    raw = no_caps(asdict(CwmConfig()))
    cfg = CwmConfig.from_mapping(raw)
    assert asdict(cfg) == raw
    assert cfg.n_unique_observations_in_initial_collection == 20
    assert expand_id_ranges("[1:250]", max_items=None) == list(range(1, 251))
    assert len(differences([1, 2], [3, 4, 5], None)["differences"]) == 3
    for invalid in (-1, 0, True, float("inf")):
        with pytest.raises(ValueError):
            CwmConfig.from_mapping({"execution": {"max_seconds_per_call": invalid}})


def test_shared_null_limits_and_controller_limit():
    raw = {k: None for k in asdict(LimitsConfig())}
    limits = _limits_from(raw)
    assert asdict(limits) == raw
    assert _controller_from({"max_moves": None}).max_moves is None
    assert (
        _decide_stop(
            interrupted=False, turns=10**9, elapsed_s=10**9, tool_calls_total=10**9, limits=limits
        )
        is None
    )
    assert (
        _decide_stop(
            interrupted=False, turns=2, elapsed_s=0, limits=LimitsConfig(max_turns_per_task=2)
        )
        == "loop_limit"
    )


def test_middle_error_truncation():
    text = "START: " + "long context " * 100 + " END: actual cause"
    for cap in (1, 2, 10, 24, 100, 1000):
        short = truncate_error(text, cap)
        assert len(short) == cap
        if cap >= 100:
            assert short.startswith("START: ") and short.endswith("actual cause")
    assert truncate_error(text, None) == text
    assert truncate_error("short", 100) == "short"


def test_all_cwm_caps_disabled_still_finishes(rig):
    c, _ = rig
    raw = no_caps(asdict(c.options))
    c.options = CwmConfig.from_mapping(raw)
    c.collect_initial()
    model(c.workdir)
    result = c.tool("UpdateCodeWorldModel", {})
    assert result["accepted"], result
    (c.workdir / "goal.py").write_text(
        '"""Reach position four."""\ndef achieved(s): return s.n >= 4\n'
    )
    result = c.tool("PlanInCWM", {})
    assert result.get("candidate_found") and result.get("achieved"), result
    exploration(c)
    result = c.tool("RunController", {})
    assert result["stop_reason"] == "plan_exhausted", result
    assert result["real_actions"] == 4
    assert math.isinf(c.deadline)


def test_bulk_byte_cap_never_makes_a_single_record_unreadable(rig):
    c, _ = rig
    c.collect_initial()
    c.options.data_api.max_response_bytes = 1
    with pytest.raises(ValueError, match="smaller batch"):
        c.data({"op": "observations", "ids": [1, 2]})
    assert len(c.data({"op": "observations", "ids": [1]})) == 1
    assert len(c.data({"op": "transitions", "ids": [1]})) == 1
    did = c.store.diagnostic({"kind": "test", "evidence": "large" * 1000})
    assert c.data({"op": "diagnostic", "id": did})["evidence"] == "large" * 1000
    c.options.data_api.max_response_bytes = None
    c.options.data_api.max_items = None
    assert len(c.data({"op": "observations", "ids": "[1:3]"})) == 3
    assert c.data({"op": "list_observation_ids"}) == [1, 2, 3]
    assert c.data({"op": "list_transition_ids"}) == [1, 2]


def test_episode_budget_resets_without_erasing_task_action_count():
    from regact.env.lifecycle import MultiInstancePolicy
    from regact.env.renderer import RawRenderer
    from regact.env.session import EnvSession
    from regact.obs.errors import RegactError
    from regact.testing.fakes import FakeNativeEnv

    session = EnvSession(
        make_native=lambda: FakeNativeEnv(goal=10),
        key="test",
        renderer=RawRenderer(),
        lifecycle=MultiInstancePolicy(),
        step_budget=1,
    )
    try:
        env = session.make()
        env.reset()
        env.step(1)
        with pytest.raises(RegactError):
            env.step(1)
        env.reset()
        env.step(1)
        assert env.episode_action_count == 1
        assert session.total_action_count == 2
        with pytest.raises(RegactError):
            env.step(1)
    finally:
        session.close()


def test_policy_controller_unlimited_actions_stops_on_environment_done():
    from regact.controllers.runner import run_controller
    from test_eval import _client

    class Controller:
        def act(self, obs):
            return 1

    client = _client()
    client.reset()
    result = run_controller(client, Controller(), max_steps=None)
    assert result.stop_kind == "env_done" and result.total_steps == 3


def test_unlimited_callback_still_obeys_operation_deadline(rig):
    c, _ = rig
    c.collect_initial()
    model(c.workdir)
    (c.workdir / "world_model/model_parser.py").write_text(
        "import time\ndef parse(o):\n time.sleep(10)\n"
    )
    c.options.execution.max_seconds_per_call = None
    c.options.execution.max_seconds_per_UpdateCodeWorldModel = 0.3
    result = c.tool("UpdateCodeWorldModel", {})
    assert result["error_type"] == "operation_timeout", result
    assert result["error_context"]["budget"]["value"] == 0.3


def test_long_worker_error_keeps_its_cause_and_can_be_unlimited(rig):
    c, _ = rig
    c.collect_initial()
    model(c.workdir)
    (c.workdir / "world_model/model_parser.py").write_text(
        'def parse(o):\n raise ValueError("BEGIN: " + "x" * 6000 + " END: cause")\n'
    )
    c.options.feedback.max_error_chars = 100
    result = c.tool("UpdateCodeWorldModel", {})
    assert len(result["error"]) == 100
    assert result["error"].startswith("ValueError: BEGIN:")
    assert result["error"].endswith("END: cause")
    c.options.feedback.max_error_chars = None
    result = c.tool("UpdateCodeWorldModel", {})
    assert len(result["error"]) > 6000 and result["error"].endswith("END: cause")
