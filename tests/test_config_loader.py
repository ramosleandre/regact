"""Unit tests: the shared dict -> RunConfig mapper + the Kaggle profile loader."""

import json
from pathlib import Path

import pytest

from regact.config.loader import run_config_from_mapping
from regact.config.schema import AgentName, InfoMode, Lifecycle, redacted_config_dict
from regact.run_kaggle import build_run_config_from_profile

_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "regact"
    / "conf"
    / "experiment"
    / "competition.yaml"
)


def test_mapping_builds_typed_config_with_enums() -> None:
    config = run_config_from_mapping(
        {
            "agent": {"name": "claude", "model": "x"},
            "problem": {
                "name": "arc_agi",
                "tasks": ["ls20"],
                "lifecycle": "single_instance",
                "info_mode": "minimal",
                "kwargs": {"operation_mode": "offline"},
            },
            "features": ["controller"],
            "parallel_workers": 4,
        }
    )
    assert config.agent.name is AgentName.CLAUDE
    assert config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE
    assert config.problem.tasks == ["ls20"]
    assert config.problem.info_mode is InfoMode.MINIMAL
    assert config.problem.kwargs == {"operation_mode": "offline"}
    assert config.parallel_workers == 4


def test_mapping_defaults() -> None:
    config = run_config_from_mapping(
        {"agent": {"name": "scripted"}, "problem": {"name": "minigrid"}}
    )
    assert config.features == {"controller": {}}
    assert config.problem.lifecycle is Lifecycle.MULTI_INSTANCE
    assert config.problem.tasks == []


def test_mapping_preserves_controller_eval_knobs() -> None:
    """The controller's eval knobs (record_video/shadow_replay) live on the feature and
    must survive the mapping (regression: as run-level flags they were once dropped, so
    `shadow_replay: true` silently ran with the anti-cheat replay OFF)."""
    cfg = run_config_from_mapping(
        {
            "agent": {"name": "scripted"},
            "problem": {"name": "arc_agi"},
            "features": {"controller": {"record_video": False, "shadow_replay": True}},
        }
    )
    assert cfg.features["controller"] == {"record_video": False, "shadow_replay": True}


def test_mapping_parses_sandbox_bool() -> None:
    on = run_config_from_mapping(
        {
            "agent": {"name": "scripted"},
            "problem": {"name": "arc_agi"},
            "sandbox": True,
        }
    )
    assert on.sandbox is True
    off = run_config_from_mapping({"agent": {"name": "scripted"}, "problem": {"name": "arc_agi"}})
    assert off.sandbox is False
    # Legacy backend-name strings must fail loudly, not coerce (bool("none") is True).
    with pytest.raises(ValueError, match="sandbox"):
        run_config_from_mapping(
            {
                "agent": {"name": "scripted"},
                "problem": {"name": "arc_agi"},
                "sandbox": "none",
            }
        )


def test_mapping_normalizes_features() -> None:
    # A mapping keeps per-feature params; a plain name list means no params.
    cfg = run_config_from_mapping(
        {
            "agent": {"name": "scripted"},
            "problem": {"name": "arc_agi"},
            "features": {"controller": {"n_episodes": 3}},
        }
    )
    assert cfg.features == {"controller": {"n_episodes": 3}}
    legacy = run_config_from_mapping(
        {
            "agent": {"name": "scripted"},
            "problem": {"name": "arc_agi"},
            "features": ["controller"],
        }
    )
    assert legacy.features == {"controller": {}}


def test_redacted_config_dict_masks_api_key() -> None:
    """config.json must never carry a real api_key (it lands in the experiment dir)."""
    config = run_config_from_mapping(
        {"agent": {"name": "claude", "api_key": "sk-secret-123"}, "problem": {"name": "arc_agi"}}
    )
    dumped = redacted_config_dict(config)
    assert dumped["agent"]["api_key"] == "***redacted***"
    assert "sk-secret-123" not in json.dumps(dumped)
    # A None key (the common CLI-auth case) stays None, not the mask.
    plain = run_config_from_mapping({"agent": {"name": "claude"}, "problem": {"name": "arc_agi"}})
    assert redacted_config_dict(plain)["agent"]["api_key"] is None


def test_competition_profile_loads() -> None:
    # The Kaggle profile drives the in-process Alan agent against a local vLLM
    # endpoint (the cloud CLI agents are unreachable offline), single make per game.
    config = build_run_config_from_profile(str(_PROFILE))
    assert config.problem.name == "arc_agi"
    assert config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE
    assert config.agent.name is AgentName.ALAN
    assert config.agent.model == "openai/qwen3-14b"  # default model knob
    assert config.agent.base_url == "http://127.0.0.1:8000/v1"  # default vLLM endpoint
    # default offline; ARC_OPERATION_MODE flips it to online for the gateway
    assert config.problem.kwargs["operation_mode"] == "offline"
    assert config.sandbox is False  # the ARC gateway already isolates the kernel


def test_competition_profile_env_knobs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The notebook switches model/endpoint/mode via env vars without editing the YAML."""
    monkeypatch.setenv("LLM_MODEL_NAME", "qwen3-32b")
    monkeypatch.setenv("AGENT_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("ARC_OPERATION_MODE", "online")
    config = build_run_config_from_profile(str(_PROFILE))
    assert config.agent.model == "openai/qwen3-32b"
    assert config.agent.base_url == "http://127.0.0.1:9999/v1"
    assert config.problem.kwargs["operation_mode"] == "online"


def test_run_exp_hydra_composes_a_config() -> None:
    """run_exp's path: Hydra composes conf/config.yaml + CLI overrides -> RunConfig."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import regact

    conf_dir = str(Path(regact.__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        # Select config groups (per-CLI / per-game yaml) + override a field.
        cfg = compose(
            config_name="config",
            overrides=["agent=claude", "problem=arc_agi", "agent.args.effort=high"],
        )
    config = run_config_from_mapping(OmegaConf.to_container(cfg, resolve=True))
    assert config.problem.name == "arc_agi"
    assert config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE  # from the arc_agi group
    assert config.agent.name is AgentName.CLAUDE
    assert config.agent.args["permission_mode"] == "bypassPermissions"  # from the claude group
    assert config.agent.args["effort"] == "high"  # CLI override
    # The features group carries each feature's own knobs.
    assert config.features == {
        "controller": {
            "n_episodes": 1,
            "max_moves": 2500,
            "record_video": True,
            "shadow_replay": True,
        }
    }


def test_minigrid_suite_groups_compose() -> None:
    """The named MiniGrid configs expose the exact lite and full catalogues."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import regact
    from regact.problems.minigrid import ALL_MINIGRID_TASKS, LITE_MINIGRID_TASKS

    conf_dir = str(Path(regact.__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        lite_raw = compose(config_name="config", overrides=["problem=minigrid_lite"])
        full_raw = compose(config_name="config", overrides=["problem=minigrid_full"])

    lite = run_config_from_mapping(OmegaConf.to_container(lite_raw, resolve=True))
    full = run_config_from_mapping(OmegaConf.to_container(full_raw, resolve=True))
    assert lite.problem.tasks == list(LITE_MINIGRID_TASKS)
    assert full.problem.tasks == []
    # Empty means the problem's complete catalogue.
    assert len(ALL_MINIGRID_TASKS) == 72


def test_experiment_profile_respects_cli_agent_override() -> None:
    """An experiment profile SELECTS groups (defaults: override /agent), it must not merge
    a partial agent dict — else `experiment=research agent=codex` yields a claude/codex
    hybrid and the wrong backend launches (regression guard)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import regact

    conf_dir = str(Path(regact.__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=["experiment=research", "agent=codex"])
    config = run_config_from_mapping(OmegaConf.to_container(cfg, resolve=True))
    assert config.agent.name is AgentName.CODEX  # the CLI choice wins, not the profile's
    assert "reasoning_effort" in config.agent.args  # codex's own args, no claude leftovers
    # the profile's experiment fields still apply (feature-owned knob)
    assert config.features["controller"]["shadow_replay"] is True
