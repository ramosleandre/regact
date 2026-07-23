"""Unit tests: the shared dict -> RunConfig mapper + the Kaggle profile loader."""

from pathlib import Path

from regact.config.loader import run_config_from_mapping
from regact.config.schema import AgentName, Execution, InfoMode, Lifecycle
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
                "lifecycle": "single_instance",
                "info_mode": "minimal",
                "kwargs": {"operation_mode": "offline"},
            },
            "features": ["controller"],
            "execution": "parallel",
            "parallel_workers": 4,
        }
    )
    assert config.agent.name is AgentName.CLAUDE
    assert config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE
    assert config.problem.info_mode is InfoMode.MINIMAL
    assert config.problem.kwargs == {"operation_mode": "offline"}
    assert config.execution is Execution.PARALLEL
    assert config.parallel_workers == 4


def test_mapping_defaults() -> None:
    config = run_config_from_mapping(
        {"agent": {"name": "scripted"}, "problem": {"name": "minigrid"}}
    )
    assert config.features == ["controller"]
    assert config.execution is Execution.SEQUENTIAL
    assert config.problem.lifecycle is Lifecycle.MULTI_INSTANCE


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
    assert config.security.sandbox.value == "none"  # ARC gateway already isolates the kernel


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
    assert config.features == ["controller"]
