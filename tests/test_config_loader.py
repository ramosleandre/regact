"""Unit tests: the shared dict -> RunConfig mapper + the Kaggle profile loader."""

from pathlib import Path

from regact.config.loader import run_config_from_mapping
from regact.config.schema import AgentName, Execution, InfoMode, Lifecycle
from regact.run_kaggle import build_run_config_from_profile

_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "regact"
    / "config"
    / "profile"
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
    config = build_run_config_from_profile(str(_PROFILE))
    assert config.problem.name == "arc_agi"
    assert config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE
    assert config.agent.name is AgentName.CLAUDE
    assert config.problem.kwargs["operation_mode"] == "offline"


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
