"""CWM settings. Unknown keys fail rather than silently changing an experiment."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannerConfig:
    enabled: bool = False
    algorithm: str = "bfs"
    max_seconds_per_planner_call: float | None = 30
    max_cwm_calls_per_planner_call: int | None = 10000
    max_nodes_per_planner_call: int | None = 10000
    max_depth_per_planner_call: int | None = 100


@dataclass
class ExecutionConfig:
    max_seconds_per_call: float | None = 5
    # Below the 120 s the agents' shells (Claude Code, Alan) give one command: a call at its full
    # budget plus cleanup must still return before the shell kills the command that started it.
    max_seconds_per_UpdateCodeWorldModel: float | None = 90
    max_seconds_per_RunController: float | None = 90
    max_memory_mb: int | None = 512  # MiB


@dataclass
class FeedbackConfig:
    max_counterexamples: int | None = 5
    max_diff_items: int | None = 6
    max_error_chars: int | None = 1000


@dataclass
class DataApiConfig:
    max_items: int | None = 100
    max_response_bytes: int | None = 2097152


@dataclass
class CwmConfig:
    n_unique_observations_in_initial_collection: int = 20
    max_actions_per_initial_collection: int | None = 1000
    max_seconds_per_initial_collection: float | None = 30
    threshold_max_state_obs_size_ratio: float = 0.5
    cwm_validation_policy: str = "required"
    max_actions_per_exploration: int | None = 2500
    n_tmp_images_saved_per_exploration: int = 0
    workspace_helpers_enabled: bool = True
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    data_api: DataApiConfig = field(default_factory=DataApiConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CwmConfig:
        values = dict(raw)
        for key, typ in (
            ("planner", PlannerConfig),
            ("execution", ExecutionConfig),
            ("feedback", FeedbackConfig),
            ("data_api", DataApiConfig),
        ):
            values[key] = typ(**dict(values.get(key) or {}))
        config = cls(**values)
        if config.cwm_validation_policy != "required":
            raise ValueError("protocol.cwm_validation_policy supports only 'required'")
        if config.planner.algorithm != "bfs":
            raise ValueError("protocol.planner.algorithm supports only 'bfs'")

        def validate(items: dict[str, Any], prefix: str) -> None:
            for key, value in items.items():
                if isinstance(value, dict):
                    validate(value, prefix + key + ".")
                elif key.startswith("max_") and value is None:
                    continue
                elif key in ("workspace_helpers_enabled", "enabled"):
                    if type(value) is not bool:
                        raise ValueError(f"{prefix}{key} must be a boolean")
                elif key == "n_tmp_images_saved_per_exploration":
                    if type(value) is not int or value < 0:
                        raise ValueError(f"{prefix}{key} must be a nonnegative integer")
                elif key not in ("algorithm", "cwm_validation_policy"):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise ValueError(f"{prefix}{key} must be a positive finite number")
                    if "seconds" not in key and "ratio" not in key and not isinstance(value, int):
                        raise ValueError(f"{prefix}{key} must be an integer")

        validate(dataclasses.asdict(config), "protocol.")
        if config.threshold_max_state_obs_size_ratio >= 1:
            raise ValueError("protocol.threshold_max_state_obs_size_ratio must be < 1")
        if (
            config.planner.enabled
            and config.max_actions_per_exploration is not None
            and (
                config.planner.max_depth_per_planner_call is not None
                and config.planner.max_depth_per_planner_call > config.max_actions_per_exploration
            )
        ):
            raise ValueError("planner depth cannot exceed protocol.max_actions_per_exploration")
        return config
