"""CWM v4 settings. Unknown keys fail rather than silently changing an experiment."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannerConfig:
    algorithm: str = "bfs"
    max_seconds_per_planner_call: float = 30
    max_cwm_calls_per_planner_call: int = 10000
    max_nodes_per_planner_call: int = 10000
    max_depth_per_planner_call: int = 100


@dataclass
class ExecutionConfig:
    max_seconds_per_call: float = 5
    max_seconds_per_validation: float = 120
    max_seconds_per_rollout: float = 120
    max_memory_mb: int = 512  # MiB


@dataclass
class FeedbackConfig:
    max_counterexamples: int = 5
    max_diff_items: int = 6
    max_error_chars: int = 1000


@dataclass
class DataApiConfig:
    max_items: int = 100
    max_response_bytes: int = 2097152


@dataclass
class CwmConfig:
    initial_unique_observations: int = 15
    threshold_state_obs_size_ratio: float = 0.5
    cwm_validation_policy: str = "required"
    max_actions_per_exploration: int = 2500
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
        if config.threshold_state_obs_size_ratio >= 1:
            raise ValueError("protocol.threshold_state_obs_size_ratio must be < 1")
        if config.planner.max_depth_per_planner_call > config.max_actions_per_exploration:
            raise ValueError("planner depth cannot exceed protocol.max_actions_per_exploration")
        return config
