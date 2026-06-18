"""Competition entry point (Kaggle): argparse + a YAML profile -> RunConfig.

No Hydra here — a plain profile YAML (single_instance + offline ARC by default)
loaded into the typed config, with a few CLI flags for the notebook cell. Builds
the same :class:`RunConfig` as ``run_exp`` and runs the same experiment.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, cast

from omegaconf import OmegaConf

from regact.config.loader import run_config_from_mapping
from regact.config.schema import Execution, RunConfig
from regact.orchestration.experiment import run_experiment


def build_run_config_from_profile(profile_path: str) -> RunConfig:
    """Load a plain-YAML profile into a typed :class:`RunConfig`."""
    raw = OmegaConf.to_container(OmegaConf.load(profile_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"profile {profile_path!r} must be a mapping")
    return run_config_from_mapping(cast("dict[str, Any]", raw))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="regact.run_kaggle")
    parser.add_argument("--config", required=True, help="Path to a profile YAML.")
    parser.add_argument("--games", nargs="*", default=None, help="Override the task list.")
    parser.add_argument("--parallel", type=int, default=None, help="Worker count (>1 => parallel).")
    parser.add_argument("--output-root", default=None, help="Where to write experiment outputs.")
    return parser.parse_args(argv)


def run_kaggle(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = build_run_config_from_profile(args.config)
    if args.games is not None:
        config.task_names = args.games
    if args.parallel is not None:
        config.parallel_workers = args.parallel
        config.execution = Execution.PARALLEL if args.parallel > 1 else Execution.SEQUENTIAL
    if args.output_root is not None:
        config.output_root = args.output_root

    reasons = asyncio.run(run_experiment(config))
    for task, reason in reasons.items():
        print(f"{task}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_kaggle())
