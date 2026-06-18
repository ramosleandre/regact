"""Research entry point: Hydra composes the config; we run the experiment.

Hydra handles composition + CLI overrides (``problem.name=arc_agi agent.name=claude
problem.lifecycle=single_instance``) and sweeps; ``chdir`` is off (our Scheduler
owns where things run). The composed config is converted to a plain dict and mapped
to the typed :class:`RunConfig` — the same path ``run_kaggle`` uses — so both
front-ends share one config schema and one ``run_experiment``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from regact.config.loader import run_config_from_mapping
from regact.orchestration.experiment import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    raw: Any = OmegaConf.to_container(cfg, resolve=True)
    config = run_config_from_mapping(raw)
    reasons = asyncio.run(run_experiment(config))
    for task, reason in reasons.items():
        print(f"{task}: {reason}")


if __name__ == "__main__":
    main()
