"""Independent Alpha_H × Alpha_D grid and multi-seed training runner."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from knight_boss.game.config import GameConfig
from knight_boss.training.config import PPOConfig
from knight_boss.training.regimes import REGIMES, TrainingRunConfig, run_training


@dataclass(frozen=True)
class ExperimentConfig:
    alpha_h_min: float
    alpha_h_max: float
    alpha_d_min: float
    alpha_d_max: float
    resolution: float
    seeds: tuple[int, ...]
    regimes: tuple[str, ...]
    output_root: str

    def __post_init__(self) -> None:
        if not isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be finite and positive.")
        for label, minimum, maximum in (
            ("alpha_h", self.alpha_h_min, self.alpha_h_max),
            ("alpha_d", self.alpha_d_min, self.alpha_d_max),
        ):
            if (
                not isfinite(minimum)
                or not isfinite(maximum)
                or minimum <= 0
                or maximum < minimum
            ):
                raise ValueError(f"Invalid {label} range.")
            if not _is_on_grid(minimum, maximum, self.resolution):
                raise ValueError(f"{label} range must align with resolution.")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("At least one non-negative random seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Experiment seeds must be unique.")
        if not self.regimes or not set(self.regimes).issubset(REGIMES):
            raise ValueError("regimes must select known training regimes.")
        if len(set(self.regimes)) != len(self.regimes):
            raise ValueError("Experiment regimes must be unique.")
        if not self.output_root:
            raise ValueError("output_root must be specified.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            data: Any = yaml.safe_load(config_file)
        if not isinstance(data, dict):
            raise ValueError("Experiment configuration must be a YAML mapping.")
        alpha_h = data["alpha_h"]
        alpha_d = data["alpha_d"]
        return cls(
            alpha_h_min=float(alpha_h["min"]),
            alpha_h_max=float(alpha_h["max"]),
            alpha_d_min=float(alpha_d["min"]),
            alpha_d_max=float(alpha_d["max"]),
            resolution=float(data["resolution"]),
            seeds=tuple(int(seed) for seed in data["seeds"]),
            regimes=tuple(str(regime) for regime in data["regimes"]),
            output_root=str(data["output_root"]),
        )

    @property
    def expected_run_count(self) -> int:
        grid_size = len(_decimal_range(
            self.alpha_h_min, self.alpha_h_max, self.resolution
        )) * len(_decimal_range(
            self.alpha_d_min, self.alpha_d_max, self.resolution
        ))
        return grid_size * len(self.seeds) * len(self.regimes)


def parameter_grid(config: ExperimentConfig) -> tuple[tuple[float, float], ...]:
    """Return all independent (alpha_h, alpha_d) pairs in stable order."""
    alpha_h_values = _decimal_range(
        config.alpha_h_min, config.alpha_h_max, config.resolution
    )
    alpha_d_values = _decimal_range(
        config.alpha_d_min, config.alpha_d_max, config.resolution
    )
    return tuple(
        (float(alpha_h), float(alpha_d))
        for alpha_h in alpha_h_values
        for alpha_d in alpha_d_values
    )


def run_sweep(
    experiment_config: ExperimentConfig,
    game_config: GameConfig,
    ppo_config: PPOConfig,
    regime_configs: dict[str, TrainingRunConfig],
) -> dict[str, Any]:
    missing = set(experiment_config.regimes) - set(regime_configs)
    if missing:
        raise ValueError(f"Missing regime configurations: {sorted(missing)}")
    experiment_id = (
        f"capability-grid-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    experiment_directory = Path(experiment_config.output_root) / experiment_id
    experiment_directory.mkdir(parents=True, exist_ok=False)
    metadata_path = experiment_directory / "experiment.json"
    manifest_path = experiment_directory / "training_manifest.csv"
    metadata = {
        "experiment_id": experiment_id,
        "status": "running",
        "config": asdict(experiment_config),
        "game_config": asdict(game_config),
        "ppo_config": asdict(ppo_config),
        "expected_run_count": experiment_config.expected_run_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(metadata_path, metadata)

    fields = (
        "experiment_id",
        "run_id",
        "regime",
        "seed",
        "training_alpha_h",
        "training_alpha_d",
        "completed_episodes",
        "environment_steps",
        "training_directory",
    )
    completed_runs = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        for regime in experiment_config.regimes:
            base_run_config = regime_configs[regime]
            for seed in experiment_config.seeds:
                for alpha_h, alpha_d in parameter_grid(experiment_config):
                    run_config = replace(
                        base_run_config,
                        output_root=str(experiment_directory / "runs"),
                    )
                    summary = run_training(
                        run_config,
                        game_config,
                        ppo_config,
                        seed=seed,
                        alpha_h=alpha_h,
                        alpha_d=alpha_d,
                    )
                    writer.writerow(
                        {
                            "experiment_id": experiment_id,
                            "run_id": summary["run_id"],
                            "regime": regime,
                            "seed": seed,
                            "training_alpha_h": alpha_h,
                            "training_alpha_d": alpha_d,
                            "completed_episodes": summary["completed_episodes"],
                            "environment_steps": summary["environment_steps"],
                            "training_directory": summary["run_directory"],
                        }
                    )
                    manifest_file.flush()
                    completed_runs += 1

    metadata.update(
        {
            "status": "completed",
            "completed_run_count": completed_runs,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(metadata_path, metadata)
    return {
        "experiment_id": experiment_id,
        "experiment_directory": str(experiment_directory),
        "training_manifest": str(manifest_path),
        "completed_run_count": completed_runs,
    }


def _decimal_range(minimum: float, maximum: float, resolution: float) -> tuple[Decimal, ...]:
    start = Decimal(str(minimum))
    stop = Decimal(str(maximum))
    step = Decimal(str(resolution))
    if step <= 0:
        raise ValueError("resolution must be positive.")
    count = int((stop - start) / step)
    return tuple(start + step * index for index in range(count + 1))


def _is_on_grid(minimum: float, maximum: float, resolution: float) -> bool:
    if resolution <= 0:
        return False
    delta = Decimal(str(maximum)) - Decimal(str(minimum))
    step = Decimal(str(resolution))
    return delta >= 0 and delta % step == 0


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2)
