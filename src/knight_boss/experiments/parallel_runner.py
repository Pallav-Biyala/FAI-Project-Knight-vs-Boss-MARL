"""Parallel and custom-axis experiment sweep execution."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import torch

from knight_boss.experiments.parameter_sweep import ExperimentConfig, parameter_grid
from knight_boss.game.config import GameConfig
from knight_boss.training.config import PPOConfig
from knight_boss.training.regimes import TrainingRunConfig, run_training


MANIFEST_FIELDS = (
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


@dataclass(frozen=True)
class SweepTrainTask:
    regime: str
    seed: int
    alpha_h: float
    alpha_d: float
    run_config: TrainingRunConfig
    game_config: GameConfig
    ppo_config: PPOConfig
    knight_checkpoint: str | None = None
    boss_checkpoint: str | None = None


def explicit_parameter_grid(
    alpha_h_values: Sequence[float],
    alpha_d_values: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """Return independent (alpha_h, alpha_d) grid pairs from explicit axis lists."""
    if not alpha_h_values or not alpha_d_values:
        raise ValueError("alpha_h_values and alpha_d_values must be non-empty.")
    h_axis = tuple(sorted({round(float(val), 4) for val in alpha_h_values}))
    d_axis = tuple(sorted({round(float(val), 4) for val in alpha_d_values}))
    if any(val <= 0 for val in h_axis) or any(val <= 0 for val in d_axis):
        raise ValueError("All alpha_h and alpha_d multipliers must be positive.")
    return tuple((alpha_h, alpha_d) for alpha_h in h_axis for alpha_d in d_axis)


def _train_single_task(task: SweepTrainTask) -> tuple[SweepTrainTask, dict[str, Any]]:
    torch.set_num_threads(1)
    summary = run_training(
        task.run_config,
        task.game_config,
        task.ppo_config,
        seed=task.seed,
        alpha_h=task.alpha_h,
        alpha_d=task.alpha_d,
        knight_checkpoint=task.knight_checkpoint,
        boss_checkpoint=task.boss_checkpoint,
    )
    return task, summary


def run_parallel_sweep(
    experiment_config: ExperimentConfig,
    game_config: GameConfig,
    ppo_config: PPOConfig,
    regime_configs: dict[str, TrainingRunConfig],
    *,
    parameter_pairs: tuple[tuple[float, float], ...] | None = None,
    max_workers: int = 1,
    knight_checkpoint: str | Path | None = None,
    boss_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run an experiment sweep sequentially or across multiple CPU worker processes."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")
    missing = set(experiment_config.regimes) - set(regime_configs)
    if missing:
        raise ValueError(f"Missing regime configurations: {sorted(missing)}")

    grid = (
        tuple(parameter_pairs)
        if parameter_pairs is not None
        else parameter_grid(experiment_config)
    )
    if not grid:
        raise ValueError("Parameter grid must contain at least one (alpha_h, alpha_d) pair.")

    expected_runs = len(experiment_config.regimes) * len(experiment_config.seeds) * len(grid)
    experiment_id = (
        f"capability-grid-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    experiment_directory = Path(experiment_config.output_root) / experiment_id
    experiment_directory.mkdir(parents=True, exist_ok=False)
    metadata_path = experiment_directory / "experiment.json"
    manifest_path = experiment_directory / "training_manifest.csv"

    k_ckpt = str(knight_checkpoint) if knight_checkpoint is not None else None
    b_ckpt = str(boss_checkpoint) if boss_checkpoint is not None else None

    metadata: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": "running",
        "config": asdict(experiment_config),
        "parameter_grid": [[alpha_h, alpha_d] for alpha_h, alpha_d in grid],
        "game_config": asdict(game_config),
        "ppo_config": asdict(ppo_config),
        "regime_configs": {
            regime: asdict(regime_configs[regime])
            for regime in experiment_config.regimes
        },
        "max_workers": max_workers,
        "warm_start": {
            "knight_checkpoint": str(Path(k_ckpt).resolve()) if k_ckpt else None,
            "boss_checkpoint": str(Path(b_ckpt).resolve()) if b_ckpt else None,
        },
        "expected_run_count": expected_runs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(metadata_path, metadata)

    tasks: list[SweepTrainTask] = []
    runs_root = str(experiment_directory / "runs")
    for regime in experiment_config.regimes:
        base_run_config = regime_configs[regime]
        for seed in experiment_config.seeds:
            for alpha_h, alpha_d in grid:
                run_config = replace(
                    base_run_config,
                    output_root=runs_root,
                    validation_alpha_h=alpha_h,
                    validation_alpha_d=alpha_d,
                )
                tasks.append(
                    SweepTrainTask(
                        regime=regime,
                        seed=seed,
                        alpha_h=alpha_h,
                        alpha_d=alpha_d,
                        run_config=run_config,
                        game_config=game_config,
                        ppo_config=ppo_config,
                        knight_checkpoint=k_ckpt,
                        boss_checkpoint=b_ckpt,
                    )
                )

    completed_runs = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        handle.flush()

        if max_workers == 1 or len(tasks) == 1:
            iterator = (_train_single_task(task) for task in tasks)
            for task, summary in iterator:
                writer.writerow(
                    {
                        "experiment_id": experiment_id,
                        "run_id": summary["run_id"],
                        "regime": task.regime,
                        "seed": task.seed,
                        "training_alpha_h": task.alpha_h,
                        "training_alpha_d": task.alpha_d,
                        "completed_episodes": summary["completed_episodes"],
                        "environment_steps": summary["environment_steps"],
                        "training_directory": summary["run_directory"],
                    }
                )
                handle.flush()
                completed_runs += 1
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
            ) as pool:
                for task, summary in pool.map(_train_single_task, tasks):
                    writer.writerow(
                        {
                            "experiment_id": experiment_id,
                            "run_id": summary["run_id"],
                            "regime": task.regime,
                            "seed": task.seed,
                            "training_alpha_h": task.alpha_h,
                            "training_alpha_d": task.alpha_d,
                            "completed_episodes": summary["completed_episodes"],
                            "environment_steps": summary["environment_steps"],
                            "training_directory": summary["run_directory"],
                        }
                    )
                    handle.flush()
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
