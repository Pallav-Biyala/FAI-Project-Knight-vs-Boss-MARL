import csv
import json
from pathlib import Path

import pytest

from knight_boss.experiments.parameter_sweep import (
    ExperimentConfig,
    parameter_grid,
    run_sweep,
)
from knight_boss.game.config import GameConfig
from knight_boss.training.config import PPOConfig
from knight_boss.training.regimes import TrainingRunConfig


def test_default_grid_has_121_independent_configurations():
    config = ExperimentConfig.from_yaml(Path("configs") / "experiments.yaml")
    grid = parameter_grid(config)
    assert len(grid) == 121
    assert grid[0] == (0.5, 0.5)
    assert grid[-1] == (3.0, 3.0)
    assert len({alpha_h for alpha_h, _ in grid}) == 11
    assert len({alpha_d for _, alpha_d in grid}) == 11
    assert (0.5, 3.0) in grid
    assert (3.0, 0.5) in grid
    assert config.expected_run_count == 2420


def test_parameter_grid_does_not_combine_multipliers():
    config = ExperimentConfig(
        alpha_h_min=0.5,
        alpha_h_max=1.0,
        alpha_d_min=0.5,
        alpha_d_max=1.0,
        resolution=0.25,
        seeds=(0,),
        regimes=("fixed_opponent",),
        output_root="runs",
    )
    assert parameter_grid(config) == (
        (0.5, 0.5),
        (0.5, 0.75),
        (0.5, 1.0),
        (0.75, 0.5),
        (0.75, 0.75),
        (0.75, 1.0),
        (1.0, 0.5),
        (1.0, 0.75),
        (1.0, 1.0),
    )


def test_invalid_ranges_and_grid_resolution_are_rejected():
    with pytest.raises(ValueError, match="align with resolution"):
        ExperimentConfig(
            alpha_h_min=0.5,
            alpha_h_max=1.1,
            alpha_d_min=0.5,
            alpha_d_max=1.0,
            resolution=0.25,
            seeds=(0,),
            regimes=("fixed_opponent",),
            output_root="runs",
        )
    with pytest.raises(ValueError, match="positive"):
        ExperimentConfig(
            alpha_h_min=0.5,
            alpha_h_max=1.0,
            alpha_d_min=0.5,
            alpha_d_max=1.0,
            resolution=0.0,
            seeds=(0,),
            regimes=("fixed_opponent",),
            output_root="runs",
        )


def test_sweep_records_training_parameters_separately_from_evaluation(tmp_path, monkeypatch):
    config = ExperimentConfig(
        alpha_h_min=0.5,
        alpha_h_max=0.5,
        alpha_d_min=0.5,
        alpha_d_max=0.75,
        resolution=0.25,
        seeds=(3, 4),
        regimes=("fixed_opponent",),
        output_root=str(tmp_path),
    )
    base_run = TrainingRunConfig(
        regime="fixed_opponent",
        training_episodes=1,
        checkpoint_interval_updates=1,
        validation_interval_updates=1,
        validation_episodes=1,
        validation_seed=900,
        validation_alpha_h=1.0,
        validation_alpha_d=1.0,
        output_root="unused",
        opponent="heuristic",
    )
    calls = []

    def fake_run_training(run_config, game_config, ppo_config, **kwargs):
        calls.append((run_config, kwargs))
        index = len(calls)
        return {
            "run_id": f"run-{index}",
            "run_directory": f"train-{index}",
            "completed_episodes": run_config.training_episodes,
            "environment_steps": 1,
        }

    monkeypatch.setattr(
        "knight_boss.experiments.parameter_sweep.run_training",
        fake_run_training,
    )
    result = run_sweep(
        config,
        GameConfig(),
        PPOConfig(
            learning_rate=0.001,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            epochs=1,
            batch_size=1,
            rollout_steps=1,
            hidden_sizes=(8,),
            max_grad_norm=0.5,
            device="cpu",
        ),
        {"fixed_opponent": base_run},
    )
    assert result["completed_run_count"] == 4
    assert len(calls) == 4
    with Path(result["training_manifest"]).open(newline="", encoding="utf-8") as manifest:
        rows = list(csv.DictReader(manifest))
    assert [(float(row["training_alpha_h"]), float(row["training_alpha_d"])) for row in rows] == [
        (0.5, 0.5),
        (0.5, 0.75),
        (0.5, 0.5),
        (0.5, 0.75),
    ]
    assert [row["seed"] for row in rows] == ["3", "3", "4", "4"]
    metadata = json.loads(
        (Path(result["experiment_directory"]) / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "completed"
    assert metadata["completed_run_count"] == 4
    assert all(call[1]["alpha_h"] != call[1]["alpha_d"] for call in calls[1::2])
