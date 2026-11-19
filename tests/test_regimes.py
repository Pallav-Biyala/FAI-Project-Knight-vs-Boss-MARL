import csv
import json

import pytest

from knight_boss.game.config import GameConfig
from knight_boss.training.config import PPOConfig
from knight_boss.training.regimes import (
    REGIMES,
    TrainingRunConfig,
    run_training,
)


def tiny_ppo_config():
    return PPOConfig(
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=1,
        batch_size=1,
        rollout_steps=8,
        hidden_sizes=(8,),
        max_grad_norm=0.5,
        device="cpu",
    )


def tiny_game_config():
    return GameConfig(
        max_steps=1,
        knight_position=(0, 0),
        boss_position=(8, 8),
    )


def run_config(tmp_path, regime, **overrides):
    values = {
        "regime": regime,
        "training_episodes": 2,
        "checkpoint_interval_updates": 1,
        "validation_interval_updates": 1,
        "validation_episodes": 2,
        "validation_seed": 900,
        "validation_alpha_h": 1.0,
        "validation_alpha_d": 1.0,
        "output_root": str(tmp_path),
        "opponent": "heuristic" if regime == "fixed_opponent" else None,
        "phase_episodes": 1 if regime == "alternating_learning" else None,
    }
    values.update(overrides)
    return TrainingRunConfig(**values)


def test_training_regime_configs_are_separate_and_loadable(tmp_path):
    from pathlib import Path

    for regime in REGIMES:
        config = TrainingRunConfig.from_yaml(
            Path("configs") / f"{regime}.yaml"
        )
        assert config.regime == regime
        assert config.training_episodes > 0
        assert config.output_root == "runs"
    path = tmp_path / "custom.yaml"
    path.write_text(
        "regime: fixed_opponent\n"
        "training_episodes: 2\n"
        "checkpoint_interval_updates: 1\n"
        "validation_interval_updates: 1\n"
        "validation_episodes: 2\n"
        "validation_seed: 900\n"
        "validation_alpha_h: 1.0\n"
        "validation_alpha_d: 1.0\n"
        "output_root: runs\n"
        "opponent: heuristic\n",
        encoding="utf-8",
    )
    assert TrainingRunConfig.from_yaml(path).opponent == "heuristic"


def test_fixed_heuristic_run_trains_only_knight_and_saves_metadata(tmp_path):
    result = run_training(
        run_config(
            tmp_path,
            "fixed_opponent",
            training_episodes=1,
        ),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=6,
        alpha_h=1.5,
        alpha_d=0.75,
    )
    run_dir = tmp_path / "fixed_opponent" / result["run_id"]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["alpha_h"] == 1.5
    assert metadata["alpha_d"] == 0.75
    assert result["completed_episodes"] == 1
    assert (run_dir / "checkpoints" / "knight-final.pt").is_file()
    assert (run_dir / "checkpoints" / "knight-best.pt").is_file()
    assert "best_validation_knight_win_rate" in metadata
    assert not (run_dir / "checkpoints" / "boss-final.pt").exists()
    with (run_dir / "training.csv").open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert len(rows) == 1
    assert rows[0]["phase"] == "knight_vs_heuristic"
    assert rows[0]["agent"] == "knight"
    with (run_dir / "validation.csv").open(newline="", encoding="utf-8") as file:
        validation_rows = list(csv.DictReader(file))
    assert len(validation_rows) == 1
    assert validation_rows[0]["training_alpha_h"] == "1.5"
    assert validation_rows[0]["validation_alpha_h"] == "1.0"
    assert validation_rows[0]["validation_seed"] == "900"


def test_simultaneous_regime_updates_and_saves_both_agents(tmp_path):
    result = run_training(
        run_config(tmp_path, "simultaneous_learning", training_episodes=1),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=7,
    )
    run_dir = tmp_path / "simultaneous_learning" / result["run_id"]
    assert (run_dir / "checkpoints" / "knight-final.pt").is_file()
    assert (run_dir / "checkpoints" / "boss-final.pt").is_file()
    with (run_dir / "training.csv").open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert {row["agent"] for row in rows} == {"knight", "boss"}
    assert all(row["phase"] == "simultaneous" for row in rows)


def test_alternating_regime_records_frozen_training_phases(tmp_path):
    result = run_training(
        run_config(
            tmp_path,
            "alternating_learning",
            training_episodes=2,
            phase_episodes=1,
        ),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=8,
    )
    run_dir = tmp_path / "alternating_learning" / result["run_id"]
    assert result["completed_episodes"] == 2
    with (run_dir / "training.csv").open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert [(row["phase"], row["agent"]) for row in rows] == [
        ("knight", "knight"),
        ("boss", "boss"),
    ]


def test_alternating_regime_respects_phase_episodes_across_multiple_rollouts(tmp_path):
    ppo = PPOConfig(
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=1,
        batch_size=1,
        rollout_steps=2,
        hidden_sizes=(8,),
        max_grad_norm=0.5,
        device="cpu",
    )
    result = run_training(
        run_config(
            tmp_path,
            "alternating_learning",
            training_episodes=8,
            phase_episodes=4,
        ),
        tiny_game_config(),
        ppo,
        seed=8,
    )
    run_dir = tmp_path / "alternating_learning" / result["run_id"]
    with (run_dir / "training.csv").open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert [(row["phase"], row["agent"]) for row in rows] == [
        ("knight", "knight"),
        ("knight", "knight"),
        ("boss", "boss"),
        ("boss", "boss"),
    ]


def test_resume_from_checkpoint_continues_update_count_and_records_metadata(tmp_path):
    first = run_training(
        run_config(tmp_path, "simultaneous_learning", training_episodes=1),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=9,
    )
    first_dir = tmp_path / "simultaneous_learning" / first["run_id"]
    knight_ckpt = first_dir / "checkpoints" / "knight-final.pt"
    boss_ckpt = first_dir / "checkpoints" / "boss-final.pt"

    second = run_training(
        run_config(tmp_path, "simultaneous_learning", training_episodes=1),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=10,
        knight_checkpoint=knight_ckpt,
        boss_checkpoint=boss_ckpt,
    )
    second_dir = tmp_path / "simultaneous_learning" / second["run_id"]
    metadata = json.loads((second_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "knight" in metadata["resumed_from"]
    assert "boss" in metadata["resumed_from"]
    with (second_dir / "training.csv").open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert [int(row["update"]) for row in rows] == [2, 2]


def test_resume_boss_in_fixed_opponent_is_rejected(tmp_path):
    first = run_training(
        run_config(tmp_path, "simultaneous_learning", training_episodes=1),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=11,
    )
    boss_ckpt = (
        tmp_path
        / "simultaneous_learning"
        / first["run_id"]
        / "checkpoints"
        / "boss-final.pt"
    )
    with pytest.raises(ValueError, match="not a PPOAgent"):
        run_training(
            run_config(tmp_path, "fixed_opponent", training_episodes=1),
            tiny_game_config(),
            tiny_ppo_config(),
            seed=12,
            boss_checkpoint=boss_ckpt,
        )


def test_step_budget_stops_at_exact_environment_step_count(tmp_path):
    result = run_training(
        run_config(tmp_path, "simultaneous_learning", training_steps=20),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=3,
    )
    assert result["environment_steps"] == 20


def test_alternating_step_phases_switch_agents_every_phase_steps(tmp_path):
    result = run_training(
        run_config(
            tmp_path,
            "alternating_learning",
            training_steps=32,
            phase_steps=8,
            phase_episodes=None,
        ),
        tiny_game_config(),
        tiny_ppo_config(),
        seed=4,
    )
    assert result["environment_steps"] == 32
    run_dir = tmp_path / "alternating_learning" / result["run_id"]
    with (run_dir / "training.csv").open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert [row["agent"] for row in rows] == ["knight", "boss", "knight", "boss"]


def test_alternating_step_budget_requires_phase_steps(tmp_path):
    with pytest.raises(ValueError):
        run_config(
            tmp_path,
            "alternating_learning",
            training_steps=32,
            phase_episodes=None,
        )
