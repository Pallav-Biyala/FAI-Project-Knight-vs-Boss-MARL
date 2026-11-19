import csv
import json
from pathlib import Path
import pytest

from knight_boss.evaluation.sweep_evaluator import (
    evaluate_sweep_manifest,
    resolve_knight_checkpoint,
)
from knight_boss.experiments.parallel_runner import (
    explicit_parameter_grid,
    run_parallel_sweep,
)
from knight_boss.experiments.parameter_sweep import ExperimentConfig
from knight_boss.game.config import GameConfig
from knight_boss.training.config import PPOConfig
from knight_boss.training.regimes import TrainingRunConfig


def _micro_ppo() -> PPOConfig:
    return PPOConfig(
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=1,
        batch_size=4,
        rollout_steps=8,
        hidden_sizes=(8,),
        max_grad_norm=0.5,
        device="cpu",
    )


def _micro_regimes() -> dict[str, TrainingRunConfig]:
    return {
        "fixed_opponent": TrainingRunConfig(
            regime="fixed_opponent",
            training_episodes=2,
            checkpoint_interval_updates=1,
            validation_interval_updates=1,
            validation_episodes=2,
            validation_seed=900,
            validation_alpha_h=1.0,
            validation_alpha_d=1.0,
            output_root="unused",
            opponent="heuristic",
        ),
        "simultaneous_learning": TrainingRunConfig(
            regime="simultaneous_learning",
            training_episodes=2,
            checkpoint_interval_updates=1,
            validation_interval_updates=1,
            validation_episodes=2,
            validation_seed=900,
            validation_alpha_h=1.0,
            validation_alpha_d=1.0,
            output_root="unused",
            reward_mode="shaped",
            shaping_weight=0.5,
        ),
    }


def test_explicit_parameter_grid_keeps_alpha_h_and_alpha_d_independent():
    grid = explicit_parameter_grid([1.5, 0.5, 1.0], [2.0, 0.75])
    assert grid == (
        (0.5, 0.75),
        (0.5, 2.0),
        (1.0, 0.75),
        (1.0, 2.0),
        (1.5, 0.75),
        (1.5, 2.0),
    )
    with pytest.raises(ValueError, match="positive"):
        explicit_parameter_grid([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        explicit_parameter_grid([], [1.0])


def test_parallel_sweep_and_manifest_evaluation_end_to_end(tmp_path: Path):
    exp_cfg = ExperimentConfig(
        alpha_h_min=1.0,
        alpha_h_max=1.5,
        alpha_d_min=1.0,
        alpha_d_max=1.0,
        resolution=0.5,
        seeds=(0, 1),
        regimes=("fixed_opponent", "simultaneous_learning"),
        output_root=str(tmp_path / "experiments"),
    )
    pairs = explicit_parameter_grid([1.0], [1.5])
    sweep_res = run_parallel_sweep(
        exp_cfg,
        GameConfig(),
        _micro_ppo(),
        _micro_regimes(),
        parameter_pairs=pairs,
        max_workers=2,
    )
    assert sweep_res["completed_run_count"] == 4
    exp_meta = json.loads(
        (Path(sweep_res["experiment_directory"]) / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert exp_meta["status"] == "completed"
    assert exp_meta["parameter_grid"] == [[1.0, 1.5]]

    with Path(sweep_res["training_manifest"]).open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == 4
    fixed_dir = Path(manifest_rows[0]["training_directory"])
    fixed_meta = json.loads((fixed_dir / "metadata.json").read_text(encoding="utf-8"))
    assert fixed_meta["run_config"]["validation_alpha_h"] == 1.0
    assert fixed_meta["run_config"]["validation_alpha_d"] == 1.5
    sim_dir = Path(manifest_rows[2]["training_directory"])
    assert resolve_knight_checkpoint(fixed_dir, "auto").name == "knight-best.pt"
    assert resolve_knight_checkpoint(fixed_dir, "final").name == "knight-final.pt"
    assert resolve_knight_checkpoint(sim_dir, "auto").name == "knight-final.pt"
    with pytest.raises(FileNotFoundError):
        resolve_knight_checkpoint(sim_dir, "best")

    # Matched evaluation against HeuristicBoss across all runs in parallel
    eval_heur = evaluate_sweep_manifest(
        sweep_res["training_manifest"],
        GameConfig(),
        episodes=4,
        base_seed=2000,
        opponent_mode="heuristic",
        max_workers=2,
        output_root=str(tmp_path / "results_heur"),
    )
    assert eval_heur["total_summaries"] == 4
    with Path(eval_heur["summaries_csv"]).open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 4
    assert all(float(row["evaluation_alpha_h"]) == 1.0 for row in summary_rows)
    assert all(float(row["evaluation_alpha_d"]) == 1.5 for row in summary_rows)
    assert all(row["boss_opponent_type"] == "heuristic" for row in summary_rows)
    assert {row["reward_mode"] for row in summary_rows} == {"terminal", "shaped"}

    # Grid evaluation using matched opponents (heuristic for fixed, trained boss for simultaneous)
    eval_grid = evaluate_sweep_manifest(
        sweep_res["training_manifest"],
        GameConfig(),
        episodes=3,
        base_seed=3000,
        opponent_mode="matched",
        grid_pairs=((0.75, 1.0), (1.25, 1.0)),
        max_workers=2,
        save_episodes=False,
        output_root=str(tmp_path / "results_grid"),
    )
    assert eval_grid["total_summaries"] == 8
    assert eval_grid["episodes_csv"] is None
    with Path(eval_grid["summaries_csv"]).open(newline="", encoding="utf-8") as handle:
        grid_rows = list(csv.DictReader(handle))
    assert len(grid_rows) == 8
    assert any(row["boss_opponent_type"] == "ppo_trained" for row in grid_rows)
    assert any(row["boss_opponent_type"] == "heuristic" for row in grid_rows)


def test_combat_reward_mode_adds_approach_and_anti_stall_penalties():
    from knight_boss.marl.parallel_env import KnightBossParallelEnv
    from knight_boss.marl.rewards import combat_rewards
    from knight_boss.game.types import Outcome

    # Moving apart from d=1 to d=2 subtracts distance_weight (0.01) + step_penalty (0.005).
    r_back = combat_rewards(
        None,
        truncated=False,
        knight_damage_received_delta=0,
        boss_damage_received_delta=0,
        knight_max_health=150,
        boss_max_health=150,
        prev_distance=1,
        curr_distance=2,
    )
    assert r_back["knight"] == pytest.approx(-0.015)
    assert r_back["boss"] == pytest.approx(-0.015)

    # Closing from d=6 to d=5 gives +0.01 - 0.005 = +0.005 symmetrically.
    r_close = combat_rewards(
        None,
        truncated=False,
        knight_damage_received_delta=0,
        boss_damage_received_delta=0,
        knight_max_health=150,
        boss_max_health=150,
        prev_distance=6,
        curr_distance=5,
    )
    assert r_close["knight"] == pytest.approx(0.005)
    assert r_close["boss"] == pytest.approx(0.005)

    # Timeout adds -0.5 to both agents; default env stays terminal.
    r_to = combat_rewards(
        Outcome.DRAW,
        truncated=True,
        knight_damage_received_delta=0,
        boss_damage_received_delta=0,
        knight_max_health=150,
        boss_max_health=150,
        prev_distance=4,
        curr_distance=4,
    )
    assert r_to["knight"] == pytest.approx(-0.505)
    env = KnightBossParallelEnv(GameConfig(), reward_mode="combat")
    obs, _ = env.reset(seed=7)
    _, rew, _, _, _ = env.step({"knight": 3, "boss": 2})  # move closer: d=6 -> d=4
    assert rew["knight"] == pytest.approx(0.015)
    assert rew["boss"] == pytest.approx(0.015)


def test_draw_penalty_reward_mode_penalizes_draws_without_step_bias():
    from knight_boss.marl.parallel_env import KnightBossParallelEnv
    from knight_boss.marl.rewards import draw_penalty_rewards
    from knight_boss.game.types import Outcome

    r_step = draw_penalty_rewards(
        None,
        truncated=False,
        knight_damage_received_delta=0,
        boss_damage_received_delta=0,
        knight_max_health=150,
        boss_max_health=150,
    )
    assert r_step == {"knight": 0.0, "boss": 0.0}

    r_draw = draw_penalty_rewards(
        Outcome.DRAW,
        truncated=True,
        knight_damage_received_delta=0,
        boss_damage_received_delta=0,
        knight_max_health=150,
        boss_max_health=150,
    )
    assert r_draw == {"knight": -0.75, "boss": -0.75}

    r_win = draw_penalty_rewards(
        Outcome.KNIGHT_WIN,
        truncated=False,
        knight_damage_received_delta=0,
        boss_damage_received_delta=30,
        knight_max_health=150,
        boss_max_health=150,
    )
    assert r_win["knight"] == pytest.approx(1.1)
    assert r_win["boss"] == pytest.approx(-1.1)

    env = KnightBossParallelEnv(GameConfig(), reward_mode="draw_penalty")
    env.reset(seed=7)
    _, rew, _, _, _ = env.step({"knight": 3, "boss": 2})
    assert rew == {"knight": 0.0, "boss": 0.0}
