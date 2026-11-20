import pytest

from knight_boss.agents import HeuristicAgent
from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.evaluation.evaluator import evaluate_policies
from knight_boss.evaluation.metrics import EpisodeMetrics, summarize_episodes, wilson_interval
from knight_boss.evaluation.statistics import mean_ci_across_seeds
from knight_boss.evaluation.thresholds import estimate_threshold_curves
from knight_boss.game.config import GameConfig
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.training.config import PPOConfig


def test_wilson_interval_handles_extreme_rates_and_validates_counts():
    lower, upper = wilson_interval(10, 10)
    assert 0.7 < lower < 1.0
    assert upper == 1.0
    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0
    assert 0.0 < upper < 0.4
    with pytest.raises(ValueError, match="trials"):
        wilson_interval(0, 0)


def test_training_seed_statistics_do_not_treat_matches_as_training_runs():
    single = mean_ci_across_seeds([0.6])
    assert single["n_training_seeds"] == 1
    assert single["standard_deviation"] is None
    assert single["ci_lower"] is None
    multiple = mean_ci_across_seeds([0.4, 0.5, 0.6])
    assert multiple["n_training_seeds"] == 3
    assert multiple["mean"] == pytest.approx(0.5)
    assert multiple["standard_deviation"] == pytest.approx(0.1)
    assert multiple["ci_lower"] < 0.5 < multiple["ci_upper"]


def test_summary_includes_proposal_behavior_diagnostics():
    episode = EpisodeMetrics(
        episode_index=0,
        evaluation_seed=1,
        outcome="draw",
        timeout=True,
        episode_length=20,
        knight_damage_dealt=40,
        knight_damage_received=80,
        boss_damage_dealt=80,
        boss_damage_received=40,
        invalid_knight_moves=2,
        special2_casts=3,
        special2_hits=2,
        special2_blocks=1,
        special2_escapes=1,
        knight_heal_noops=1,
        knight_heals=2,
        knight_attack_attempts=4,
        knight_attack_hits=3,
    )
    summary = summarize_episodes([episode])
    assert summary.average_invalid_knight_moves == 2
    assert summary.average_special2_hits == 2
    assert summary.average_special2_blocks == 1
    assert summary.average_special2_escapes == 1
    assert summary.average_knight_heal_noops == 1
    assert summary.average_knight_attack_attempts == 4
    assert summary.average_knight_attack_hits == 3
    assert summary.special2_hit_rate == pytest.approx(2 / 3)
    assert summary.special2_block_rate == pytest.approx(1 / 3)
    assert summary.invalid_knight_moves_per_step == pytest.approx(0.1)


def test_evaluator_reports_outcomes_damage_and_timeout_separately():
    config = GameConfig(
        knight_position=(4, 4),
        boss_position=(4, 5),
        max_steps=10,
    )
    knight = HeuristicAgent(seed=1)
    boss = HeuristicAgent(seed=2)
    episodes, summary = evaluate_policies(
        {"knight": knight, "boss": boss},
        config,
        episodes=3,
        seed=7,
        alpha_h=0.05,
        alpha_d=1.0,
        deterministic=True,
    )
    assert len(episodes) == 3
    assert summary.knight_wins == 3
    assert summary.knight_win_rate == 1.0
    assert summary.knight_win_ci_lower < 1.0
    assert summary.boss_wins == 0
    assert summary.timeouts == 0
    assert summary.average_knight_damage_dealt > 0
    assert all(not episode.timeout for episode in episodes)


def test_evaluation_at_timeout_records_draw_and_truncation():
    config = GameConfig(
        knight_position=(0, 0),
        boss_position=(8, 8),
        max_steps=1,
    )
    _, summary = evaluate_policies(
        {
            "knight": HeuristicAgent(),
            "boss": HeuristicAgent(),
        },
        config,
        episodes=2,
        seed=4,
        deterministic=True,
    )
    assert summary.draws == 2
    assert summary.timeouts == 2
    assert summary.timeout_rate == 1.0


def test_evaluation_does_not_update_or_mutate_saved_ppo_policy():
    config = GameConfig(
        knight_position=(0, 0), boss_position=(8, 8), max_steps=2
    )
    env = KnightBossParallelEnv(config)
    observation, _ = env.reset(seed=2)
    ppo_config = PPOConfig(
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
    )
    policy = PPOAgent(
        observation_to_vector(observation["knight"]).size,
        ppo_config,
        seed=3,
    )
    before = {
        key: value.clone() for key, value in policy.policy.state_dict().items()
    }
    episodes, _ = evaluate_policies(
        {"knight": policy, "boss": HeuristicAgent()},
        config,
        episodes=2,
        seed=9,
    )
    assert len(episodes) == 2
    assert policy.update_count == 0
    assert all(
        value.equal(policy.policy.state_dict()[key])
        for key, value in before.items()
    )


def test_threshold_reports_empirical_grid_and_nonmonotone_neighbors():
    records = []
    rates = {0.5: 70, 1.0: 40, 1.5: 60}
    for seed in (2, 4, 7):
        for alpha_h, wins in rates.items():
            records.append(
                {
                    "training_regime": "simultaneous_learning",
                    "training_seed": seed,
                    "evaluation_alpha_h": alpha_h,
                    "evaluation_alpha_d": 1.0,
                    "fixed_parameter": 1.0,
                    "varied_axis": "alpha_h",
                    "knight_wins": wins,
                    "evaluation_episodes": 100,
                }
            )
    [estimate] = estimate_threshold_curves(records)
    assert estimate["threshold_estimate"] == 1.5
    assert estimate["criterion"] == 0.5
    assert not estimate["neighboring_results_monotone"]
    assert estimate["threshold_ci_lower_grid"] <= estimate["threshold_estimate"]
    assert estimate["threshold_ci_upper_grid"] >= estimate["threshold_estimate"]
    assert [row["parameter"] for row in estimate["tested_parameters"]] == [
        0.5,
        1.0,
        1.5,
    ]
