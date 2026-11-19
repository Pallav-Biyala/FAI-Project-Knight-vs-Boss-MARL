import csv

from knight_boss.visualization.plots import (
    plot_evaluation_analysis,
    plot_training_analysis,
    plot_validation_analysis,
)


def test_training_and_evaluation_plots_use_saved_csv_data(tmp_path):
    training_csv = tmp_path / "training.csv"
    training_fields = [
        "run_id",
        "training_regime",
        "training_seed",
        "training_alpha_h",
        "training_alpha_d",
        "agent",
        "environment_steps",
        "rollout_reward",
        "knight_win_rate",
        "boss_win_rate",
        "average_episode_length",
    ]
    with training_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=training_fields)
        writer.writeheader()
        for regime in ("fixed_opponent", "simultaneous_learning", "alternating_learning"):
            for seed in (1, 2):
                writer.writerow(
                    {
                        "run_id": f"{regime}-{seed}",
                        "training_regime": regime,
                        "training_seed": seed,
                        "training_alpha_h": 1.0,
                        "training_alpha_d": 1.0,
                        "agent": "knight",
                        "environment_steps": 10,
                        "rollout_reward": 2.0,
                        "knight_win_rate": 0.6,
                        "boss_win_rate": 0.3,
                        "average_episode_length": 25,
                    }
                )

    evaluation_csv = tmp_path / "evaluation.csv"
    evaluation_fields = [
        "training_regime",
        "training_seed",
        "evaluation_alpha_h",
        "evaluation_alpha_d",
        "knight_wins",
        "evaluation_episodes",
        "knight_win_rate",
    ]
    with evaluation_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=evaluation_fields)
        writer.writeheader()
        for regime in ("fixed_opponent", "simultaneous_learning", "alternating_learning"):
            for seed in (1, 2):
                for alpha_h in (0.5, 1.0, 1.5):
                    for alpha_d in (0.5, 1.0, 1.5):
                        wins = 8 if alpha_h + alpha_d <= 2.0 else 4
                        writer.writerow(
                            {
                                "training_regime": regime,
                                "training_seed": seed,
                                "evaluation_alpha_h": alpha_h,
                                "evaluation_alpha_d": alpha_d,
                                "knight_wins": wins,
                                "evaluation_episodes": 10,
                                "knight_win_rate": wins / 10,
                            }
                        )

    training_outputs = plot_training_analysis(
        [training_csv], tmp_path / "plots" / "training"
    )
    evaluation_outputs = plot_evaluation_analysis(
        [evaluation_csv], tmp_path / "plots" / "evaluation"
    )
    assert len(training_outputs) == 4
    assert len(evaluation_outputs) >= 8
    assert all(path.is_file() for path in training_outputs + evaluation_outputs)
    assert (tmp_path / "plots" / "evaluation" / "threshold_estimates.csv").is_file()


def test_validation_plot_uses_saved_checkpoint_validation_metrics(tmp_path):
    validation_csv = tmp_path / "validation.csv"
    fields = [
        "run_id",
        "training_regime",
        "training_seed",
        "training_alpha_h",
        "training_alpha_d",
        "validation_alpha_h",
        "validation_alpha_d",
        "environment_steps",
        "knight_win_rate",
        "knight_win_ci_lower",
        "knight_win_ci_upper",
    ]
    with validation_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "run-1",
                "training_regime": "fixed_opponent",
                "training_seed": 1,
                "training_alpha_h": 1.0,
                "training_alpha_d": 1.0,
                "validation_alpha_h": 1.0,
                "validation_alpha_d": 1.0,
                "environment_steps": 10,
                "knight_win_rate": 0.6,
                "knight_win_ci_lower": 0.3,
                "knight_win_ci_upper": 0.8,
            }
        )
    outputs = plot_validation_analysis(
        [validation_csv], tmp_path / "plots" / "validation"
    )
    assert len(outputs) == 1
    assert outputs[0].is_file()


def test_evaluation_plots_handle_single_point_and_empty_thresholds(tmp_path):
    import warnings

    evaluation_csv = tmp_path / "single_eval.csv"
    fields = [
        "training_regime",
        "training_seed",
        "evaluation_alpha_h",
        "evaluation_alpha_d",
        "knight_wins",
        "evaluation_episodes",
        "knight_win_rate",
    ]
    with evaluation_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "training_regime": "fixed_opponent",
                "training_seed": 0,
                "evaluation_alpha_h": 2.0,
                "evaluation_alpha_d": 2.0,
                "knight_wins": 1,
                "evaluation_episodes": 10,
                "knight_win_rate": 0.1,
            }
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outputs = plot_evaluation_analysis(
            [evaluation_csv], tmp_path / "plots" / "single"
        )
    assert len(outputs) == 7
    assert all(path.is_file() for path in outputs)
