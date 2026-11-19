"""Empirical grid threshold estimates without monotonicity assumptions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from knight_boss.evaluation.metrics import wilson_interval
from knight_boss.evaluation.statistics import mean_ci_across_seeds


def estimate_threshold_curves(
    records: Sequence[Mapping[str, Any]],
    *,
    criterion: float = 0.5,
) -> list[dict[str, Any]]:
    """Estimate maximum tested successes and seed-uncertainty brackets.

    Records must be one aggregate per trained checkpoint and evaluation
    configuration. Repeated evaluation runs for the same checkpoint/grid cell
    are pooled by episode counts before comparing independent training seeds.
    """
    if not 0 <= criterion <= 1:
        raise ValueError("criterion must be between zero and one.")
    grouped: dict[tuple[str, str, float], dict[int, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        regime = str(record["training_regime"])
        varied_axis = str(record.get("varied_axis", "alpha_h"))
        if varied_axis not in {"alpha_h", "alpha_d"}:
            raise ValueError(f"Unknown varied axis: {varied_axis}")
        fixed_value = float(
            record.get(
                "fixed_parameter",
                record["evaluation_alpha_d"]
                if varied_axis == "alpha_h"
                else record["evaluation_alpha_h"],
            )
        )
        grouped[(regime, varied_axis, fixed_value)][int(record["training_seed"])].append(
            record
        )

    estimates: list[dict[str, Any]] = []
    for (regime, varied_axis, fixed_value), seed_records in sorted(grouped.items()):
        by_parameter: dict[float, dict[int, float]] = defaultdict(dict)
        for training_seed, rows in seed_records.items():
            seed_parameter_data: dict[float, list[tuple[int, int]]] = defaultdict(list)
            for row in rows:
                alpha = float(
                    row["evaluation_alpha_h"]
                    if varied_axis == "alpha_h"
                    else row["evaluation_alpha_d"]
                )
                wins = int(row["knight_wins"])
                count = int(row["evaluation_episodes"])
                seed_parameter_data[alpha].append((wins, count))
            for alpha, observations in seed_parameter_data.items():
                total_wins = sum(wins for wins, _ in observations)
                total_episodes = sum(count for _, count in observations)
                by_parameter[alpha][training_seed] = total_wins / total_episodes

        parameter_rows = []
        for parameter, seed_rates in sorted(by_parameter.items()):
            seed_statistics = mean_ci_across_seeds(seed_rates.values())
            successful = seed_statistics["mean"] >= criterion
            lower = seed_statistics["ci_lower"]
            upper = seed_statistics["ci_upper"]
            matching_rows = [
                row
                for seed, rows in seed_records.items()
                if seed in seed_rates
                for row in rows
                if float(
                    row["evaluation_alpha_h"]
                    if varied_axis == "alpha_h"
                    else row["evaluation_alpha_d"]
                ) == parameter
            ]
            total_wins = sum(int(row["knight_wins"]) for row in matching_rows)
            total_games = sum(
                int(row["evaluation_episodes"]) for row in matching_rows
            )
            binomial_lower, binomial_upper = wilson_interval(total_wins, total_games)
            lower = (
                binomial_lower
                if lower is None
                else max(0.0, min(lower, binomial_lower))
            )
            upper = (
                binomial_upper
                if upper is None
                else min(1.0, max(upper, binomial_upper))
            )
            parameter_rows.append(
                {
                    "parameter": parameter,
                    "mean_knight_win_rate": seed_statistics["mean"],
                    "standard_deviation_across_seeds": seed_statistics[
                        "standard_deviation"
                    ],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_training_seeds": seed_statistics["n_training_seeds"],
                    "success": successful,
                }
            )

        successful_parameters = [
            row["parameter"] for row in parameter_rows if row["success"]
        ]
        confirmed_successes = [
            row["parameter"]
            for row in parameter_rows
            if row["ci_lower"] is not None and row["ci_lower"] >= criterion
        ]
        possible_successes = [
            row["parameter"]
            for row in parameter_rows
            if row["ci_upper"] is not None and row["ci_upper"] >= criterion
        ]
        statuses = [row["success"] for row in parameter_rows]
        monotone = not any(
            statuses[index] is False and statuses[index + 1] is True
            for index in range(len(statuses) - 1)
        )
        estimates.append(
            {
                "training_regime": regime,
                "varied_axis": varied_axis,
                "fixed_parameter": fixed_value,
                "criterion": criterion,
                "threshold_estimate": (
                    max(successful_parameters) if successful_parameters else None
                ),
                "threshold_ci_lower_grid": (
                    max(confirmed_successes) if confirmed_successes else None
                ),
                "threshold_ci_upper_grid": (
                    max(possible_successes) if possible_successes else None
                ),
                "neighboring_results_monotone": monotone,
                "tested_parameters": parameter_rows,
            }
        )
    return estimates
