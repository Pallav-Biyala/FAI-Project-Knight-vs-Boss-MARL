"""Training, capability-grid, regime-comparison, and threshold plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from knight_boss.evaluation.statistics import mean_ci_across_seeds
from knight_boss.evaluation.thresholds import estimate_threshold_curves


def plot_training_analysis(
    csv_paths: list[str | Path], output_directory: str | Path
) -> list[Path]:
    if not csv_paths:
        raise ValueError("At least one training CSV is required.")
    frames = [pd.read_csv(path) for path in csv_paths]
    data = pd.concat(frames, ignore_index=True)
    required = {
        "training_regime",
        "agent",
        "environment_steps",
        "rollout_reward",
        "knight_win_rate",
        "boss_win_rate",
        "average_episode_length",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    plots = (
        ("rollout_reward", "Rollout reward sum", "rollout_reward.png"),
        ("knight_win_rate", "Knight win rate per rollout", "knight_win_rate.png"),
        ("boss_win_rate", "Boss win rate per rollout", "boss_win_rate.png"),
        (
            "average_episode_length",
            "Average episode length per rollout (steps)",
            "episode_length.png",
        ),
    )
    group_columns = [
        column
        for column in (
            "training_regime",
            "agent",
            "training_alpha_h",
            "training_alpha_d",
        )
        if column in data.columns
    ]
    for metric, label, filename in plots:
        figure, axis = plt.subplots(figsize=(8, 5))
        plotted = False
        for group_key, group in data.groupby(group_columns, dropna=False):
            curve = (
                group.groupby("environment_steps", as_index=False)[metric]
                .mean()
                .dropna()
                .sort_values("environment_steps")
            )
            if curve.empty:
                continue
            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            label_text = ", ".join(
                f"{name}={value}"
                for name, value in zip(group_columns, key_values)
                if pd.notna(value)
            )
            axis.plot(
                curve["environment_steps"],
                curve[metric],
                label=label_text,
                linewidth=1.5,
            )
            plotted = True
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
        if plotted:
            axis.legend(fontsize="small", loc="best")
        figure.tight_layout()
        output = destination / filename
        figure.savefig(output, dpi=160)
        plt.close(figure)
        outputs.append(output)
    return outputs


def plot_validation_analysis(
    csv_paths: list[str | Path], output_directory: str | Path
) -> list[Path]:
    if not csv_paths:
        raise ValueError("At least one validation CSV is required.")
    data = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    required = {
        "run_id",
        "training_regime",
        "training_alpha_h",
        "training_alpha_d",
        "environment_steps",
        "validation_alpha_h",
        "validation_alpha_d",
        "knight_win_rate",
        "knight_win_ci_lower",
        "knight_win_ci_upper",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Validation CSV is missing columns: {sorted(missing)}")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for (run_id, regime, alpha_h, alpha_d), group in data.groupby(
        ["run_id", "training_regime", "training_alpha_h", "training_alpha_d"]
    ):
        group = group.sort_values("environment_steps")
        lower = group["knight_win_ci_lower"].to_numpy(dtype=float)
        upper = group["knight_win_ci_upper"].to_numpy(dtype=float)
        means = group["knight_win_rate"].to_numpy(dtype=float)
        yerr = np.vstack((means - lower, upper - means))
        axis.errorbar(
            group["environment_steps"],
            means,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=f"{regime} train ({alpha_h:g}, {alpha_d:g}) seed {run_id}",
        )
    axis.set_xlabel("Training environment steps")
    axis.set_ylabel("Validation Knight win rate")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Validation performance (separate from training and final evaluation)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="x-small", loc="best")
    figure.tight_layout()
    output = destination / "validation_knight_win_rate.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return [output]


def plot_evaluation_analysis(
    csv_paths: list[str | Path], output_directory: str | Path
) -> list[Path]:
    if not csv_paths:
        raise ValueError("At least one evaluation summary CSV is required.")
    data = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    required = {
        "training_regime",
        "training_seed",
        "evaluation_alpha_h",
        "evaluation_alpha_d",
        "knight_wins",
        "evaluation_episodes",
        "knight_win_rate",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    summaries = _aggregate_training_seeds(data)
    outputs: list[Path] = []

    for regime, regime_data in summaries.groupby("training_regime"):
        pivot = regime_data.pivot(
            index="evaluation_alpha_h",
            columns="evaluation_alpha_d",
            values="mean_knight_win_rate",
        )
        figure, axis = plt.subplots(figsize=(8, 6))
        xs = pivot.columns.to_numpy(dtype=float)
        ys = pivot.index.to_numpy(dtype=float)
        values = pivot.to_numpy(dtype=float)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.set_xticks(np.arange(len(xs)))
        axis.set_xticklabels([f"{value:.2f}" for value in xs])
        axis.set_yticks(np.arange(len(ys)))
        axis.set_yticklabels([f"{value:.2f}" for value in ys])
        if len(xs) <= 8 and len(ys) <= 8:
            for row_idx in range(len(ys)):
                for col_idx in range(len(xs)):
                    val = values[row_idx, col_idx]
                    if np.isfinite(val):
                        text_color = "black" if val >= 0.6 else "white"
                        axis.text(
                            col_idx,
                            row_idx,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                        )
        axis.set_xlabel("Boss damage multiplier (Alpha_D)")
        axis.set_ylabel("Boss health multiplier (Alpha_H)")
        axis.set_title(f"Knight win rate — {regime}")
        figure.colorbar(image, ax=axis, label="Mean Knight win rate across seeds")
        figure.tight_layout()
        output = destination / f"heatmap-{regime}.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        outputs.append(output)

    for axis_name, fixed_value in (
        ("alpha_h", _nearest_value(summaries["evaluation_alpha_h"], 1.0)),
        ("alpha_d", _nearest_value(summaries["evaluation_alpha_d"], 1.0)),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        varied_column = f"evaluation_{axis_name}"
        fixed_column = "evaluation_alpha_d" if axis_name == "alpha_h" else "evaluation_alpha_h"
        fixed_label = "Alpha_D" if axis_name == "alpha_h" else "Alpha_H"
        slice_data = summaries[
            np.isclose(summaries[fixed_column], fixed_value)
        ]
        for regime, regime_data in slice_data.groupby("training_regime"):
            regime_data = regime_data.sort_values(varied_column)
            axis.errorbar(
                regime_data[varied_column],
                regime_data["mean_knight_win_rate"],
                yerr=_confidence_errors(regime_data),
                marker="o",
                capsize=3,
                label=regime,
            )
        axis.set_xlabel(
            "Boss health multiplier (Alpha_H)"
            if axis_name == "alpha_h"
            else "Boss damage multiplier (Alpha_D)"
        )
        axis.set_ylabel("Knight win rate")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(
            f"Regime comparison at {fixed_label}={fixed_value:g}"
        )
        axis.grid(True, alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        figure.tight_layout()
        output = destination / f"regime-comparison-vs-{axis_name}.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        outputs.append(output)

    threshold_records = []
    for axis_name in ("alpha_h", "alpha_d"):
        for record in data.to_dict(orient="records"):
            threshold_records.append(
                {
                    **record,
                    "varied_axis": axis_name,
                    "fixed_parameter": (
                        record["evaluation_alpha_d"]
                        if axis_name == "alpha_h"
                        else record["evaluation_alpha_h"]
                    ),
                }
            )
    thresholds = estimate_threshold_curves(threshold_records)
    threshold_json = destination / "threshold_estimates.json"
    with threshold_json.open("w", encoding="utf-8") as output_file:
        json.dump(thresholds, output_file, indent=2)
    threshold_rows = []
    for estimate in thresholds:
        threshold_rows.append(
            {
                key: value
                for key, value in estimate.items()
                if key != "tested_parameters"
            }
        )
    _write_csv(destination / "threshold_estimates.csv", threshold_rows)
    outputs.extend((threshold_json, destination / "threshold_estimates.csv"))

    for axis_name in ("alpha_h", "alpha_d"):
        figure, axis = plt.subplots(figsize=(8, 5))
        for regime in sorted({item["training_regime"] for item in thresholds}):
            curve = [
                item
                for item in thresholds
                if item["training_regime"] == regime
                and item["varied_axis"] == axis_name
                and item["threshold_estimate"] is not None
            ]
            curve.sort(key=lambda item: item["fixed_parameter"])
            if curve:
                axis.plot(
                    [item["fixed_parameter"] for item in curve],
                    [item["threshold_estimate"] for item in curve],
                    marker="o",
                    label=regime,
                )
        axis.set_xlabel(
            "Fixed Boss damage multiplier (Alpha_D)"
            if axis_name == "alpha_h"
            else "Fixed Boss health multiplier (Alpha_H)"
        )
        axis.set_ylabel(
            "Maximum tested successful Alpha_H"
            if axis_name == "alpha_h"
            else "Maximum tested successful Alpha_D"
        )
        axis.set_title(
            "Empirical grid threshold; success criterion: Knight win rate ≥ 0.50"
        )
        axis.grid(True, alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        figure.tight_layout()
        output = destination / f"threshold-vs-{axis_name}.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        outputs.append(output)

    return outputs


def _aggregate_training_seeds(data: pd.DataFrame) -> pd.DataFrame:
    config_columns = [
        "training_regime",
        "evaluation_alpha_h",
        "evaluation_alpha_d",
    ]
    per_seed = (
        data.groupby(config_columns + ["training_seed"], dropna=False, as_index=False)
        .agg(knight_wins=("knight_wins", "sum"), episodes=("evaluation_episodes", "sum"))
    )
    per_seed["seed_win_rate"] = per_seed["knight_wins"] / per_seed["episodes"]
    rows = []
    for key, group in per_seed.groupby(config_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        interval = mean_ci_across_seeds(group["seed_win_rate"].tolist())
        rows.append(
            {
                **dict(zip(config_columns, key_values)),
                **{
                    "mean_knight_win_rate": interval["mean"],
                    "standard_deviation": interval["standard_deviation"],
                    "ci_lower": interval["ci_lower"],
                    "ci_upper": interval["ci_upper"],
                    "n_training_seeds": interval["n_training_seeds"],
                },
            }
        )
    return pd.DataFrame(rows)


def _confidence_errors(data: pd.DataFrame) -> np.ndarray:
    means = data["mean_knight_win_rate"].to_numpy(dtype=float)
    lower = data["ci_lower"].to_numpy(dtype=float)
    upper = data["ci_upper"].to_numpy(dtype=float)
    errors = np.zeros((2, len(data)), dtype=float)
    available = np.isfinite(lower) & np.isfinite(upper)
    errors[0, available] = means[available] - lower[available]
    errors[1, available] = upper[available] - means[available]
    return errors


def _nearest_value(values: pd.Series, target: float) -> float:
    unique = values.dropna().unique()
    if len(unique) == 0:
        raise ValueError("No evaluation parameter values are available.")
    return float(min(unique, key=lambda value: abs(float(value) - target)))


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
