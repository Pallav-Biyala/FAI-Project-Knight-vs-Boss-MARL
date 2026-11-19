"""Persistence helpers for evaluation summaries and episode-level records."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from knight_boss.evaluation.metrics import EpisodeMetrics, EvaluationSummary


def save_evaluation(
    summaries: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    output_root: str | Path,
    metadata: dict[str, Any],
) -> Path:
    directory = Path(output_root) / f"evaluation-{uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    if summaries:
        _write_csv(directory / "summaries.csv", summaries)
    if episodes:
        _write_csv(directory / "episodes.csv", episodes)
    with (directory / "evaluation.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    return directory


def evaluation_records(
    summary: EvaluationSummary,
    episodes: list[EpisodeMetrics],
    *,
    training_metadata: dict[str, Any],
    evaluation_alpha_h: float,
    evaluation_alpha_d: float,
    evaluation_seed: int,
    deterministic: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_record = {
        "training_regime": training_metadata.get("regime", "unknown"),
        "training_seed": training_metadata.get("seed"),
        "training_alpha_h": training_metadata.get("alpha_h"),
        "training_alpha_d": training_metadata.get("alpha_d"),
        "evaluation_alpha_h": evaluation_alpha_h,
        "evaluation_alpha_d": evaluation_alpha_d,
        "evaluation_seed": evaluation_seed,
        "evaluation_episodes": summary.episodes,
        "deterministic_policy": deterministic,
        **summary.to_dict(),
    }
    episode_records = [
        {
            "training_regime": training_metadata.get("regime", "unknown"),
            "training_seed": training_metadata.get("seed"),
            "training_alpha_h": training_metadata.get("alpha_h"),
            "training_alpha_d": training_metadata.get("alpha_d"),
            "evaluation_alpha_h": evaluation_alpha_h,
            "evaluation_alpha_d": evaluation_alpha_d,
            **asdict(episode),
        }
        for episode in episodes
    ]
    return summary_record, episode_records


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
