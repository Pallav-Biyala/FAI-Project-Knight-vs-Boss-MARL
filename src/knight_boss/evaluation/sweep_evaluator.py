"""Batch independent evaluation across parameter-sweep training manifests."""

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from knight_boss.agents.heuristic_agent import HeuristicAgent
from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.evaluation.evaluator import evaluate_policies
from knight_boss.evaluation.io import evaluation_records, save_evaluation
from knight_boss.evaluation.thresholds import estimate_threshold_curves
from knight_boss.game.config import GameConfig
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.training.checkpointing import load_checkpoint
from knight_boss.training.config import PPOConfig


CHECKPOINT_POLICIES = ("auto", "final", "best")
OPPONENT_MODES = ("heuristic", "trained_boss", "matched")


@dataclass(frozen=True)
class SweepEvalTask:
    run_id: str
    run_directory: str
    knight_checkpoint: str
    boss_checkpoint: str | None
    opponent_mode: str
    cell_targets: tuple[tuple[float, float, int], ...]
    episodes: int
    deterministic: bool
    confidence: float
    game_config: GameConfig
    save_episodes: bool


def resolve_knight_checkpoint(run_directory: str | Path, policy: str = "auto") -> Path:
    """Resolve the Knight checkpoint file for a completed training run."""
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(f"Unsupported checkpoint policy {policy!r}.")
    checkpoints_dir = Path(run_directory) / "checkpoints"
    best_path = checkpoints_dir / "knight-best.pt"
    final_path = checkpoints_dir / "knight-final.pt"
    if policy == "best":
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {best_path}.")
        return best_path
    if policy == "final":
        if not final_path.exists():
            raise FileNotFoundError(f"Final checkpoint not found at {final_path}.")
        return final_path
    if best_path.exists():
        return best_path
    if final_path.exists():
        return final_path
    raise FileNotFoundError(f"Neither knight-best.pt nor knight-final.pt exists in {checkpoints_dir}.")


def resolve_boss_checkpoint(run_directory: str | Path, opponent_mode: str) -> Path | None:
    """Resolve the Boss checkpoint if required by opponent_mode."""
    if opponent_mode not in OPPONENT_MODES:
        raise ValueError(f"Unsupported opponent mode {opponent_mode!r}.")
    if opponent_mode == "heuristic":
        return None
    boss_final = Path(run_directory) / "checkpoints" / "boss-final.pt"
    if boss_final.exists():
        return boss_final
    if opponent_mode == "trained_boss":
        raise FileNotFoundError(f"Trained boss checkpoint not found at {boss_final}.")
    return None


def _ppo_config_from_metadata(metadata: dict[str, Any]) -> PPOConfig:
    raw = metadata["ppo_config"]
    return PPOConfig(
        learning_rate=float(raw["learning_rate"]),
        gamma=float(raw["gamma"]),
        gae_lambda=float(raw["gae_lambda"]),
        clip_range=float(raw["clip_range"]),
        entropy_coef=float(raw["entropy_coef"]),
        value_coef=float(raw["value_coef"]),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        rollout_steps=int(raw["rollout_steps"]),
        hidden_sizes=tuple(int(size) for size in raw["hidden_sizes"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        device=str(raw.get("device", "cpu")),
    )


def _observation_size(game_config: GameConfig) -> int:
    env = KnightBossParallelEnv(game_config)
    observation, _ = env.reset(seed=0)
    return observation_to_vector(observation["knight"]).size


def _evaluate_task_batch(
    task: SweepEvalTask,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch.set_num_threads(1)
    run_dir = Path(task.run_directory)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    ppo_config = _ppo_config_from_metadata(metadata)
    obs_size = _observation_size(task.game_config)

    knight = PPOAgent(obs_size, ppo_config, seed=task.cell_targets[0][2])
    knight_meta = load_checkpoint(knight, task.knight_checkpoint, load_optimizer=False)

    boss_meta: dict[str, Any] = {}
    if task.boss_checkpoint is not None:
        boss = PPOAgent(obs_size, ppo_config, seed=task.cell_targets[0][2] + 1)
        boss_meta = load_checkpoint(boss, task.boss_checkpoint, load_optimizer=False)
        boss_kind = "ppo_trained"
    else:
        boss = HeuristicAgent(seed=task.cell_targets[0][2] + 1)
        boss_kind = "heuristic"

    run_cfg = metadata.get("run_config", {})
    summary_records: list[dict[str, Any]] = []
    all_episode_records: list[dict[str, Any]] = []
    for alpha_h, alpha_d, eval_seed in task.cell_targets:
        episodes_result, summary = evaluate_policies(
            {"knight": knight, "boss": boss},
            task.game_config,
            episodes=task.episodes,
            seed=eval_seed,
            alpha_h=alpha_h,
            alpha_d=alpha_d,
            deterministic=task.deterministic,
            confidence=task.confidence,
        )
        summary_record, episode_records = evaluation_records(
            summary,
            episodes_result,
            training_metadata=knight_meta,
            evaluation_alpha_h=alpha_h,
            evaluation_alpha_d=alpha_d,
            evaluation_seed=eval_seed,
            deterministic=task.deterministic,
        )
        summary_record["run_id"] = task.run_id
        summary_record["reward_mode"] = run_cfg.get("reward_mode", "terminal")
        summary_record["shaping_weight"] = run_cfg.get("shaping_weight")
        summary_record["knight_checkpoint_name"] = Path(task.knight_checkpoint).name
        summary_record["boss_opponent_type"] = boss_kind
        summary_record["boss_training_regime"] = boss_meta.get("regime")
        summary_record["boss_training_seed"] = boss_meta.get("seed")
        summary_records.append(summary_record)
        if task.save_episodes:
            for record in episode_records:
                record["run_id"] = task.run_id
                record["boss_opponent_type"] = boss_kind
            all_episode_records.extend(episode_records)
    return summary_records, all_episode_records


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    """Read completed run records from a training_manifest.csv file."""
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training manifest {path} is empty.")
    return rows


def evaluate_sweep_manifest(
    manifest_path: str | Path,
    game_config: GameConfig,
    *,
    episodes: int = 200,
    base_seed: int = 100000,
    deterministic: bool = False,
    confidence: float = 0.95,
    checkpoint_policy: str = "auto",
    opponent_mode: str = "heuristic",
    grid_pairs: Sequence[tuple[float, float]] | None = None,
    output_root: str | Path = "results",
    max_workers: int = 1,
    save_episodes: bool = True,
) -> dict[str, Any]:
    """Independently evaluate every run in a sweep manifest."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")
    rows = load_manifest_rows(manifest_path)

    tasks: list[SweepEvalTask] = []
    chunk_size = 11
    for row in rows:
        run_dir = Path(row["training_directory"])
        knight_ckpt = resolve_knight_checkpoint(run_dir, policy=checkpoint_policy)
        boss_ckpt = resolve_boss_checkpoint(run_dir, opponent_mode=opponent_mode)
        target_pairs = (
            list(grid_pairs)
            if grid_pairs is not None
            else [(float(row["training_alpha_h"]), float(row["training_alpha_d"]))]
        )
        targets = tuple(
            (float(alpha_h), float(alpha_d), base_seed + pair_idx * episodes)
            for pair_idx, (alpha_h, alpha_d) in enumerate(target_pairs)
        )
        for start in range(0, len(targets), chunk_size):
            tasks.append(
                SweepEvalTask(
                    run_id=row["run_id"],
                    run_directory=str(run_dir),
                    knight_checkpoint=str(knight_ckpt),
                    boss_checkpoint=str(boss_ckpt) if boss_ckpt is not None else None,
                    opponent_mode=opponent_mode,
                    cell_targets=targets[start : start + chunk_size],
                    episodes=episodes,
                    deterministic=deterministic,
                    confidence=confidence,
                    game_config=game_config,
                    save_episodes=save_episodes,
                )
            )

    summaries: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    if max_workers == 1 or len(tasks) == 1:
        results_iter = (_evaluate_task_batch(task) for task in tasks)
        for batch_summaries, batch_episodes in results_iter:
            summaries.extend(batch_summaries)
            all_episodes.extend(batch_episodes)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for batch_summaries, batch_episodes in pool.map(_evaluate_task_batch, tasks):
                summaries.extend(batch_summaries)
                all_episodes.extend(batch_episodes)

    thresholds = estimate_threshold_curves(summaries)
    eval_metadata = {
        "training_manifest": str(Path(manifest_path).resolve()),
        "checkpoint_policy": checkpoint_policy,
        "opponent_mode": opponent_mode,
        "evaluation_mode": "grid" if grid_pairs is not None else "matched",
        "grid_pair_count": len(grid_pairs) if grid_pairs is not None else 1,
        "episodes_per_evaluation": episodes,
        "total_summaries": len(summaries),
        "total_episodes_played": len(summaries) * episodes,
        "episodes_csv_saved": save_episodes,
        "base_seed": base_seed,
        "deterministic": deterministic,
        "confidence_level": confidence,
        "thresholds": thresholds,
    }
    output_dir = save_evaluation(
        summaries,
        all_episodes,
        output_root=output_root,
        metadata=eval_metadata,
    )
    return {
        "evaluation_directory": str(output_dir),
        "summaries_csv": str(output_dir / "summaries.csv"),
        "episodes_csv": str(output_dir / "episodes.csv") if save_episodes else None,
        "total_summaries": len(summaries),
        "thresholds": thresholds,
    }
