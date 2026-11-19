"""Training-regime configuration and run orchestration."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from knight_boss.agents import HeuristicAgent
from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.evaluation.evaluator import evaluate_policies
from knight_boss.game.config import GameConfig
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.marl.rewards import REWARD_MODES
from knight_boss.training.checkpointing import load_checkpoint, save_checkpoint
from knight_boss.training.config import PPOConfig
from knight_boss.training.trainer import MultiAgentPPOTrainer


REGIMES = (
    "fixed_opponent",
    "simultaneous_learning",
    "alternating_learning",
)


@dataclass(frozen=True)
class TrainingRunConfig:
    regime: str
    training_episodes: int
    checkpoint_interval_updates: int
    validation_interval_updates: int
    validation_episodes: int
    validation_seed: int
    validation_alpha_h: float
    validation_alpha_d: float
    output_root: str
    opponent: str | None = None
    phase_episodes: int | None = None
    validation_deterministic: bool = False
    reward_mode: str = "terminal"
    shaping_weight: float = 0.5
    draw_penalty: float = 0.75
    # Optional environment-step budget. When set it replaces the episode
    # budget: training stops after `training_steps` environment steps, and
    # alternating phases last `phase_steps` steps (proposal Sec. 4.3 defines N
    # in environment steps). Episodes are short (~25 steps) when the Knight
    # loses quickly, so an episode budget gives very few PPO updates.
    training_steps: int | None = None
    phase_steps: int | None = None

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"Unknown training regime: {self.regime}")
        if self.training_episodes < 1:
            raise ValueError("training_episodes must be positive.")
        if self.checkpoint_interval_updates < 1:
            raise ValueError("checkpoint_interval_updates must be positive.")
        if self.validation_interval_updates < 1 or self.validation_episodes < 1:
            raise ValueError("Validation interval and episode count must be positive.")
        if self.validation_seed < 0:
            raise ValueError("validation_seed must be non-negative.")
        if self.validation_alpha_h <= 0 or self.validation_alpha_d <= 0:
            raise ValueError("Validation capability multipliers must be positive.")
        if not self.output_root:
            raise ValueError("output_root must be specified.")
        if self.regime == "fixed_opponent" and self.opponent != "heuristic":
            raise ValueError("fixed_opponent currently requires opponent: heuristic.")
        if self.training_steps is not None and self.training_steps < 1:
            raise ValueError("training_steps must be positive when set.")
        if self.phase_steps is not None and self.phase_steps < 1:
            raise ValueError("phase_steps must be positive when set.")
        if self.regime == "alternating_learning":
            if self.training_steps is not None:
                if self.phase_steps is None:
                    raise ValueError(
                        "alternating_learning with training_steps requires phase_steps."
                    )
            elif self.phase_episodes is None or self.phase_episodes < 1:
                raise ValueError(
                    "alternating_learning requires positive phase_episodes."
                )
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}.")
        if not isfinite(self.shaping_weight) or self.shaping_weight < 0:
            raise ValueError("shaping_weight must be non-negative and finite.")
        if not isfinite(self.draw_penalty) or self.draw_penalty < 0:
            raise ValueError("draw_penalty must be non-negative and finite.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingRunConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            data: Any = yaml.safe_load(config_file)
        if not isinstance(data, dict):
            raise ValueError("Training run configuration must be a mapping.")
        return cls(
            regime=str(data["regime"]),
            training_episodes=int(data["training_episodes"]),
            checkpoint_interval_updates=int(data["checkpoint_interval_updates"]),
            validation_interval_updates=int(data["validation_interval_updates"]),
            validation_episodes=int(data["validation_episodes"]),
            validation_seed=int(data["validation_seed"]),
            validation_alpha_h=float(data["validation_alpha_h"]),
            validation_alpha_d=float(data["validation_alpha_d"]),
            output_root=str(data["output_root"]),
            opponent=data.get("opponent"),
            phase_episodes=(
                int(data["phase_episodes"]) if "phase_episodes" in data else None
            ),
            validation_deterministic=bool(
                data.get("validation_deterministic", False)
            ),
            reward_mode=str(data.get("reward_mode", "terminal")),
            shaping_weight=float(data.get("shaping_weight", 0.5)),
            draw_penalty=float(data.get("draw_penalty", 0.75)),
            training_steps=(
                int(data["training_steps"]) if "training_steps" in data else None
            ),
            phase_steps=int(data["phase_steps"]) if "phase_steps" in data else None,
        )


def run_training(
    run_config: TrainingRunConfig,
    game_config: GameConfig,
    ppo_config: PPOConfig,
    *,
    seed: int,
    alpha_h: float | None = None,
    alpha_d: float | None = None,
    knight_checkpoint: str | Path | None = None,
    boss_checkpoint: str | Path | None = None,
    env_factory: Callable[[float | None, float | None], KnightBossParallelEnv] | None = None,
    heuristic_factory: Callable[[int], HeuristicAgent] | None = None,
) -> dict[str, Any]:
    env = (
        env_factory(alpha_h, alpha_d)
        if env_factory is not None
        else KnightBossParallelEnv(
            game_config,
            alpha_h=alpha_h,
            alpha_d=alpha_d,
            reward_mode=run_config.reward_mode,
            shaping_weight=run_config.shaping_weight,
            draw_penalty=run_config.draw_penalty,
        )
    )
    trainer = MultiAgentPPOTrainer(env, ppo_config, seed=seed)
    if run_config.regime == "fixed_opponent":
        trainer.policies["boss"] = (
            heuristic_factory(seed + 1)
            if heuristic_factory is not None
            else HeuristicAgent(
                rows=game_config.rows, columns=game_config.columns, seed=seed + 1
            )
        )

    resumed_from: dict[str, str] = {}
    for agent_name, ckpt_path in (
        ("knight", knight_checkpoint),
        ("boss", boss_checkpoint),
    ):
        if ckpt_path is not None:
            policy = trainer.policies[agent_name]
            if not isinstance(policy, PPOAgent):
                raise ValueError(
                    f"Cannot resume {agent_name} from checkpoint in regime "
                    f"{run_config.regime} because {agent_name} is not a PPOAgent."
                )
            load_checkpoint(policy, ckpt_path, load_optimizer=True)
            resumed_from[agent_name] = str(Path(ckpt_path).resolve())

    run_id = (
        f"{run_config.regime}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    run_directory = Path(run_config.output_root) / run_config.regime / run_id
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=False)
    log_path = run_directory / "training.csv"
    validation_path = run_directory / "validation.csv"
    metadata_path = run_directory / "metadata.json"
    metadata = {
        "run_id": run_id,
        "regime": run_config.regime,
        "seed": seed,
        "alpha_h": trainer.env.alpha_h,
        "alpha_d": trainer.env.alpha_d,
        "resumed_from": resumed_from,
        "game_config": asdict(game_config),
        "ppo_config": asdict(ppo_config),
        "run_config": asdict(run_config),
        "status": "running",
    }
    _write_json(metadata_path, metadata)

    fields = (
        "run_id",
        "training_regime",
        "training_seed",
        "training_alpha_h",
        "training_alpha_d",
        "update",
        "environment_steps",
        "completed_episodes",
        "episodes_in_rollout",
        "phase",
        "agent",
        "rollout_reward",
        "knight_win_rate",
        "boss_win_rate",
        "draw_rate",
        "timeout_rate",
        "average_episode_length",
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
    )
    validation_fields = (
        "run_id",
        "training_regime",
        "training_seed",
        "training_alpha_h",
        "training_alpha_d",
        "validation_alpha_h",
        "validation_alpha_d",
        "validation_seed",
        "checkpoint_updates",
        "environment_steps",
        "episodes",
        "confidence_level",
        "knight_wins",
        "boss_wins",
        "draws",
        "timeouts",
        "knight_win_rate",
        "boss_win_rate",
        "draw_rate",
        "timeout_rate",
        "knight_win_ci_lower",
        "knight_win_ci_upper",
        "boss_win_ci_lower",
        "boss_win_ci_upper",
        "draw_ci_lower",
        "draw_ci_upper",
        "average_episode_length",
        "average_knight_damage_dealt",
        "average_knight_damage_received",
        "average_boss_damage_dealt",
        "average_boss_damage_received",
    )
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fields)
        writer.writeheader()
        with validation_path.open("w", newline="", encoding="utf-8") as validation_file:
            validation_writer = csv.DictWriter(
                validation_file, fieldnames=validation_fields
            )
            validation_writer.writeheader()
            alternating_agent = "knight"
            phase = "knight"
            # Budget unit: environment steps when `training_steps` is set,
            # otherwise completed episodes (original behaviour).
            use_steps = run_config.training_steps is not None
            budget_total = (
                run_config.training_steps
                if use_steps
                else run_config.training_episodes
            )
            phase_length = (
                run_config.phase_steps if use_steps else run_config.phase_episodes
            ) or 0

            def progress() -> int:
                return (
                    trainer.total_environment_steps
                    if use_steps
                    else trainer.completed_episodes
                )

            phase_end = min(budget_total, progress() + phase_length)
            best_validation_win_rate = -1.0
            while progress() < budget_total:
                episodes_before = trainer.completed_episodes
                if run_config.regime == "alternating_learning":
                    trainable_agents = (alternating_agent,)
                    phase = alternating_agent
                    target = min(budget_total, phase_end)
                elif run_config.regime == "fixed_opponent":
                    trainable_agents = ("knight",)
                    phase = "knight_vs_heuristic"
                    target = budget_total
                else:
                    trainable_agents = ("knight", "boss")
                    phase = "simultaneous"
                    target = budget_total

                if use_steps:
                    metrics = trainer.train_rollout(
                        steps=min(
                            ppo_config.rollout_steps,
                            target - trainer.total_environment_steps,
                        ),
                        trainable_agents=trainable_agents,
                    )
                else:
                    metrics = trainer.train_rollout(
                        trainable_agents=trainable_agents,
                        stop_after_episodes=target,
                    )
                episode_summary = _summarize_training_episodes(
                    trainer.last_rollout_episodes
                )
                for agent in trainable_agents:
                    policy = trainer.policies[agent]
                    update = policy.update_count
                    if update % run_config.checkpoint_interval_updates == 0:
                        _save_run_checkpoint(
                            policy,
                            checkpoint_directory,
                            agent,
                            update,
                            metadata,
                        )
                    batch_reward = trainer.last_rollout_rewards[agent]
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "training_regime": run_config.regime,
                            "training_seed": seed,
                            "training_alpha_h": trainer.env.alpha_h,
                            "training_alpha_d": trainer.env.alpha_d,
                            "update": update,
                            "environment_steps": trainer.total_environment_steps,
                            "completed_episodes": trainer.completed_episodes,
                            "episodes_in_rollout": trainer.completed_episodes - episodes_before,
                            "phase": phase,
                            "agent": agent,
                            "rollout_reward": batch_reward,
                            **episode_summary,
                            **metrics[agent],
                        }
                    )
                log_file.flush()

                should_validate = (
                    progress() >= budget_total
                    or any(
                        trainer.policies[agent].update_count
                        % run_config.validation_interval_updates
                        == 0
                        for agent in trainable_agents
                    )
                )
                if should_validate:
                    validation_env_factory = (
                        (
                            lambda: env_factory(
                                run_config.validation_alpha_h,
                                run_config.validation_alpha_d,
                            )
                        )
                        if env_factory is not None
                        else None
                    )
                    _, validation_summary = evaluate_policies(
                        trainer.policies,
                        game_config,
                        episodes=run_config.validation_episodes,
                        seed=run_config.validation_seed,
                        alpha_h=run_config.validation_alpha_h,
                        alpha_d=run_config.validation_alpha_d,
                        deterministic=run_config.validation_deterministic,
                        env_factory=validation_env_factory,
                    )
                    validation_writer.writerow(
                        {
                            "run_id": run_id,
                            "training_regime": run_config.regime,
                            "training_seed": seed,
                            "training_alpha_h": trainer.env.alpha_h,
                            "training_alpha_d": trainer.env.alpha_d,
                            "validation_alpha_h": run_config.validation_alpha_h,
                            "validation_alpha_d": run_config.validation_alpha_d,
                            "validation_seed": run_config.validation_seed,
                            "checkpoint_updates": json.dumps(
                                {
                                    agent: getattr(
                                        trainer.policies[agent], "update_count", None
                                    )
                                    for agent in ("knight", "boss")
                                },
                                sort_keys=True,
                            ),
                            "environment_steps": trainer.total_environment_steps,
                            **validation_summary.to_dict(),
                        }
                    )
                    validation_file.flush()
                    if (
                        run_config.regime == "fixed_opponent"
                        and validation_summary.knight_win_rate >= best_validation_win_rate
                    ):
                        best_validation_win_rate = validation_summary.knight_win_rate
                        knight_policy = trainer.policies["knight"]
                        if hasattr(knight_policy, "update_count"):
                            _save_run_checkpoint(
                                knight_policy,
                                checkpoint_directory,
                                "knight",
                                knight_policy.update_count,
                                metadata,
                                label="best",
                            )

                if (
                    run_config.regime == "alternating_learning"
                    and progress() >= phase_end
                ):
                    alternating_agent = (
                        "boss" if alternating_agent == "knight" else "knight"
                    )
                    phase_end = min(budget_total, progress() + phase_length)

    for agent, policy in trainer.policies.items():
        if hasattr(policy, "update_count"):
            _save_run_checkpoint(
                policy,
                checkpoint_directory,
                agent,
                policy.update_count,
                metadata,
                label="final",
            )
    completed_metadata: dict[str, Any] = {
        "status": "completed",
        "completed_episodes": trainer.completed_episodes,
        "environment_steps": trainer.total_environment_steps,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if best_validation_win_rate >= 0.0:
        completed_metadata["best_validation_knight_win_rate"] = (
            best_validation_win_rate
        )
    metadata.update(completed_metadata)
    _write_json(metadata_path, metadata)
    env.close()
    return {
        "run_id": run_id,
        "run_directory": str(run_directory),
        "completed_episodes": trainer.completed_episodes,
        "environment_steps": trainer.total_environment_steps,
        "training_csv": str(log_path),
        "validation_csv": str(validation_path),
    }


def _save_run_checkpoint(
    policy: Any,
    directory: Path,
    agent: str,
    update: int,
    run_metadata: dict[str, Any],
    *,
    label: str | None = None,
) -> None:
    suffix = label if label is not None else f"update-{update:06d}"
    save_checkpoint(
        policy,
        directory / f"{agent}-{suffix}.pt",
        metadata={
            "run_id": run_metadata["run_id"],
            "regime": run_metadata["regime"],
            "agent": agent,
            "update": update,
            "seed": run_metadata["seed"],
            "alpha_h": run_metadata["alpha_h"],
            "alpha_d": run_metadata["alpha_d"],
        },
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2)


def _summarize_training_episodes(
    episodes: list[dict[str, object]],
) -> dict[str, float | None]:
    if not episodes:
        return {
            "knight_win_rate": None,
            "boss_win_rate": None,
            "draw_rate": None,
            "timeout_rate": None,
            "average_episode_length": None,
        }
    count = len(episodes)
    return {
        "knight_win_rate": sum(
            episode["outcome"] == "knight_win" for episode in episodes
        ) / count,
        "boss_win_rate": sum(
            episode["outcome"] == "boss_win" for episode in episodes
        ) / count,
        "draw_rate": sum(episode["outcome"] == "draw" for episode in episodes)
        / count,
        "timeout_rate": sum(bool(episode["timeout"]) for episode in episodes)
        / count,
        "average_episode_length": sum(
            int(episode["episode_length"]) for episode in episodes
        )
        / count,
    }
