"""Evaluate a saved PPO policy pair independently of training."""

from __future__ import annotations

import argparse
from pathlib import Path

from knight_boss.agents import HeuristicAgent, RandomAgent
from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.evaluation.evaluator import evaluate_policies
from knight_boss.evaluation.io import evaluation_records, save_evaluation
from knight_boss.experiments.parameter_sweep import (
    ExperimentConfig,
    parameter_grid,
)
from knight_boss.game.config import GameConfig
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.marl.proposal_env import (
    ProposalCombatEnv,
    ProposalHeuristicBoss,
    ProposalRandomAgent,
)
from knight_boss.training.checkpointing import load_checkpoint
from knight_boss.training.config import PPOConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knight-checkpoint", required=True)
    parser.add_argument("--boss-checkpoint")
    parser.add_argument("--boss-opponent", choices=("heuristic", "random"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha-h", type=float)
    parser.add_argument("--alpha-d", type=float)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--env-type",
        choices=("proposal", "base"),
        default="proposal",
        help="Environment type (default: proposal).",
    )
    args = parser.parse_args()
    if bool(args.boss_checkpoint) == bool(args.boss_opponent):
        parser.error("Provide exactly one of --boss-checkpoint or --boss-opponent.")

    ppo_config = PPOConfig.from_yaml(project_root / "configs" / "ppo.yaml")
    game_config = GameConfig.from_yaml(project_root / "configs" / "game.yaml")
    evaluation_config = _load_evaluation_config(project_root / "configs" / "evaluation.yaml")
    episodes = (
        args.episodes
        if args.episodes is not None
        else int(evaluation_config["episodes_per_model"])
    )
    if episodes < 1:
        parser.error("--episodes must be positive.")
    seed = args.seed if args.seed is not None else int(evaluation_config["base_seed"])
    deterministic = args.deterministic or bool(evaluation_config["deterministic"])
    confidence = float(evaluation_config["confidence_level"])

    if args.env_type == "proposal":
        env = ProposalCombatEnv(
            game_config,
            reward_mode="shaped",
            shaping_weight=0.5,
            draw_penalty=0.0,
            gamma=ppo_config.gamma,
        )
    else:
        env = KnightBossParallelEnv(game_config)
    observation, _ = env.reset(seed=seed)
    observation_size = observation_to_vector(observation["knight"]).size
    knight = PPOAgent(
        observation_size,
        ppo_config,
        seed=seed,
        n_actions=int(env.action_space("knight").n),
    )
    knight_metadata = load_checkpoint(
        knight, args.knight_checkpoint, load_optimizer=False
    )
    if args.boss_checkpoint:
        boss = PPOAgent(
            observation_size,
            ppo_config,
            seed=seed + 1,
            n_actions=int(env.action_space("boss").n),
        )
        boss_metadata = load_checkpoint(
            boss, args.boss_checkpoint, load_optimizer=False
        )
        _validate_checkpoint_pair(knight_metadata, boss_metadata, parser)
    else:
        if args.boss_opponent == "heuristic":
            boss = (
                ProposalHeuristicBoss(
                    rows=game_config.rows,
                    columns=game_config.columns,
                    seed=seed + 1,
                )
                if args.env_type == "proposal"
                else HeuristicAgent(
                    rows=game_config.rows,
                    columns=game_config.columns,
                    seed=seed + 1,
                )
            )
        else:
            boss = (
                ProposalRandomAgent(
                    rows=game_config.rows,
                    columns=game_config.columns,
                    seed=seed + 1,
                )
                if args.env_type == "proposal"
                else RandomAgent(
                    rows=game_config.rows,
                    columns=game_config.columns,
                    seed=seed + 1,
                )
            )
        boss_metadata = {"regime": "baseline", "seed": None}
    env.close()

    if args.sweep:
        experiment_config = ExperimentConfig.from_yaml(
            project_root / "configs" / "experiments.yaml"
        )
        evaluation_parameters = parameter_grid(experiment_config)
    else:
        train_alpha_h = knight_metadata.get("alpha_h", game_config.alpha_h)
        train_alpha_d = knight_metadata.get("alpha_d", game_config.alpha_d)
        evaluation_parameters = (
            (
                float(args.alpha_h if args.alpha_h is not None else train_alpha_h),
                float(args.alpha_d if args.alpha_d is not None else train_alpha_d),
            ),
        )

    summaries = []
    episode_records = []
    for alpha_h, alpha_d in evaluation_parameters:
        if args.env_type == "proposal":
            env_factory = lambda: ProposalCombatEnv(
                game_config,
                alpha_h=alpha_h,
                alpha_d=alpha_d,
                reward_mode="shaped",
                shaping_weight=0.5,
                draw_penalty=0.0,
                gamma=ppo_config.gamma,
            )
        else:
            env_factory = None

        episodes_result, summary = evaluate_policies(
            {"knight": knight, "boss": boss},
            game_config,
            episodes=episodes,
            seed=seed,
            alpha_h=alpha_h,
            alpha_d=alpha_d,
            deterministic=deterministic,
            confidence=confidence,
            env_factory=env_factory,
        )
        summary_record, match_records = evaluation_records(
            summary,
            episodes_result,
            training_metadata=knight_metadata,
            evaluation_alpha_h=alpha_h,
            evaluation_alpha_d=alpha_d,
            evaluation_seed=seed,
            deterministic=deterministic,
        )
        summary_record["boss_training_regime"] = boss_metadata.get("regime")
        summary_record["boss_training_seed"] = boss_metadata.get("seed")
        summaries.append(summary_record)
        episode_records.extend(match_records)

    output_directory = save_evaluation(
        summaries,
        episode_records,
        output_root=args.output_root,
        metadata={
            "training_checkpoint": str(Path(args.knight_checkpoint).resolve()),
            "boss_checkpoint": (
                str(Path(args.boss_checkpoint).resolve())
                if args.boss_checkpoint
                else None
            ),
            "training_metadata": knight_metadata,
            "boss_metadata": boss_metadata,
            "evaluation_episodes_per_configuration": episodes,
            "evaluation_seed": seed,
            "deterministic_policy": deterministic,
            "evaluation_configurations": len(evaluation_parameters),
        },
    )
    print(f"Evaluated configurations: {len(summaries)}")
    print(f"Evaluation results: {output_directory}")


def _load_evaluation_config(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Evaluation configuration must be a YAML mapping.")
    return data


def _validate_checkpoint_pair(knight: dict, boss: dict, parser) -> None:
    for key in ("run_id", "regime", "seed", "alpha_h", "alpha_d"):
        if key in knight and key in boss and knight[key] != boss[key]:
            parser.error(f"Knight and Boss checkpoints do not match on {key}.")


if __name__ == "__main__":
    main()
