"""Train PPO agents using one of the three configured learning regimes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from knight_boss.game.config import GameConfig
from knight_boss.marl.proposal_env import (
    ProposalCombatEnv,
    ProposalHeuristicBoss,
)
from knight_boss.marl.rewards import PROPOSAL_REWARD_MODES
from knight_boss.training.config import PPOConfig
from knight_boss.training.regimes import REGIMES, TrainingRunConfig, run_training


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-h", type=float)
    parser.add_argument("--alpha-d", type=float)
    parser.add_argument("--output-root")
    parser.add_argument("--episodes", type=int, help="Override number of training episodes.")
    parser.add_argument(
        "--steps",
        type=int,
        help="Override training budget in environment steps (replaces the episode budget).",
    )
    parser.add_argument(
        "--phase-steps",
        type=int,
        help="Alternating regime: environment steps per Knight/Boss phase.",
    )
    parser.add_argument("--knight-checkpoint", help="Resume Knight policy from this checkpoint.")
    parser.add_argument("--boss-checkpoint", help="Resume Boss policy from this checkpoint.")
    parser.add_argument(
        "--reward-mode",
        choices=PROPOSAL_REWARD_MODES,
        help="Override reward mode (default: terminal).",
    )
    parser.add_argument(
        "--shaping-weight",
        type=float,
        help="Health-delta shaping scale when --reward-mode shaped is selected.",
    )
    parser.add_argument(
        "--draw-penalty",
        type=float,
        help="Draw/timeout penalty magnitude when --reward-mode draw_penalty is selected.",
    )
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument(
        "--env-type",
        choices=("proposal", "base"),
        default="proposal",
        help="Environment type (default: proposal).",
    )
    args = parser.parse_args()

    run_config = TrainingRunConfig.from_yaml(
        project_root / "configs" / f"{args.regime}.yaml"
    )
    overrides = {}
    if args.output_root:
        overrides["output_root"] = args.output_root
    if args.episodes is not None:
        if args.episodes < 1:
            parser.error("--episodes must be positive.")
        overrides["training_episodes"] = args.episodes
    if args.steps is not None:
        if args.steps < 1:
            parser.error("--steps must be positive.")
        overrides["training_steps"] = args.steps
    if args.phase_steps is not None:
        if args.phase_steps < 1:
            parser.error("--phase-steps must be positive.")
        overrides["phase_steps"] = args.phase_steps
    if args.reward_mode:
        overrides["reward_mode"] = args.reward_mode
    if args.shaping_weight is not None:
        overrides["shaping_weight"] = args.shaping_weight
    if args.draw_penalty is not None:
        overrides["draw_penalty"] = args.draw_penalty
    if args.alpha_h is not None:
        overrides["validation_alpha_h"] = args.alpha_h
    if args.alpha_d is not None:
        overrides["validation_alpha_d"] = args.alpha_d
    if overrides:
        run_config = replace(run_config, **overrides)

    game_config = GameConfig.from_yaml(project_root / "configs" / "game.yaml")
    ppo_config = PPOConfig.from_yaml(project_root / "configs" / "ppo.yaml")
    ppo_updates: dict[str, object] = {}
    if args.gamma is not None:
        ppo_updates["gamma"] = args.gamma
    if args.entropy_coef is not None:
        ppo_updates["entropy_coef"] = args.entropy_coef
    if ppo_updates:
        ppo_config = replace(ppo_config, **ppo_updates)

    if args.env_type == "proposal":
        env_factory = lambda ah, ad: ProposalCombatEnv(
            game_config,
            alpha_h=ah,
            alpha_d=ad,
            reward_mode=run_config.reward_mode,
            shaping_weight=run_config.shaping_weight,
            draw_penalty=run_config.draw_penalty,
            gamma=ppo_config.gamma,
        )
        heuristic_factory = lambda s: ProposalHeuristicBoss(
            rows=game_config.rows,
            columns=game_config.columns,
            seed=s,
        )
    else:
        env_factory = None
        heuristic_factory = None

    summary = run_training(
        run_config,
        game_config,
        ppo_config,
        seed=args.seed,
        alpha_h=args.alpha_h,
        alpha_d=args.alpha_d,
        knight_checkpoint=args.knight_checkpoint,
        boss_checkpoint=args.boss_checkpoint,
        env_factory=env_factory,
        heuristic_factory=heuristic_factory,
    )
    print(f"Run ID: {summary['run_id']}")
    print(f"Run directory: {summary['run_directory']}")
    print(f"Completed episodes: {summary['completed_episodes']}")
    print(f"Environment steps: {summary['environment_steps']}")
    print(f"Training log: {summary['training_csv']}")


if __name__ == "__main__":
    main()
