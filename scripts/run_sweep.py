"""Execute or preview the independent capability parameter sweep."""

import argparse
from dataclasses import replace
from pathlib import Path

from knight_boss.experiments.parallel_runner import (
    explicit_parameter_grid,
    run_parallel_sweep,
)
from knight_boss.experiments.parameter_sweep import (
    ExperimentConfig,
    parameter_grid,
    run_sweep,
)
from knight_boss.game.config import GameConfig
from knight_boss.marl.rewards import REWARD_MODES
from knight_boss.training.config import PPOConfig
from knight_boss.training.regimes import TrainingRunConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default=str(project_root / "configs" / "experiments.yaml"))
    parser.add_argument("--output-root")
    parser.add_argument("--regimes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--alpha-h-values", nargs="+", type=float)
    parser.add_argument("--alpha-d-values", nargs="+", type=float)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--phase-episodes", type=int)
    parser.add_argument("--reward-mode", choices=REWARD_MODES)
    parser.add_argument("--shaping-weight", type=float)
    parser.add_argument("--draw-penalty", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--knight-checkpoint")
    parser.add_argument("--boss-checkpoint")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    experiment_config = ExperimentConfig.from_yaml(args.config)
    if args.output_root is not None:
        experiment_config = replace(experiment_config, output_root=args.output_root)
    if args.regimes is not None:
        experiment_config = replace(experiment_config, regimes=tuple(args.regimes))
    if args.seeds is not None:
        experiment_config = replace(experiment_config, seeds=tuple(args.seeds))

    if (args.alpha_h_values is None) != (args.alpha_d_values is None):
        parser.error("Provide both --alpha-h-values and --alpha-d-values together.")
    if args.alpha_h_values is not None and args.alpha_d_values is not None:
        grid = explicit_parameter_grid(args.alpha_h_values, args.alpha_d_values)
        custom_grid = True
    else:
        grid = parameter_grid(experiment_config)
        custom_grid = False

    total_runs = len(experiment_config.regimes) * len(experiment_config.seeds) * len(grid)
    print(
        f"Grid: {len(grid)} independent (alpha_h, alpha_d) pairs; "
        f"{len(experiment_config.seeds)} seeds x "
        f"{len(experiment_config.regimes)} regimes = {total_runs} training runs."
    )
    if args.dry_run:
        h_vals = sorted({alpha_h for alpha_h, _ in grid})
        d_vals = sorted({alpha_d for _, alpha_d in grid})
        print(f"Alpha_H values ({len(h_vals)}): {h_vals[0]}..{h_vals[-1]}")
        print(f"Alpha_D values ({len(d_vals)}): {d_vals[0]}..{d_vals[-1]}")
        if not custom_grid:
            print(f"Resolution: {experiment_config.resolution}")
        print(f"Regimes: {list(experiment_config.regimes)} | Seeds: {list(experiment_config.seeds)}")
        return

    regime_configs: dict[str, TrainingRunConfig] = {}
    for regime in experiment_config.regimes:
        cfg = TrainingRunConfig.from_yaml(project_root / "configs" / f"{regime}.yaml")
        updates: dict[str, object] = {}
        if args.episodes is not None:
            updates["training_episodes"] = args.episodes
        if args.phase_episodes is not None and regime == "alternating_learning":
            updates["phase_episodes"] = args.phase_episodes
        if args.reward_mode is not None:
            updates["reward_mode"] = args.reward_mode
        if args.shaping_weight is not None:
            updates["shaping_weight"] = args.shaping_weight
        if args.draw_penalty is not None:
            updates["draw_penalty"] = args.draw_penalty
        if args.validation_episodes is not None:
            updates["validation_episodes"] = args.validation_episodes
        if args.validation_interval is not None:
            updates["validation_interval_updates"] = args.validation_interval
        if args.checkpoint_interval is not None:
            updates["checkpoint_interval_updates"] = args.checkpoint_interval
        if updates:
            cfg = replace(cfg, **updates)
        regime_configs[regime] = cfg

    game_config = GameConfig.from_yaml(project_root / "configs" / "game.yaml")
    ppo_config = PPOConfig.from_yaml(project_root / "configs" / "ppo.yaml")
    if args.entropy_coef is not None:
        ppo_config = replace(ppo_config, entropy_coef=args.entropy_coef)
    if args.gamma is not None:
        ppo_config = replace(ppo_config, gamma=args.gamma)

    if (
        args.workers == 1
        and not custom_grid
        and args.knight_checkpoint is None
        and args.boss_checkpoint is None
    ):
        result = run_sweep(
            experiment_config,
            game_config,
            ppo_config,
            regime_configs,
        )
    else:
        result = run_parallel_sweep(
            experiment_config,
            game_config,
            ppo_config,
            regime_configs,
            parameter_pairs=grid if custom_grid else None,
            max_workers=args.workers,
            knight_checkpoint=args.knight_checkpoint,
            boss_checkpoint=args.boss_checkpoint,
        )
    print(f"Experiment: {result['experiment_id']}")
    print(f"Completed runs: {result['completed_run_count']}")
    print(f"Manifest: {result['training_manifest']}")


if __name__ == "__main__":
    main()
