"""Evaluate all trained checkpoints listed in a parameter-sweep manifest."""

import argparse
from pathlib import Path
import yaml

from knight_boss.evaluation.sweep_evaluator import (
    CHECKPOINT_POLICIES,
    OPPONENT_MODES,
    evaluate_sweep_manifest,
)
from knight_boss.experiments.parallel_runner import explicit_parameter_grid
from knight_boss.experiments.parameter_sweep import ExperimentConfig, parameter_grid
from knight_boss.game.config import GameConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--game-config", default=str(project_root / "configs" / "game.yaml"))
    parser.add_argument("--evaluation-config", default=str(project_root / "configs" / "evaluation.yaml"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint-policy", choices=CHECKPOINT_POLICIES, default="auto")
    parser.add_argument("--opponent-mode", choices=OPPONENT_MODES, default="heuristic")
    parser.add_argument("--grid-config")
    parser.add_argument("--alpha-h-values", nargs="+", type=float)
    parser.add_argument("--alpha-d-values", nargs="+", type=float)
    parser.add_argument("--output-root", default=str(project_root / "results"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-episodes-csv", action="store_false", dest="save_episodes")
    parser.set_defaults(save_episodes=True)
    args = parser.parse_args()

    with Path(args.evaluation_config).open("r", encoding="utf-8") as handle:
        eval_cfg = yaml.safe_load(handle)

    episodes = args.episodes if args.episodes is not None else int(eval_cfg["episodes_per_model"])
    seed = args.seed if args.seed is not None else int(eval_cfg["base_seed"])
    deterministic = args.deterministic or bool(eval_cfg["deterministic"])
    confidence = float(eval_cfg["confidence_level"])

    if (args.alpha_h_values is None) != (args.alpha_d_values is None):
        parser.error("Provide both --alpha-h-values and --alpha-d-values together.")

    grid_pairs = None
    if args.alpha_h_values is not None and args.alpha_d_values is not None:
        grid_pairs = explicit_parameter_grid(args.alpha_h_values, args.alpha_d_values)
    elif args.grid_config is not None:
        grid_pairs = parameter_grid(ExperimentConfig.from_yaml(args.grid_config))

    game_config = GameConfig.from_yaml(args.game_config)
    result = evaluate_sweep_manifest(
        args.manifest,
        game_config,
        episodes=episodes,
        base_seed=seed,
        deterministic=deterministic,
        confidence=confidence,
        checkpoint_policy=args.checkpoint_policy,
        opponent_mode=args.opponent_mode,
        grid_pairs=grid_pairs,
        output_root=args.output_root,
        max_workers=args.workers,
        save_episodes=args.save_episodes,
    )
    print(f"Evaluation directory: {result['evaluation_directory']}")
    print(f"Summaries CSV: {result['summaries_csv']}")
    print(f"Total summaries: {result['total_summaries']}")


if __name__ == "__main__":
    main()
