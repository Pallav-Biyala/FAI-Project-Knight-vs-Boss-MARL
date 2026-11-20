"""Re-evaluate saved unmasked alternating checkpoints with behavior counters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.evaluation.evaluator import evaluate_policies
from knight_boss.game.config import GameConfig
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.proposal_env import ProposalCombatEnv, ProposalHeuristicBoss
from knight_boss.training.checkpointing import load_checkpoint
from knight_boss.training.config import PPOConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=project_root / "experiments_confirmatory_v1" / "results.jsonl",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root
        / "experiments_confirmatory_v1"
        / "behavior_diagnostics.jsonl",
    )
    args = parser.parse_args()

    game_config = GameConfig.from_yaml(project_root / "configs" / "game.yaml")
    ppo_config = PPOConfig.from_yaml(project_root / "configs" / "ppo.yaml")
    env = ProposalCombatEnv(game_config, reward_mode="tactical")
    probe_observation = env.reset(seed=0)[0]
    observation_size = observation_to_vector(probe_observation["knight"]).size

    old_runs: list[dict] = []
    with args.results.open(encoding="utf-8") as results_file:
        for line in results_file:
            if line.strip():
                old_runs.extend(
                    row
                    for row in json.loads(line).get("rows", [])
                    if row.get("regime") == "alternating_learning"
                )

    by_seed: dict[int, dict] = {}
    for row in old_runs:
        by_seed[int(row["seed"])] = row
    if not by_seed:
        raise SystemExit("No alternating-learning runs found in the result file.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for seed, run_row in sorted(by_seed.items()):
            run_root = (
                args.results.resolve().parent
                / "runs"
                / run_row["key"].replace("|", "_")
            )
            knight_path = next(run_root.rglob("knight-final.pt"), None)
            boss_path = next(run_root.rglob("boss-final.pt"), None)
            if knight_path is None or boss_path is None:
                raise FileNotFoundError(
                    f"Expected Knight and Boss checkpoints below {run_root}."
                )

            knight = PPOAgent(
                observation_size,
                ppo_config,
                seed=seed,
                n_actions=8,
                respect_action_masks=False,
            )
            boss = PPOAgent(
                observation_size,
                ppo_config,
                seed=seed + 1,
                n_actions=8,
                respect_action_masks=False,
            )
            load_checkpoint(knight, knight_path, load_optimizer=False)
            load_checkpoint(boss, boss_path, load_optimizer=False)

            def env_factory() -> ProposalCombatEnv:
                return ProposalCombatEnv(
                    game_config,
                    alpha_h=float(run_row["alpha_h"]),
                    alpha_d=float(run_row["alpha_d"]),
                    reward_mode="tactical",
                    gamma=ppo_config.gamma,
                )

            opponents = {
                "heuristic": ProposalHeuristicBoss(
                    rows=game_config.rows,
                    columns=game_config.columns,
                    seed=seed + 1,
                ),
                "matched": boss,
            }
            rows = []
            for opponent_name, opponent in opponents.items():
                _, summary = evaluate_policies(
                    {"knight": knight, "boss": opponent},
                    game_config,
                    episodes=args.episodes,
                    seed=args.seed,
                    alpha_h=float(run_row["alpha_h"]),
                    alpha_d=float(run_row["alpha_d"]),
                    deterministic=False,
                    env_factory=env_factory,
                )
                rows.append(
                    {
                        "key": run_row["key"],
                        "regime": "alternating_learning",
                        "seed": seed,
                        "opponent": opponent_name,
                        "episodes": args.episodes,
                        "respect_action_masks": False,
                        **summary.to_dict(),
                    }
                )
            output.write(json.dumps({"key": run_row["key"], "rows": rows}) + "\n")
            output.flush()
            print(f"Seed {seed} behavior audit complete.")

    env.close()
    print(f"Behavior diagnostics saved to {args.out}")


if __name__ == "__main__":
    main()
