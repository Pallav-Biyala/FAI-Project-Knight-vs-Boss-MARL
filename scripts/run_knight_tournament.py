from pathlib import Path
import json
import csv
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]

from knight_boss.game.config import GameConfig
from knight_boss.training.config import PPOConfig
from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.training.checkpointing import load_checkpoint
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.proposal_env import (
    ProposalCombatEnv,
    ProposalHeuristicBoss,
)
from knight_boss.evaluation.evaluator import evaluate_policies


# ---------------------------------------------------------
# Tournament configuration
# ---------------------------------------------------------

EPISODES = 500
SEED = 200000
ALPHA_H = 1.0
ALPHA_D = 1.0
REWARD_MODE = "tactical"
SHAPING_WEIGHT = 0.5

RUNS = ROOT / "experiments_corrected_final_v1" / "runs"

KNIGHTS = {
    "simultaneous": RUNS / "simultaneous_learning_1_1_tactical_s0/simultaneous_learning/simultaneous_learning-20261120T124204Z-62aef3c2/checkpoints/knight-final.pt",
    "alternating": RUNS / "alternating_learning_1_1_tactical_s0/alternating_learning/alternating_learning-20261120T124311Z-19355dcf/checkpoints/knight-final.pt",
    "population": RUNS / "population_self_play_1_1_tactical_s0/population_self_play/population_self_play-20261120T124349Z-c45d302f/checkpoints/knight-final.pt",
}

BOSSES = {
    "simultaneous": RUNS / "simultaneous_learning_1_1_tactical_s0/simultaneous_learning/simultaneous_learning-20261120T124204Z-62aef3c2/checkpoints/boss-final.pt",
    "alternating": RUNS / "alternating_learning_1_1_tactical_s0/alternating_learning/alternating_learning-20261120T124311Z-19355dcf/checkpoints/boss-final.pt",
    "population": RUNS / "population_self_play_1_1_tactical_s0/population_self_play/population_self_play-20261120T124349Z-c45d302f/checkpoints/boss-final.pt",
}


# ---------------------------------------------------------
# Load configurations and determine observation size
# ---------------------------------------------------------

game_config = GameConfig.from_yaml(ROOT / "configs/game.yaml")
ppo_config = PPOConfig.from_yaml(ROOT / "configs/ppo.yaml")

probe_env = ProposalCombatEnv(
    game_config,
    alpha_h=ALPHA_H,
    alpha_d=ALPHA_D,
    reward_mode=REWARD_MODE,
    shaping_weight=SHAPING_WEIGHT,
    gamma=ppo_config.gamma,
)

probe_obs, _ = probe_env.reset(seed=0)
obs_size = observation_to_vector(probe_obs["knight"]).size
probe_env.close()


def load_agent(path):
    agent = PPOAgent(
        obs_size,
        ppo_config,
        seed=0,
        n_actions=8,
    )
    load_checkpoint(agent, path, load_optimizer=False)
    agent.policy.eval()
    return agent

print("Loading three Knight policies...")
knights = {name: load_agent(path) for name, path in KNIGHTS.items()}

print("Loading Boss policies...")
bosses = {}
for name, path in BOSSES.items():
    if path is None:
        bosses[name] = ProposalHeuristicBoss(
            rows=game_config.rows,
            columns=game_config.columns,
            seed=SEED,
        )
    else:
        bosses[name] = load_agent(path)


def make_env():
    return ProposalCombatEnv(
        game_config,
        alpha_h=ALPHA_H,
        alpha_d=ALPHA_D,
        reward_mode=REWARD_MODE,
        shaping_weight=SHAPING_WEIGHT,
        gamma=ppo_config.gamma,
    )


# ---------------------------------------------------------
# Run cross-play tournament
# ---------------------------------------------------------

results = []

total = len(knights) * len(bosses)
current = 0

for knight_name, knight in knights.items():
    for boss_name, boss in bosses.items():
        current += 1

        print(
            f"\n[{current}/{total}] "
            f"Knight: {knight_name} vs Boss: {boss_name}"
        )

        _, summary = evaluate_policies(
            {"knight": knight, "boss": boss},
            game_config,
            episodes=EPISODES,
            seed=SEED,
            alpha_h=ALPHA_H,
            alpha_d=ALPHA_D,
            deterministic=False,
            env_factory=make_env,
        )

        row = {
            "knight": knight_name,
            "boss": boss_name,
            "episodes": EPISODES,
            **summary.to_dict(),
        }

        results.append(row)

        print(
            f"Knight wins: {row.get('knight_wins')} | "
            f"Boss wins: {row.get('boss_wins')} | "
            f"Draws: {row.get('draws')} | "
            f"Win rate: {row.get('knight_win_rate')}"
        )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_dir = ROOT / "experiments_corrected_tournament_v1"
output_dir.mkdir(exist_ok=True)

with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(output_dir / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 75)
print("TOURNAMENT COMPLETE")
print("=" * 75)

print(f"{'Knight':<15} {'Boss':<15} {'Wins':>7} {'Losses':>7} {'Draws':>7} {'Win rate':>10}")

for row in results:
    print(
        f"{row['knight']:<15} "
        f"{row['boss']:<15} "
        f"{row['knight_wins']:>7} "
        f"{row['boss_wins']:>7} "
        f"{row['draws']:>7} "
        f"{row['knight_win_rate']:>9.1%}"
    )

print(f"\nResults saved to: {output_dir}")
