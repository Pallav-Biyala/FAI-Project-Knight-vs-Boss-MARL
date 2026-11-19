"""Play one complete match between random/heuristic baselines or trained PPO policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from knight_boss.agents import HeuristicAgent, RandomAgent
from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.game.config import GameConfig
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.marl.proposal_env import (
    ProposalCombatEnv,
    ProposalHeuristicBoss,
    ProposalHeuristicKnight,
    ProposalRandomAgent,
    ProposalKnightAction,
    ProposalBossAction,
)
from knight_boss.training.checkpointing import load_checkpoint
from knight_boss.training.config import PPOConfig
from knight_boss.visualization.battle_replay import BattleReplayRecorder


def _agent(kind: str, role: str, *, rows: int, columns: int, seed: int, env_type: str = "proposal"):
    if env_type == "proposal":
        if role == "knight":
            return (
                ProposalHeuristicKnight(rows=rows, columns=columns, seed=seed)
                if kind == "heuristic"
                else ProposalRandomAgent(rows=rows, columns=columns, seed=seed)
            )
        return (
            ProposalHeuristicBoss(rows=rows, columns=columns, seed=seed)
            if kind == "heuristic"
            else ProposalRandomAgent(rows=rows, columns=columns, seed=seed)
        )
    agent_class = {"random": RandomAgent, "heuristic": HeuristicAgent}[kind]
    return agent_class(rows=rows, columns=columns, seed=seed)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knight", choices=("random", "heuristic"), default="heuristic")
    parser.add_argument("--boss", choices=("random", "heuristic"), default="heuristic")
    parser.add_argument("--knight-checkpoint", help="Path to a trained Knight PPO checkpoint.")
    parser.add_argument("--boss-checkpoint", help="Path to a trained Boss PPO checkpoint.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic action selection for PPO.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-h", type=float)
    parser.add_argument("--alpha-d", type=float)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", help="Save a JSON battle replay to this path.")
    parser.add_argument(
        "--env-type",
        choices=("proposal", "base"),
        default="proposal",
        help="Environment type (default: proposal).",
    )
    args = parser.parse_args()

    config_path = project_root / "configs" / "game.yaml"
    config = GameConfig.from_yaml(config_path)
    if args.env_type == "proposal":
        env = ProposalCombatEnv(
            config,
            alpha_h=args.alpha_h,
            alpha_d=args.alpha_d,
            render_mode="ansi" if args.render else None,
        )
    else:
        env = KnightBossParallelEnv(
            config,
            alpha_h=args.alpha_h,
            alpha_d=args.alpha_d,
            render_mode="ansi" if args.render else None,
        )
    observations, _ = env.reset(seed=args.seed)

    policies = {}
    policy_labels = {}

    if args.knight_checkpoint:
        ppo_config = PPOConfig.from_yaml(project_root / "configs" / "ppo.yaml")
        obs_size = observation_to_vector(observations["knight"]).size
        knight_agent = PPOAgent(
            obs_size,
            ppo_config,
            seed=args.seed,
            n_actions=int(env.action_space("knight").n),
        )
        load_checkpoint(knight_agent, args.knight_checkpoint, load_optimizer=False)
        knight_agent.reset(seed=args.seed)
        policies["knight"] = knight_agent
        policy_labels["knight"] = f"ppo({Path(args.knight_checkpoint).name})"
    else:
        policies["knight"] = _agent(
            args.knight,
            "knight",
            rows=config.rows,
            columns=config.columns,
            seed=args.seed,
            env_type=args.env_type,
        )
        policy_labels["knight"] = args.knight

    if args.boss_checkpoint:
        ppo_config = PPOConfig.from_yaml(project_root / "configs" / "ppo.yaml")
        obs_size = observation_to_vector(observations["boss"]).size
        boss_agent = PPOAgent(
            obs_size,
            ppo_config,
            seed=args.seed + 1,
            n_actions=int(env.action_space("boss").n),
        )
        load_checkpoint(boss_agent, args.boss_checkpoint, load_optimizer=False)
        boss_agent.reset(seed=args.seed + 1)
        policies["boss"] = boss_agent
        policy_labels["boss"] = f"ppo({Path(args.boss_checkpoint).name})"
    else:
        policies["boss"] = _agent(
            args.boss,
            "boss",
            rows=config.rows,
            columns=config.columns,
            seed=args.seed + 1,
            env_type=args.env_type,
        )
        policy_labels["boss"] = args.boss

    recorder = (
        BattleReplayRecorder(
            env.engine,
            seed=args.seed,
            policies=policy_labels,
        )
        if args.record and hasattr(env, "engine")
        else None
    )

    if args.render:
        print("Initial state:")
        print(env.render())

    while env.agents:
        actions = {}
        for agent in env.agents:
            policy = policies[agent]
            if isinstance(policy, PPOAgent):
                actions[agent] = policy.act(observations[agent], deterministic=args.deterministic)
            else:
                actions[agent] = policy.act(observations[agent])

        observations, rewards, terminations, truncations, infos = env.step(actions)
        if recorder is not None:
            recorder.record_step(
                env.engine, actions, rewards, infos, terminations, truncations
            )
        if args.render:
            action_names = {
                agent: (
                    ProposalKnightAction(actions[agent]).name
                    if agent == "knight"
                    else ProposalBossAction(actions[agent]).name
                )
                for agent in actions
            }
            print(f"Step {env.step_count} | Actions: {action_names}")
            print(env.render())

    print(f"Outcome: {infos['knight']['outcome']}")
    print(f"Steps: {infos['knight']['step_count']}")
    print(f"Rewards: {rewards}")
    print(f"Damage dealt: Knight={infos['knight']['damage_dealt']}, "
          f"Boss={infos['boss']['damage_dealt']}")
    print(f"Damage received: Knight={infos['knight']['damage_received']}, "
          f"Boss={infos['boss']['damage_received']}")
    print(f"Terminated: {terminations}; truncated: {truncations}")
    if recorder is not None:
        replay_path = recorder.save(args.record)
        print(f"Replay: {replay_path}")
    env.close()


if __name__ == "__main__":
    main()
