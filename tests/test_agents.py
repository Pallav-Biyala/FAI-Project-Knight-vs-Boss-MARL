import numpy as np

from knight_boss.agents import HeuristicAgent, RandomAgent, legal_actions
from knight_boss.game.config import GameConfig
from knight_boss.game.types import Action, Outcome
from knight_boss.marl.parallel_env import KnightBossParallelEnv


def observation_at(knight, boss):
    env = KnightBossParallelEnv(
        GameConfig(knight_position=knight, boss_position=boss)
    )
    observations, _ = env.reset(seed=3)
    return observations


def test_legal_actions_exclude_boundaries_occupied_moves_and_unavailable_attacks():
    observations = observation_at((0, 0), (0, 1))
    actions = legal_actions(observations["knight"])
    assert Action.UP not in actions
    assert Action.LEFT not in actions
    assert Action.RIGHT not in actions
    assert Action.BASIC_ATTACK in actions
    assert Action.WAIT in actions


def test_random_agent_samples_only_legal_actions_and_repeats_with_seed():
    observation = observation_at((4, 4), (4, 5))["knight"]
    first = RandomAgent(seed=42)
    second = RandomAgent(seed=42)
    first_actions = [first.act(observation) for _ in range(20)]
    second_actions = [second.act(observation) for _ in range(20)]
    assert first_actions == second_actions
    assert set(first_actions).issubset(set(legal_actions(observation)))


def test_heuristic_attacks_in_range_and_moves_towards_opponent():
    policy = HeuristicAgent()
    adjacent = observation_at((4, 4), (4, 5))["knight"]
    assert policy.act(adjacent) is Action.BASIC_ATTACK

    two_cells_away = observation_at((4, 3), (4, 5))["knight"]
    assert policy.act(two_cells_away) is Action.HEAVY_ATTACK

    far = observation_at((4, 1), (4, 7))["knight"]
    action = policy.act(far)
    assert action is Action.RIGHT


def test_heuristic_is_deterministic_and_waits_when_no_move_is_legal():
    policy = HeuristicAgent()
    far = observation_at((4, 1), (4, 7))["knight"]
    assert policy.act(far) is policy.act(far)

    trapped = observation_at((0, 0), (0, 1))["knight"]
    trapped["cooldowns"][0] = 1.0
    trapped["cooldowns"][1] = 1.0
    assert policy.act(trapped) is Action.DOWN


def test_agents_complete_a_match_through_parallel_environment():
    config = GameConfig(
        max_steps=20,
        knight_position=(4, 4),
        boss_position=(4, 5),
    )
    env = KnightBossParallelEnv(config)
    observations, _ = env.reset(seed=12)
    policies = {
        "knight": HeuristicAgent(seed=12),
        "boss": RandomAgent(seed=13),
    }
    while env.agents:
        actions = {
            agent: policies[agent].act(observations[agent])
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
    assert infos["knight"]["step_count"] <= config.max_steps
    assert infos["knight"]["outcome"] in {value.value for value in Outcome}
    assert sum(rewards.values()) == 0
    assert all(terminations[agent] or truncations[agent] for agent in env.possible_agents)
    assert np.isclose(
        infos["knight"]["damage_dealt"],
        infos["boss"]["damage_received"],
    )
