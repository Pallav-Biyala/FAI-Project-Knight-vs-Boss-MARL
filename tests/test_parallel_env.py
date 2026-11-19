import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from knight_boss.game.config import GameConfig
from knight_boss.marl.parallel_env import KnightBossParallelEnv


def test_reset_observations_are_valid_and_include_required_state():
    env = KnightBossParallelEnv()
    observations, infos = env.reset(seed=17)
    assert set(observations) == {"knight", "boss"}
    assert set(infos) == {"knight", "boss"}
    for agent, observation in observations.items():
        assert env.observation_space(agent).contains(observation)
        assert observation["positions"].shape == (4,)
        assert observation["health"].shape == (4,)
        assert observation["health_fraction"].shape == (2,)
        assert observation["cooldowns"].shape == (4,)
        assert observation["combat_stats"].shape == (12,)
        assert observation["windups"].shape == (8,)
        assert observation["remaining_steps"][0] == 300


def test_health_ranges_and_cooldowns_update_in_observation():
    env = KnightBossParallelEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5))
    )
    env.reset(seed=1)
    observations, _, _, _, _ = env.step({"knight": 4, "boss": 6})
    knight = observations["knight"]
    assert knight["health"][1] == 130
    assert knight["health_fraction"][1] == pytest.approx(130 / 150)
    assert knight["cooldowns"][0] == 1
    assert knight["remaining_steps"][0] == 299


def test_heavy_windup_state_is_observable_to_both_agents():
    env = KnightBossParallelEnv(
        GameConfig(knight_position=(4, 3), boss_position=(4, 5))
    )
    env.reset(seed=1)
    observations, _, _, _, _ = env.step({"knight": 5, "boss": 6})
    assert observations["knight"]["windups"][0] == 1
    assert tuple(observations["knight"]["windups"][1:3]) == (0.0, 1.0)
    assert observations["boss"]["windups"][4] == 1
    assert tuple(observations["boss"]["windups"][5:7]) == (0.0, 1.0)


def test_terminal_rewards_are_zero_sum_and_terminal_flags_are_distinct():
    config = GameConfig(
        knight_position=(4, 4),
        boss_position=(4, 5),
        max_steps=10,
    )
    env = KnightBossParallelEnv(config, alpha_h=0.05)
    env.reset()
    _, rewards, terminations, truncations, _ = env.step(
        {"knight": 4, "boss": 6}
    )
    assert rewards == {"knight": 1.0, "boss": -1.0}
    assert terminations == {"knight": True, "boss": True}
    assert truncations == {"knight": False, "boss": False}
    assert env.agents == []

    timeout_env = KnightBossParallelEnv(
        GameConfig(max_steps=1, knight_position=(0, 0), boss_position=(8, 8))
    )
    timeout_env.reset()
    _, rewards, terminations, truncations, _ = timeout_env.step(
        {"knight": 6, "boss": 6}
    )
    assert rewards == {"knight": 0.0, "boss": 0.0}
    assert terminations == {"knight": False, "boss": False}
    assert truncations == {"knight": True, "boss": True}


def test_action_validation_and_reset_options():
    env = KnightBossParallelEnv()
    env.reset()
    with pytest.raises(ValueError, match="Invalid action"):
        env.step({"knight": 7, "boss": 6})
    observations, _ = env.reset(options={"alpha_h": 2.0, "alpha_d": 0.5})
    assert observations["knight"]["health"][3] == 300
    assert observations["knight"]["combat_stats"][6] == 10
def test_observations_use_float32_arrays():
    env = KnightBossParallelEnv()
    observations, _ = env.reset()
    assert all(
        value.dtype == np.float32
        for observation in observations.values()
        for value in observation.values()
    )


def test_parallel_api_compatibility():
    env = KnightBossParallelEnv(
        GameConfig(max_steps=10, knight_position=(0, 0), boss_position=(8, 8))
    )
    parallel_api_test(env, num_cycles=20)


def test_terminal_reward_mode_remains_default_and_shaped_mode_is_zero_sum():
    config = GameConfig(knight_position=(4, 4), boss_position=(4, 5))
    default_env = KnightBossParallelEnv(config)
    assert default_env.reward_mode == "terminal"
    default_env.reset(seed=1)
    _, default_rewards, _, _, _ = default_env.step({"knight": 4, "boss": 6})
    assert default_rewards == {"knight": 0.0, "boss": 0.0}

    shaped_env = KnightBossParallelEnv(config, reward_mode="shaped", shaping_weight=0.5)
    shaped_env.reset(seed=1)
    _, shaped_rewards_step, _, _, _ = shaped_env.step({"knight": 4, "boss": 6})
    assert shaped_rewards_step["knight"] > 0.0
    assert shaped_rewards_step["knight"] + shaped_rewards_step["boss"] == pytest.approx(0.0)


def test_invalid_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        KnightBossParallelEnv(reward_mode="unknown")
