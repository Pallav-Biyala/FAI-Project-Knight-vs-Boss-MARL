from pathlib import Path

import numpy as np
import pytest
import torch

from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.game.config import GameConfig
from knight_boss.game.types import Action
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.training.checkpointing import load_checkpoint, save_checkpoint
from knight_boss.training.config import PPOConfig
from knight_boss.training.rollout import RolloutBuffer
from knight_boss.training.trainer import MultiAgentPPOTrainer


def small_ppo_config(**overrides):
    values = {
        "learning_rate": 0.001,
        "gamma": 0.9,
        "gae_lambda": 0.8,
        "clip_range": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "epochs": 2,
        "batch_size": 4,
        "rollout_steps": 8,
        "hidden_sizes": (16,),
        "max_grad_norm": 0.5,
        "device": "cpu",
    }
    values.update(overrides)
    return PPOConfig(**values)


def test_observation_vector_has_stable_fixed_size():
    env = KnightBossParallelEnv()
    observations, _ = env.reset()
    vector = observation_to_vector(observations["knight"])
    assert vector.dtype == np.float32
    assert vector.shape == (35,)


def test_gae_respects_terminal_boundaries_and_bootstraps_nonterminal_steps():
    buffer = RolloutBuffer()
    buffer.add(np.zeros(2), 1, 0.0, False, 0.5, -0.2)
    buffer.add(np.ones(2), 2, 1.0, True, 0.25, -0.3)
    batch = buffer.compute(last_value=99.0, gamma=0.9, gae_lambda=0.8)
    assert batch.advantages == pytest.approx([0.265, 0.75])
    assert batch.returns == pytest.approx([0.765, 1.0])


def test_terminal_rewards_are_associated_with_each_agents_own_rollout():
    config = small_ppo_config(batch_size=1, rollout_steps=1)
    env = KnightBossParallelEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        alpha_h=0.05,
    )
    trainer = MultiAgentPPOTrainer(env, config, seed=10)
    for agent, action in (("knight", Action.BASIC_ATTACK), ("boss", Action.WAIT)):
        policy = trainer.policies[agent].policy
        with torch.no_grad():
            policy.policy_head.weight.zero_()
            policy.policy_head.bias.fill_(-100.0)
            policy.policy_head.bias[int(action)] = 100.0
    batches = trainer.collect_rollout(1)
    assert batches["knight"].rewards.tolist() == [1.0]
    assert batches["boss"].rewards.tolist() == [-1.0]
    assert batches["knight"].dones.tolist() == [True]
    assert batches["boss"].dones.tolist() == [True]


def test_trainable_agent_updates_without_changing_frozen_policy_or_optimizer():
    trainer = MultiAgentPPOTrainer(
        KnightBossParallelEnv(), small_ppo_config(), seed=21
    )
    knight = trainer.policies["knight"]
    boss = trainer.policies["boss"]
    trainer.train_rollout(8)
    knight_before = {
        key: value.clone() for key, value in knight.policy.state_dict().items()
    }
    boss_before = {
        key: value.clone() for key, value in boss.policy.state_dict().items()
    }
    boss_optimizer_before = boss.optimizer.state_dict()
    metrics = trainer.train_rollout(8, trainable_agents=("knight",))
    assert set(metrics) == {"knight"}
    assert knight.update_count == 2
    assert boss.update_count == 1
    assert any(
        not torch.equal(value, knight_before[key])
        for key, value in knight.policy.state_dict().items()
    )
    assert all(
        torch.equal(value, boss_before[key])
        for key, value in boss.policy.state_dict().items()
    )
    boss_optimizer_after = boss.optimizer.state_dict()
    assert boss_optimizer_after["param_groups"] == boss_optimizer_before["param_groups"]
    assert boss_optimizer_after["state"].keys() == boss_optimizer_before["state"].keys()
    for parameter_id, state in boss_optimizer_before["state"].items():
        for key, value in state.items():
            after = boss_optimizer_after["state"][parameter_id][key]
            assert torch.equal(value, after) if isinstance(value, torch.Tensor) else value == after


def test_joint_rollout_collects_both_before_environment_step():
    config = small_ppo_config(rollout_steps=1)
    env = KnightBossParallelEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5))
    )
    trainer = MultiAgentPPOTrainer(env, config, seed=4)
    initial = {
        agent: observation_to_vector(value)
        for agent, value in trainer.observations.items()
    }
    batches = trainer.collect_rollout(1)
    assert set(batches) == {"knight", "boss"}
    for batch in batches.values():
        assert batch.observations.shape == (1, 35)
    assert all(initial[agent].shape == (35,) for agent in ("knight", "boss"))
    assert trainer.total_environment_steps == 1


def test_checkpoint_round_trip_restores_parameters_optimizer_and_rng(tmp_path):
    config = small_ppo_config()
    observation = KnightBossParallelEnv().reset(seed=1)[0]["knight"]
    vector_size = observation_to_vector(observation).size
    source = PPOAgent(vector_size, config, seed=5)
    source.update(
        observations=np.stack([observation_to_vector(observation)] * 4),
        actions=np.array([0, 1, 2, 3]),
        old_log_probabilities=np.zeros(4, dtype=np.float32),
        advantages=np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32),
        returns=np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32),
    )
    path = tmp_path / "knight.pt"
    save_checkpoint(source, path, metadata={"regime": "test"})
    expected_next_action = source.act_with_info(observation)

    restored = PPOAgent(vector_size, config, seed=999)
    metadata = load_checkpoint(restored, path)
    actual_next_action = restored.act_with_info(observation)
    assert metadata == {"regime": "test"}
    assert restored.update_count == source.update_count
    assert expected_next_action == actual_next_action
    assert all(
        torch.equal(value, restored.policy.state_dict()[key])
        for key, value in source.policy.state_dict().items()
    )
    assert restored.optimizer.state_dict()["state"].keys() == source.optimizer.state_dict()["state"].keys()


def test_checkpoint_rejects_incompatible_policy_architecture(tmp_path):
    observation_size = 35
    source = PPOAgent(observation_size, small_ppo_config(), seed=2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(source, path)
    incompatible = PPOAgent(
        observation_size,
        small_ppo_config(hidden_sizes=(32,)),
        seed=3,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(incompatible, path)


def test_ppo_yaml_loads_configured_architecture():
    config = PPOConfig.from_yaml(Path("configs") / "ppo.yaml")
    assert config.learning_rate == pytest.approx(0.0003)
    assert config.hidden_sizes == (128, 128)
    assert config.rollout_steps == 1024


def test_invalid_ppo_configuration_is_rejected():
    with pytest.raises(ValueError, match="gamma and gae_lambda"):
        small_ppo_config(gamma=1.1)


def test_ppo_config_defaults_match_yaml():
    assert PPOConfig() == PPOConfig.from_yaml(Path("configs") / "ppo.yaml")


def test_ppo_agent_reset_reseeds_sampling():
    observation = KnightBossParallelEnv().reset(seed=1)[0]["knight"]
    agent = PPOAgent(observation_to_vector(observation).size, PPOConfig(), seed=5)
    agent.reset(seed=42)
    first_sequence = [agent.act(observation) for _ in range(5)]
    agent.reset(seed=42)
    second_sequence = [agent.act(observation) for _ in range(5)]
    assert first_sequence == second_sequence


def test_ppo_agent_act_matches_act_with_info():
    observation = KnightBossParallelEnv().reset(seed=2)[0]["knight"]
    agent = PPOAgent(observation_to_vector(observation).size, PPOConfig(), seed=9)
    agent.reset(seed=77)
    act_actions = [int(agent.act(observation)) for _ in range(5)]
    agent.reset(seed=77)
    info_actions = [agent.act_with_info(observation)[0] for _ in range(5)]
    assert act_actions == info_actions


def test_partial_tail_rollout_updates_cleanly():
    config = small_ppo_config(batch_size=8, rollout_steps=16, epochs=4)
    observation = KnightBossParallelEnv().reset(seed=1)[0]["knight"]
    vec = observation_to_vector(observation)
    agent = PPOAgent(vec.size, config, seed=11)
    # 9 steps: batch_size=8 with 1 trailing sample should run cleanly without crashing
    metrics = agent.update(
        observations=np.stack([vec] * 9),
        actions=np.zeros(9, dtype=np.int64),
        old_log_probabilities=np.zeros(9, dtype=np.float32),
        advantages=np.linspace(-1.0, 1.0, 9, dtype=np.float32),
        returns=np.linspace(-1.0, 1.0, 9, dtype=np.float32),
    )
    assert agent.update_count == 1
    assert all(np.isfinite(val) for val in metrics.values())


def test_observation_vector_is_scaled_to_small_magnitudes():
    from knight_boss.marl.proposal_env import ProposalCombatEnv

    env = ProposalCombatEnv(alpha_h=3.0, alpha_d=3.0)
    observations, _ = env.reset(seed=0)
    for agent in ("knight", "boss"):
        vector = observation_to_vector(observations[agent])
        assert np.max(np.abs(vector)) <= 3.5
