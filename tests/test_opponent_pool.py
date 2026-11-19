import pytest
import torch

from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.training.config import PPOConfig
from knight_boss.training.opponent_pool import OpponentPool


def tiny_ppo_config():
    return PPOConfig(
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=1,
        batch_size=1,
        rollout_steps=8,
        hidden_sizes=(8,),
        max_grad_norm=0.5,
        device="cpu",
    )


def make_agent(seed):
    return PPOAgent(
        observation_size=10,
        config=tiny_ppo_config(),
        seed=seed,
    )


def test_pool_stores_independent_frozen_snapshot():
    agent = make_agent(1)
    pool = OpponentPool(max_size=3, seed=10)

    pool.add(agent)
    snapshot = pool.sample()

    assert snapshot is not agent
    assert not snapshot.policy.training

    original_parameter = next(agent.policy.parameters())
    snapshot_parameter = next(snapshot.policy.parameters())

    assert original_parameter.data_ptr() != snapshot_parameter.data_ptr()

    with torch.no_grad():
        original_parameter.add_(1.0)

    assert not torch.equal(original_parameter, snapshot_parameter)


def test_pool_evicts_oldest_snapshot_when_full():
    pool = OpponentPool(max_size=2, seed=10)

    first = make_agent(1)
    second = make_agent(2)
    third = make_agent(3)

    pool.add(first)
    pool.add(second)
    first_snapshot = pool._opponents[0]

    pool.add(third)

    assert len(pool) == 2
    assert first_snapshot not in pool._opponents
    assert pool._opponents[0] is not second
    assert pool._opponents[1] is not third


def test_pool_sampling_returns_stored_opponent():
    pool = OpponentPool(max_size=3, seed=10)

    pool.add(make_agent(1))
    pool.add(make_agent(2))

    for _ in range(10):
        assert pool.sample() in pool._opponents


def test_sampling_empty_pool_raises_error():
    pool = OpponentPool(max_size=3)

    with pytest.raises(RuntimeError, match="empty opponent pool"):
        pool.sample()


def test_pool_rejects_invalid_size_and_sampling_strategy():
    with pytest.raises(ValueError, match="max_size"):
        OpponentPool(max_size=0)

    with pytest.raises(ValueError, match="sampling"):
        OpponentPool(sampling="elo")
