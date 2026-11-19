"""Atomic per-agent PPO checkpoint persistence."""

from pathlib import Path
import os
import tempfile
from typing import Any

import torch

from knight_boss.agents.ppo_agent import PPOAgent


def save_checkpoint(
    agent: PPOAgent,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "observation_size": agent.observation_size,
        "n_actions": agent.n_actions,
        "hidden_sizes": agent.config.hidden_sizes,
        "policy_state": agent.policy.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "update_count": agent.update_count,
        "torch_rng_state": agent._torch_rng.get_state(),
        "numpy_rng_state": agent._numpy_rng.bit_generator.state,
        "metadata": metadata or {},
    }
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = temporary.name
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_checkpoint(
    agent: PPOAgent,
    path: str | Path,
    *,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(
        source, map_location=agent.device, weights_only=False
    )
    expected = (
        agent.observation_size,
        agent.n_actions,
        agent.config.hidden_sizes,
    )
    actual = (
        checkpoint["observation_size"],
        checkpoint["n_actions"],
        tuple(checkpoint["hidden_sizes"]),
    )
    if actual != expected:
        raise ValueError(
            f"Checkpoint architecture {actual} does not match agent {expected}."
        )
    agent.policy.load_state_dict(checkpoint["policy_state"])
    if load_optimizer:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
    agent.update_count = int(checkpoint["update_count"])
    agent._torch_rng.set_state(checkpoint["torch_rng_state"].cpu())
    agent._numpy_rng.bit_generator.state = checkpoint["numpy_rng_state"]
    return checkpoint["metadata"]
