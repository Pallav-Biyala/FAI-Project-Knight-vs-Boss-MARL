"""Historical opponent population for population-based self-play."""

from __future__ import annotations

import copy
import random

from knight_boss.agents.ppo_agent import PPOAgent


class OpponentPool:
    """Stores frozen historical PPO opponents and samples from them."""

    def __init__(
        self,
        max_size: int = 10,
        sampling: str = "uniform",
        seed: int | None = None,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1.")

        if sampling != "uniform":
            raise ValueError("Currently, only uniform sampling is supported.")

        self.max_size = max_size
        self.sampling = sampling
        self.rng = random.Random(seed)
        self._opponents: list[PPOAgent] = []

    def add(self, agent: PPOAgent) -> None:
        """Store an independent frozen copy of an agent."""
        snapshot = copy.deepcopy(agent)
        snapshot.policy.eval()

        self._opponents.append(snapshot)

        if len(self._opponents) > self.max_size:
            self._opponents.pop(0)

    def sample(self) -> PPOAgent:
        """Sample one historical opponent uniformly."""
        if not self._opponents:
            raise RuntimeError("Cannot sample from an empty opponent pool.")

        return self.rng.choice(self._opponents)

    def __len__(self) -> int:
        return len(self._opponents)

    def __repr__(self) -> str:
        return (
            f"OpponentPool(size={len(self)}, "
            f"max_size={self.max_size}, sampling={self.sampling!r})"
        )