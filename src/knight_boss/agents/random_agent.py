"""Seeded random policy over actions legal in the observed state."""

import numpy as np

from knight_boss.game.types import Action
from knight_boss.agents.base_agent import BaseAgent, Observation, legal_actions


class RandomAgent(BaseAgent):
    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Observation) -> Action:
        available = legal_actions(
            observation, rows=self.rows, columns=self.columns
        )
        return available[int(self._rng.integers(len(available)))]
