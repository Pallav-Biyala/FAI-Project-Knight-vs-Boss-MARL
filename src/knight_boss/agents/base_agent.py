"""Observation-based agent interface and legal-action calculation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from knight_boss.game.types import Action, Position

Observation = Mapping[str, np.ndarray]

_MOVE_DELTAS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


def observation_positions(
    observation: Observation, rows: int, columns: int
) -> tuple[Position, Position]:
    values = observation["positions"]
    own = (
        round(float(values[0]) * max(1, rows - 1)),
        round(float(values[1]) * max(1, columns - 1)),
    )
    opponent = (
        round(float(values[2]) * max(1, rows - 1)),
        round(float(values[3]) * max(1, columns - 1)),
    )
    return own, opponent


def legal_actions(
    observation: Observation, *, rows: int = 9, columns: int = 9
) -> tuple[Action, ...]:
    """Return actions valid in the observed state.

    Movement into the opponent's current cell is excluded because the policy
    cannot know whether the opponent will move away simultaneously.
    """
    own, opponent = observation_positions(observation, rows, columns)
    stats = observation["combat_stats"]
    cooldowns = observation["cooldowns"]
    distance = abs(own[0] - opponent[0]) + abs(own[1] - opponent[1])
    actions = [Action.WAIT]

    for action, (row_delta, column_delta) in _MOVE_DELTAS.items():
        candidate = (own[0] + row_delta, own[1] + column_delta)
        if (
            0 <= candidate[0] < rows
            and 0 <= candidate[1] < columns
            and candidate != opponent
        ):
            actions.append(action)

    if cooldowns[0] == 0 and distance <= stats[2]:
        actions.append(Action.BASIC_ATTACK)
    row_aligned = own[0] == opponent[0]
    column_aligned = own[1] == opponent[1]
    own_heavy_pending = observation["windups"][0] > 0
    if (
        cooldowns[1] == 0
        and not own_heavy_pending
        and (row_aligned or column_aligned)
        and distance <= stats[3]
    ):
        actions.append(Action.HEAVY_ATTACK)
    return tuple(actions)


class BaseAgent(ABC):
    def __init__(self, *, rows: int = 9, columns: int = 9, seed: int | None = None):
        self.rows = rows
        self.columns = columns
        self.seed = seed
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> None:
        self.seed = seed

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Select an action using only this agent's observation."""
