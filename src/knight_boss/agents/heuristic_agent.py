"""Deterministic rule-based opponent policy."""

from knight_boss.game.types import Action
from knight_boss.agents.base_agent import (
    BaseAgent,
    Observation,
    legal_actions,
    observation_positions,
)


class HeuristicAgent(BaseAgent):
    def act(self, observation: Observation) -> Action:
        available = legal_actions(
            observation, rows=self.rows, columns=self.columns
        )
        own, opponent = observation_positions(
            observation, self.rows, self.columns
        )
        distance = abs(own[0] - opponent[0]) + abs(own[1] - opponent[1])
        stats = observation["combat_stats"]
        basic_available = Action.BASIC_ATTACK in available
        heavy_available = Action.HEAVY_ATTACK in available

        if basic_available:
            return Action.BASIC_ATTACK
        if heavy_available and (
            distance > stats[2] or observation["cooldowns"][0] > 0
        ):
            return Action.HEAVY_ATTACK

        moves = [action for action in available if action in _MOVES]
        if not moves:
            return Action.WAIT
        return min(
            moves,
            key=lambda action: (
                self._distance_after(own, opponent, action),
                int(action),
            ),
        )

    @staticmethod
    def _distance_after(own: tuple[int, int], opponent: tuple[int, int], action: Action) -> int:
        row_delta, column_delta = _MOVES[action]
        candidate = (own[0] + row_delta, own[1] + column_delta)
        return abs(candidate[0] - opponent[0]) + abs(candidate[1] - opponent[1])


_MOVES = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}
