"""Attack geometry, wind-up records, and cooldown updates."""

from dataclasses import dataclass

from .board import Board
from .fighter import Fighter
from .types import Position


@dataclass(frozen=True)
class PendingHeavyAttack:
    attacker: str
    target: str
    direction: Position
    remaining_steps: int


def heavy_target_is_valid(
    attacker: Fighter, target: Fighter, direction: Position, board: Board
) -> bool:
    current_direction = board.direction_between(attacker.position, target.position)
    if current_direction != direction:
        return False
    return (
        board.manhattan_distance(attacker.position, target.position)
        <= attacker.heavy_range
    )
