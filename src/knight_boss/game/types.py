"""Shared game types and action definitions."""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TypeAlias

Position: TypeAlias = tuple[int, int]


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    BASIC_ATTACK = 4
    HEAVY_ATTACK = 5
    WAIT = 6


class Outcome(str, Enum):
    KNIGHT_WIN = "knight_win"
    BOSS_WIN = "boss_win"
    DRAW = "draw"


@dataclass(frozen=True)
class StepResult:
    """Outcome of one simultaneous game step."""

    terminated: bool
    truncated: bool
    outcome: Outcome | None
    events: tuple[str, ...]
