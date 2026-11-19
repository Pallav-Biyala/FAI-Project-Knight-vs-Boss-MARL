"""Grid geometry and boundary checks."""

from dataclasses import dataclass

from .types import Position


@dataclass(frozen=True)
class Board:
    rows: int = 9
    columns: int = 9

    def __post_init__(self) -> None:
        if self.rows < 2 or self.columns < 2:
            raise ValueError("The board must have at least two rows and columns.")

    def in_bounds(self, position: Position) -> bool:
        row, column = position
        return 0 <= row < self.rows and 0 <= column < self.columns

    @staticmethod
    def manhattan_distance(first: Position, second: Position) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def direction_between(start: Position, target: Position) -> Position | None:
        """Return a unit cardinal direction if target lies on a cardinal ray."""
        row_delta = target[0] - start[0]
        column_delta = target[1] - start[1]
        if row_delta == 0 and column_delta != 0:
            return (0, 1 if column_delta > 0 else -1)
        if column_delta == 0 and row_delta != 0:
            return (1 if row_delta > 0 else -1, 0)
        return None
