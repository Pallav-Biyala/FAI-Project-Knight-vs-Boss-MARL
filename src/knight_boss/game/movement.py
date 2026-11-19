"""Deterministic simultaneous movement and collision resolution."""

from collections.abc import Mapping

from .board import Board
from .fighter import Fighter
from .types import Action, Position

_DELTAS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


def movement_destination(position: Position, action: Action, board: Board) -> Position:
    delta = _DELTAS.get(action)
    if delta is None:
        return position
    candidate = (position[0] + delta[0], position[1] + delta[1])
    return candidate if board.in_bounds(candidate) else position


def resolve_movement(
    fighters: Mapping[str, Fighter],
    actions: Mapping[str, Action],
    board: Board,
) -> dict[str, Position]:
    """Resolve both movement intents from their common pre-move positions.

    Collisions into the same cell and direct exchanges cancel both moves.
    A move into an opponent's cell is rejected unless that opponent can
    complete a non-conflicting move away; with two fighters, the latter
    case is either a valid move to a third cell or a rejected exchange.
    """
    names = tuple(fighters)
    starts = {name: fighters[name].position for name in names}
    intents = {
        name: movement_destination(starts[name], actions[name], board)
        for name in names
    }

    first, second = names
    if intents[first] == intents[second] and intents[first] != starts[first]:
        return starts
    if intents[first] == starts[second] and intents[second] == starts[first]:
        return starts

    resolved = dict(intents)
    for name, opponent in ((first, second), (second, first)):
        if resolved[name] == starts[opponent] and resolved[opponent] == starts[opponent]:
            resolved[name] = starts[name]
    return resolved
