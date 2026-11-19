"""Typed game configuration and YAML loading."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from .board import Board
from .types import Position


@dataclass(frozen=True)
class FighterConfig:
    max_health: int = 150
    basic_damage: int = 20
    heavy_damage: int = 40
    basic_range: int = 1
    heavy_range: int = 2
    basic_cooldown: int = 1
    heavy_cooldown: int = 3

    def __post_init__(self) -> None:
        if self.max_health <= 0:
            raise ValueError("max_health must be positive.")
        if self.basic_damage <= 0 or self.heavy_damage <= 0:
            raise ValueError("Attack damage must be positive.")
        if self.basic_range < 0 or self.heavy_range < 1:
            raise ValueError("Attack ranges must be non-negative.")
        if self.basic_cooldown < 0 or self.heavy_cooldown < 0:
            raise ValueError("Cooldowns must be non-negative.")


@dataclass(frozen=True)
class FighterMechanicsConfig:
    basic_range: int = 1
    heavy_range: int = 2
    basic_cooldown: int = 1
    heavy_cooldown: int = 3

    def __post_init__(self) -> None:
        if self.basic_range < 0 or self.heavy_range < 1:
            raise ValueError("Attack ranges must be non-negative.")
        if self.basic_cooldown < 0 or self.heavy_cooldown < 0:
            raise ValueError("Cooldowns must be non-negative.")


@dataclass(frozen=True)
class GameConfig:
    rows: int = 9
    columns: int = 9
    knight_position: Position = (4, 1)
    boss_position: Position = (4, 7)
    knight: FighterConfig = field(default_factory=FighterConfig)
    boss: FighterMechanicsConfig = field(default_factory=FighterMechanicsConfig)
    alpha_h: float = 1.0
    alpha_d: float = 1.0
    max_steps: int = 300
    heavy_windup_steps: int = 1
    proposal_knight_basic_attack_chebyshev_range: int = 2

    def __post_init__(self) -> None:
        if self.rows < 2 or self.columns < 2:
            raise ValueError("The board must have at least two rows and columns.")
        if (
            not isfinite(self.alpha_h)
            or not isfinite(self.alpha_d)
            or self.alpha_h <= 0
            or self.alpha_d <= 0
        ):
            raise ValueError("alpha_h and alpha_d must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.heavy_windup_steps < 1:
            raise ValueError("heavy_windup_steps must be at least one.")
        if self.proposal_knight_basic_attack_chebyshev_range < 0:
            raise ValueError("Proposal Knight attack range must be non-negative.")
        board = Board(self.rows, self.columns)
        if not board.in_bounds(self.knight_position):
            raise ValueError("Knight position must be within the board.")
        if not board.in_bounds(self.boss_position):
            raise ValueError("Boss position must be within the board.")
        if self.knight_position == self.boss_position:
            raise ValueError("Fighters cannot start in the same cell.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GameConfig":
        """Load a game configuration from a YAML mapping."""
        with Path(path).open(encoding="utf-8") as config_file:
            data: Any = yaml.safe_load(config_file)
        if not isinstance(data, dict):
            raise ValueError("Game configuration must be a YAML mapping.")
        board = data.get("board", {})
        positions = data.get("initial_positions", {})
        rules = data.get("rules", {})
        proposal = data.get("proposal", {})
        fighters = data.get("fighters", {})
        knight_data = fighters.get("knight", {})
        boss_data = fighters.get("boss", {})
        return cls(
            rows=int(board.get("rows", 9)),
            columns=int(board.get("columns", 9)),
            knight_position=tuple(positions.get("knight", (4, 1))),
            boss_position=tuple(positions.get("boss", (4, 7))),
            knight=FighterConfig(**knight_data),
            boss=FighterMechanicsConfig(**boss_data),
            alpha_h=float(data.get("alpha_h", 1.0)),
            alpha_d=float(data.get("alpha_d", 1.0)),
            max_steps=int(rules.get("max_steps", 300)),
            heavy_windup_steps=int(rules.get("heavy_windup_steps", 1)),
            proposal_knight_basic_attack_chebyshev_range=int(
                proposal.get("knight_basic_attack_chebyshev_range", 2)
            ),
        )
