"""Fighter state and configurable combat statistics."""

from dataclasses import dataclass, field

from .config import FighterConfig
from .types import Position


@dataclass
class Fighter:
    name: str
    max_health: int
    position: Position
    basic_damage: int
    heavy_damage: int
    basic_range: int
    heavy_range: int
    basic_cooldown_duration: int
    heavy_cooldown_duration: int
    health: int = field(init=False)
    basic_cooldown: int = 0
    heavy_cooldown: int = 0

    def __post_init__(self) -> None:
        if self.max_health <= 0:
            raise ValueError("max_health must be positive.")
        self.health = self.max_health

    @classmethod
    def from_config(
        cls, name: str, config: FighterConfig, position: Position, *,
        health_multiplier: float = 1.0, damage_multiplier: float = 1.0,
    ) -> "Fighter":
        max_health = max(1, round(config.max_health * health_multiplier))
        return cls(
            name=name,
            max_health=max_health,
            position=position,
            basic_damage=max(1, round(config.basic_damage * damage_multiplier)),
            heavy_damage=max(1, round(config.heavy_damage * damage_multiplier)),
            basic_range=config.basic_range,
            heavy_range=config.heavy_range,
            basic_cooldown_duration=config.basic_cooldown,
            heavy_cooldown_duration=config.heavy_cooldown,
        )

    def receive_damage(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("Damage cannot be negative.")
        actual = min(self.health, amount)
        self.health -= actual
        return actual
