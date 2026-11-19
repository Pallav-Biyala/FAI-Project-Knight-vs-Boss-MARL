"""Standalone simultaneous-action combat engine."""

from collections.abc import Mapping
from dataclasses import dataclass

from .board import Board
from .combat import (
    PendingHeavyAttack,
    heavy_target_is_valid,
)
from .config import GameConfig
from .fighter import Fighter
from .movement import resolve_movement
from .types import Action, Outcome, StepResult

AGENTS = ("knight", "boss")


@dataclass(frozen=True)
class EpisodeStats:
    damage_dealt: dict[str, int]
    damage_received: dict[str, int]
    steps: int


class GameEngine:
    """A two-fighter game engine with simultaneous action resolution.

    Cooldowns are checked at the start of a step. Existing cooldowns tick
    once at the end; cooldowns started during that step do not tick until the
    following step. Heavy attacks commit to a cardinal direction at announce
    time and resolve after the configured number of opponent reaction steps
    if the target remains in that direction and within range.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        alpha_h: float | None = None,
        alpha_d: float | None = None,
    ) -> None:
        self.config = config or GameConfig()
        self.alpha_h = self.config.alpha_h if alpha_h is None else alpha_h
        self.alpha_d = self.config.alpha_d if alpha_d is None else alpha_d
        if self.alpha_h <= 0 or self.alpha_d <= 0:
            raise ValueError("alpha_h and alpha_d must be positive.")
        self.board = Board(self.config.rows, self.config.columns)
        self.reset()

    def reset(self) -> None:
        self.fighters = {
            "knight": Fighter.from_config(
                "Knight", self.config.knight, self.config.knight_position
            ),
            "boss": Fighter(
                name="Boss",
                max_health=max(1, round(self.config.knight.max_health * self.alpha_h)),
                position=self.config.boss_position,
                basic_damage=max(1, round(self.config.knight.basic_damage * self.alpha_d)),
                heavy_damage=max(1, round(self.config.knight.heavy_damage * self.alpha_d)),
                basic_range=self.config.boss.basic_range,
                heavy_range=self.config.boss.heavy_range,
                basic_cooldown_duration=self.config.boss.basic_cooldown,
                heavy_cooldown_duration=self.config.boss.heavy_cooldown,
            ),
        }
        self.pending_heavy: dict[str, PendingHeavyAttack] = {}
        self.step_count = 0
        self.damage_dealt = {agent: 0 for agent in AGENTS}
        self.damage_received = {agent: 0 for agent in AGENTS}
        self.outcome: Outcome | None = None
        self.terminated = False
        self.truncated = False

    def step(self, actions: Mapping[str, int | Action]) -> StepResult:
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step a completed episode; call reset().")
        if set(actions) != set(AGENTS):
            raise ValueError(f"Actions must be supplied for exactly {AGENTS}.")
        parsed = {agent: self._parse_action(actions[agent]) for agent in AGENTS}
        cooldowns_at_start = {
            agent: (
                self.fighters[agent].basic_cooldown,
                self.fighters[agent].heavy_cooldown,
            )
            for agent in AGENTS
        }
        positions_at_start = {
            agent: self.fighters[agent].position for agent in AGENTS
        }
        pending_at_start = dict(self.pending_heavy)
        events: list[str] = []

        positions = resolve_movement(self.fighters, parsed, self.board)
        for agent, position in positions.items():
            if position != self.fighters[agent].position:
                events.append(f"{agent}_moved")
            self.fighters[agent].position = position

        damage = {agent: 0 for agent in AGENTS}
        newly_started: dict[str, set[str]] = {agent: set() for agent in AGENTS}

        for attacker_name, pending in pending_at_start.items():
            if pending.remaining_steps > 1:
                self.pending_heavy[attacker_name] = PendingHeavyAttack(
                    attacker=pending.attacker,
                    target=pending.target,
                    direction=pending.direction,
                    remaining_steps=pending.remaining_steps - 1,
                )
                continue
            attacker = self.fighters[attacker_name]
            target = self.fighters[pending.target]
            if heavy_target_is_valid(attacker, target, pending.direction, self.board):
                damage[pending.target] += attacker.heavy_damage
                events.append(f"{attacker_name}_heavy_hit")
            else:
                events.append(f"{attacker_name}_heavy_missed")
            self.pending_heavy.pop(attacker_name, None)

        for attacker_name in AGENTS:
            target_name = self._opponent(attacker_name)
            attacker = self.fighters[attacker_name]
            target = self.fighters[target_name]
            action = parsed[attacker_name]
            basic_cd, heavy_cd = cooldowns_at_start[attacker_name]

            if action == Action.BASIC_ATTACK and basic_cd == 0:
                if self.board.manhattan_distance(attacker.position, target.position) <= attacker.basic_range:
                    damage[target_name] += attacker.basic_damage
                    attacker.basic_cooldown = attacker.basic_cooldown_duration
                    newly_started[attacker_name].add("basic")
                    events.append(f"{attacker_name}_basic_hit")
                else:
                    events.append(f"{attacker_name}_basic_out_of_range")
            elif (
                action == Action.HEAVY_ATTACK
                and heavy_cd == 0
                and attacker_name not in pending_at_start
            ):
                attacker_position = positions_at_start[attacker_name]
                target_position = positions_at_start[target_name]
                direction = self.board.direction_between(
                    attacker_position, target_position
                )
                distance = self.board.manhattan_distance(
                    attacker_position, target_position
                )
                if distance > attacker.heavy_range:
                    direction = None
                if direction is not None:
                    self.pending_heavy[attacker_name] = PendingHeavyAttack(
                        attacker=attacker_name,
                        target=target_name,
                        direction=direction,
                        remaining_steps=self.config.heavy_windup_steps,
                    )
                    attacker.heavy_cooldown = attacker.heavy_cooldown_duration
                    newly_started[attacker_name].add("heavy")
                    events.append(f"{attacker_name}_heavy_windup")
                else:
                    events.append(f"{attacker_name}_heavy_out_of_range")

        for target_name, amount in damage.items():
            attacker_name = self._opponent(target_name)
            actual = self.fighters[target_name].receive_damage(amount)
            self.damage_dealt[attacker_name] += actual
            self.damage_received[target_name] += actual

        for agent in AGENTS:
            self._tick_cooldowns(agent, cooldowns_at_start[agent], newly_started[agent])

        self.step_count += 1
        self._update_result()
        return StepResult(
            terminated=self.terminated,
            truncated=self.truncated,
            outcome=self.outcome,
            events=tuple(events),
        )

    def _tick_cooldowns(
        self,
        agent: str,
        initial: tuple[int, int],
        newly_started: set[str],
    ) -> None:
        fighter = self.fighters[agent]
        if "basic" not in newly_started:
            fighter.basic_cooldown = max(0, initial[0] - 1)
        if "heavy" not in newly_started:
            fighter.heavy_cooldown = max(0, initial[1] - 1)

    def _update_result(self) -> None:
        knight_alive = self.fighters["knight"].health > 0
        boss_alive = self.fighters["boss"].health > 0
        if not knight_alive and not boss_alive:
            self.outcome = Outcome.DRAW
            self.terminated = True
        elif not boss_alive:
            self.outcome = Outcome.KNIGHT_WIN
            self.terminated = True
        elif not knight_alive:
            self.outcome = Outcome.BOSS_WIN
            self.terminated = True
        elif self.step_count >= self.config.max_steps:
            self.outcome = Outcome.DRAW
            self.truncated = True

    @staticmethod
    def _opponent(agent: str) -> str:
        return "boss" if agent == "knight" else "knight"

    @staticmethod
    def _parse_action(action: int | Action) -> Action:
        try:
            return Action(action)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid action: {action!r}") from error

    @property
    def stats(self) -> EpisodeStats:
        return EpisodeStats(
            damage_dealt=dict(self.damage_dealt),
            damage_received=dict(self.damage_received),
            steps=self.step_count,
        )
