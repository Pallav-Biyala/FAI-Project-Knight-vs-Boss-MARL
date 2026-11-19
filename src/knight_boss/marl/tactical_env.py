"""Opt-in tactical extension adding DODGE and HEAL mechanics cleanly.

Preserves the base 7-action environment untouched while enabling controlled
2x2 factorial ablations over DODGE (1-step evasion) and HEAL (charge-limited
HP restoration).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gymnasium import spaces
import numpy as np

from knight_boss.agents.heuristic_agent import HeuristicAgent
from knight_boss.game.config import GameConfig
from knight_boss.game.engine import AGENTS
from knight_boss.game.types import Action
from knight_boss.marl.parallel_env import KnightBossParallelEnv


class TacticalParallelEnv(KnightBossParallelEnv):
    """Parallel combat environment extending the 7 base actions with DODGE / HEAL."""

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        alpha_h: float | None = None,
        alpha_d: float | None = None,
        render_mode: str | None = None,
        reward_mode: str = "terminal",
        shaping_weight: float = 0.5,
        draw_penalty: float = 0.75,
        gamma: float = 0.995,
        enable_dodge: bool = True,
        enable_heal: bool = True,
        dodge_cooldown: int = 3,
        heal_amount: int = 30,
        heal_cooldown: int = 4,
        heal_charges: int = 3,
    ) -> None:
        if dodge_cooldown < 0 or heal_cooldown < 0 or heal_charges < 0 or heal_amount < 0:
            raise ValueError("Tactical cooldowns, charges, and heal_amount must be non-negative.")
        super().__init__(
            config,
            alpha_h=alpha_h,
            alpha_d=alpha_d,
            render_mode=render_mode,
            reward_mode=reward_mode,
            shaping_weight=shaping_weight,
            draw_penalty=draw_penalty,
            gamma=gamma,
        )
        self.enable_dodge = enable_dodge
        self.enable_heal = enable_heal
        self.dodge_cooldown = dodge_cooldown
        self.heal_amount = heal_amount
        self.heal_cooldown = heal_cooldown
        self.heal_charges = heal_charges

        next_id = len(Action)
        self.dodge_action_id: int | None = None
        self.heal_action_id: int | None = None
        if self.enable_dodge:
            self.dodge_action_id = next_id
            next_id += 1
        if self.enable_heal:
            self.heal_action_id = next_id
            next_id += 1
        self._n_actions = next_id

        self._action_spaces = {
            agent: spaces.Discrete(self._n_actions) for agent in self.possible_agents
        }
        for agent in self.possible_agents:
            space_dict = dict(self._observation_spaces[agent].spaces)
            space_dict["tactical_stats"] = spaces.Box(
                low=0.0, high=1.0, shape=(6,), dtype=np.float32
            )
            self._observation_spaces[agent] = spaces.Dict(space_dict)

        self._dodge_cd = {agent: 0 for agent in AGENTS}
        self._heal_cd = {agent: 0 for agent in AGENTS}
        self._heal_charges = {agent: self.heal_charges for agent in AGENTS}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        self._dodge_cd = {agent: 0 for agent in AGENTS}
        self._heal_cd = {agent: 0 for agent in AGENTS}
        self._heal_charges = {agent: self.heal_charges for agent in AGENTS}
        return super().reset(seed=seed, options=options)

    def _observe(self, agent: str) -> dict[str, np.ndarray]:
        obs = super()._observe(agent)
        opponent = "boss" if agent == "knight" else "knight"
        dodge_div = max(1, self.dodge_cooldown)
        heal_div = max(1, self.heal_cooldown)
        charge_div = max(1, self.heal_charges)
        obs["tactical_stats"] = np.array(
            [
                self._dodge_cd[agent] / dodge_div if self.enable_dodge else 0.0,
                self._dodge_cd[opponent] / dodge_div if self.enable_dodge else 0.0,
                self._heal_cd[agent] / heal_div if self.enable_heal else 0.0,
                self._heal_cd[opponent] / heal_div if self.enable_heal else 0.0,
                self._heal_charges[agent] / charge_div if self.enable_heal else 0.0,
                self._heal_charges[opponent] / charge_div if self.enable_heal else 0.0,
            ],
            dtype=np.float32,
        )
        return obs

    def step(
        self, actions: Mapping[str, int | Action]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError(f"Actions must be supplied for active agents {self.agents}.")

        acting_agents = self.agents.copy()
        parsed_ids: dict[str, int] = {}
        for agent in acting_agents:
            action_id = int(actions[agent])
            if action_id < 0 or action_id >= self._n_actions:
                raise ValueError(f"Invalid tactical action {action_id!r} for {agent}.")
            parsed_ids[agent] = action_id

        prev_health = {agent: self.engine.fighters[agent].health for agent in AGENTS}
        prev_rec = {agent: self.engine.damage_received[agent] for agent in AGENTS}
        prev_k_pos = self.engine.fighters["knight"].position
        prev_b_pos = self.engine.fighters["boss"].position
        prev_distance = abs(prev_k_pos[0] - prev_b_pos[0]) + abs(prev_k_pos[1] - prev_b_pos[1])

        dodging = {agent: False for agent in AGENTS}
        healing = {agent: 0 for agent in AGENTS}
        base_actions: dict[str, int | Action] = {}
        tactical_events: list[str] = []

        for agent in AGENTS:
            aid = parsed_ids[agent]
            if self.dodge_action_id is not None and aid == self.dodge_action_id:
                if self._dodge_cd[agent] == 0:
                    dodging[agent] = True
                    self._dodge_cd[agent] = self.dodge_cooldown
                    tactical_events.append(f"{agent}_dodge_stance")
                else:
                    self._dodge_cd[agent] = max(0, self._dodge_cd[agent] - 1)
                self._heal_cd[agent] = max(0, self._heal_cd[agent] - 1)
                base_actions[agent] = Action.WAIT
            elif self.heal_action_id is not None and aid == self.heal_action_id:
                fighter = self.engine.fighters[agent]
                if (
                    self._heal_cd[agent] == 0
                    and self._heal_charges[agent] > 0
                    and fighter.health < fighter.max_health
                ):
                    healing[agent] = self.heal_amount
                    self._heal_charges[agent] -= 1
                    self._heal_cd[agent] = self.heal_cooldown
                    tactical_events.append(f"{agent}_heal_cast")
                else:
                    self._heal_cd[agent] = max(0, self._heal_cd[agent] - 1)
                self._dodge_cd[agent] = max(0, self._dodge_cd[agent] - 1)
                base_actions[agent] = Action.WAIT
            else:
                self._dodge_cd[agent] = max(0, self._dodge_cd[agent] - 1)
                self._heal_cd[agent] = max(0, self._heal_cd[agent] - 1)
                base_actions[agent] = Action(aid)

        for agent in AGENTS:
            if healing[agent] > 0:
                self.engine.fighters[agent].health += healing[agent]

        result = self.engine.step(base_actions)

        # Apply DODGE damage negation and clamp post-HEAL HP cleanly.
        for agent in AGENTS:
            opponent = "boss" if agent == "knight" else "knight"
            fighter = self.engine.fighters[agent]
            dmg_taken = self.engine.damage_received[agent] - prev_rec[agent]
            if dodging[agent] and dmg_taken > 0:
                fighter.health = prev_health[agent]
                self.engine.damage_received[agent] -= dmg_taken
                self.engine.damage_dealt[opponent] -= dmg_taken
                tactical_events.append(f"{agent}_dodged_hit")
            if healing[agent] > 0:
                fighter.health = min(fighter.max_health, fighter.health)

        # Recompute terminal/truncation outcome after dodge & heal HP adjustments.
        self.engine.outcome = None
        self.engine.terminated = False
        self.engine.truncated = False
        self.engine._update_result()

        observations = {agent: self._observe(agent) for agent in acting_agents}
        rewards = self._compute_rewards(
            outcome=self.engine.outcome,
            truncated=self.engine.truncated,
            prev_health=prev_health,
            prev_distance=prev_distance,
        )
        terminations = {agent: self.engine.terminated for agent in acting_agents}
        truncations = {agent: self.engine.truncated for agent in acting_agents}
        combined_events = tuple(result.events) + tuple(tactical_events)
        infos = {
            agent: {
                "outcome": self.engine.outcome.value if self.engine.outcome else None,
                "step_count": self.engine.step_count,
                "events": combined_events,
                "damage_dealt": self.engine.damage_dealt[agent],
                "damage_received": self.engine.damage_received[agent],
            }
            for agent in acting_agents
        }
        if self.engine.terminated or self.engine.truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _compute_rewards(
        self,
        *,
        outcome,
        truncated: bool,
        prev_health: Mapping[str, int],
        prev_distance: int,
    ) -> dict[str, float]:
        from knight_boss.marl.rewards import (
            combat_rewards,
            draw_penalty_rewards,
            shaped_rewards,
            terminal_rewards,
        )

        knight = self.engine.fighters["knight"]
        boss = self.engine.fighters["boss"]

        common = {
            "prev_knight_health": prev_health["knight"],
            "curr_knight_health": knight.health,
            "prev_boss_health": prev_health["boss"],
            "curr_boss_health": boss.health,
            "knight_max_health": knight.max_health,
            "boss_max_health": boss.max_health,
            "shaping_weight": self.shaping_weight,
            "gamma": self.gamma,
            "truncated": truncated,
        }

        if self.reward_mode == "combat":
            curr_k_pos = knight.position
            curr_b_pos = boss.position
            curr_distance = abs(curr_k_pos[0] - curr_b_pos[0]) + abs(
                curr_k_pos[1] - curr_b_pos[1]
            )
            return combat_rewards(
                outcome,
                **common,
                prev_distance=prev_distance,
                curr_distance=curr_distance,
            )

        if self.reward_mode == "draw_penalty":
            return draw_penalty_rewards(
                outcome,
                **common,
                penalty=self.draw_penalty,
            )

        if self.reward_mode == "shaped":
            return shaped_rewards(outcome, **common)

        return terminal_rewards(outcome)


class TacticalHeuristicAgent(HeuristicAgent):
    """Heuristic baseline aware of optional DODGE and HEAL actions."""

    def __init__(
        self,
        *,
        rows: int = 9,
        columns: int = 9,
        seed: int | None = None,
        dodge_action_id: int | None = None,
        heal_action_id: int | None = None,
        dodge_cooldown: int = 3,
        heal_cooldown: int = 4,
        heal_charges: int = 3,
    ) -> None:
        super().__init__(rows=rows, columns=columns, seed=seed)
        self.dodge_action_id = dodge_action_id
        self.heal_action_id = heal_action_id
        self.dodge_cooldown = dodge_cooldown
        self.heal_cooldown = heal_cooldown
        self.heal_charges = heal_charges

    def clone_for_eval(self, *, rows: int, columns: int, seed: int) -> "TacticalHeuristicAgent":
        return TacticalHeuristicAgent(
            rows=rows,
            columns=columns,
            seed=seed,
            dodge_action_id=self.dodge_action_id,
            heal_action_id=self.heal_action_id,
            dodge_cooldown=self.dodge_cooldown,
            heal_cooldown=self.heal_cooldown,
            heal_charges=self.heal_charges,
        )

    def act(self, observation: Mapping[str, np.ndarray]) -> int:
        tactical = observation.get("tactical_stats")
        if tactical is not None:
            own_dodge_cd = float(tactical[0])
            own_heal_cd = float(tactical[2])
            own_charges = float(tactical[4])
            opp_windup_active = float(observation["windups"][4]) > 0.5
            opp_windup_steps = float(observation["windups"][7])
            own_hp_frac = float(observation["health_fraction"][0])

            # Dodge immediately if opponent heavy attack is about to fire (windup == 1 step).
            if (
                self.dodge_action_id is not None
                and own_dodge_cd <= 1e-6
                and opp_windup_active
                and 0.0 < opp_windup_steps <= 1.05
            ):
                # Check if aligned cardinally within heavy range (approximated via position delta)
                own_pos = observation["positions"][:2] * np.array(
                    [max(1, self.rows - 1), max(1, self.columns - 1)], dtype=np.float32
                )
                opp_pos = observation["positions"][2:] * np.array(
                    [max(1, self.rows - 1), max(1, self.columns - 1)], dtype=np.float32
                )
                dr = abs(float(own_pos[0]) - float(opp_pos[0]))
                dc = abs(float(own_pos[1]) - float(opp_pos[1]))
                if (dr < 0.1 and dc <= 2.1) or (dc < 0.1 and dr <= 2.1):
                    return self.dodge_action_id

            # Heal when below 50% HP and off cooldown with charges left.
            if (
                self.heal_action_id is not None
                and own_charges > 1e-6
                and own_heal_cd <= 1e-6
                and own_hp_frac <= 0.50
            ):
                return self.heal_action_id

        return int(super().act(observation))
