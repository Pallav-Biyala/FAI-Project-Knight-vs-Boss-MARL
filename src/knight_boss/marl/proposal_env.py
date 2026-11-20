"""Standalone asymmetric combat environment matching the CS 329 proposal specification.

Preserves the core 7-action benchmark environment completely untouched while
implementing:
- Knight action space: {LEFT, RIGHT, UP, DOWN, ATTACK, WAIT, DODGE, HEAL}
- Boss action space:   {LEFT, RIGHT, UP, DOWN, ATTACK, WAIT, SPECIAL_1, SPECIAL_2}
- Knight invincibility frames (u_K) from DODGE and 3x +30 HP charges (q_K) from HEAL
- Shared Boss cooldown c_B across normal ATTACK, directional SPECIAL_1 beam, and radial SPECIAL_2 shockwave (z_t)
- Scripted heuristic Knight/Boss baselines and PPO/evaluator compatibility
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
import functools
from math import isfinite
from typing import Any

from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from knight_boss.agents.base_agent import BaseAgent, Observation
from knight_boss.game.config import GameConfig
from knight_boss.game.types import Outcome
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.rewards import (
    PROPOSAL_REWARD_MODES,
    combat_rewards,
    draw_penalty_rewards,
    shaped_rewards,
    tactical_rewards,
    terminal_rewards,
)

PROPOSAL_AGENTS = ("knight", "boss")


class ProposalCommonAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ATTACK = 4
    WAIT = 5


class ProposalKnightAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ATTACK = 4
    WAIT = 5
    DODGE = 6
    HEAL = 7


class ProposalBossAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ATTACK = 4
    WAIT = 5
    SPECIAL_1 = 6
    SPECIAL_2 = 7


_MOVE_DELTAS: dict[int, tuple[int, int]] = {
    int(ProposalCommonAction.LEFT): (0, -1),
    int(ProposalCommonAction.RIGHT): (0, 1),
    int(ProposalCommonAction.UP): (-1, 0),
    int(ProposalCommonAction.DOWN): (1, 0),
}


class ProposalCombatEnv(ParallelEnv):
    """Two-agent asymmetric combat environment faithful to the CS 329 proposal."""

    metadata = {"render_modes": ["ansi"], "name": "knight_boss_proposal_v0"}

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        alpha_h: float | None = None,
        alpha_d: float | None = None,
        reward_mode: str = "shaped",
        shaping_weight: float = 0.5,
        gamma: float = 0.995,
        draw_penalty: float = 0.75,
        invincibility_frames: int = 2,
        dodge_cooldown: int = 4,
        heal_amount: int = 30,
        heal_charges: int = 3,
        special1_range: int = 3,
        special2_radius: int = 2,
        enable_dodge: bool = True,
        enable_heal: bool = True,
        render_mode: str | None = None,
    ) -> None:
        self.config = config or GameConfig()
        self.alpha_h = float(self.config.alpha_h if alpha_h is None else alpha_h)
        self.alpha_d = float(self.config.alpha_d if alpha_d is None else alpha_d)
        if self.alpha_h <= 0 or self.alpha_d <= 0:
            raise ValueError("alpha_h and alpha_d must be positive.")
        if reward_mode not in PROPOSAL_REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {PROPOSAL_REWARD_MODES}."
            )
        if not isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if not isfinite(draw_penalty) or draw_penalty < 0:
            raise ValueError("draw_penalty must be non-negative and finite.")
        if not isfinite(shaping_weight) or shaping_weight < 0:
            raise ValueError("shaping_weight must be non-negative and finite.")

        self.reward_mode = reward_mode
        self.shaping_weight = float(shaping_weight)
        self.gamma = float(gamma)
        self.draw_penalty = float(draw_penalty)
        self.invincibility_frames = int(invincibility_frames)
        self.dodge_cooldown = int(dodge_cooldown)
        self.heal_amount = int(heal_amount)
        self.heal_charges_max = int(heal_charges)
        self.special1_range = int(special1_range)
        self.special2_radius = int(special2_radius)
        self.enable_dodge = bool(enable_dodge)
        self.enable_heal = bool(enable_heal)
        self.render_mode = render_mode

        self.possible_agents = list(PROPOSAL_AGENTS)
        self.agents: list[str] = []

        self.knight_max_hp = self.config.knight.max_health
        self.boss_max_hp = max(1, round(self.knight_max_hp * self.alpha_h))
        self.knight_atk_dmg = self.config.knight.basic_damage
        self.knight_atk_range = self.config.proposal_knight_basic_attack_chebyshev_range
        self.boss_atk_dmg = max(1, round(self.knight_atk_dmg * self.alpha_d))
        self.boss_spec1_dmg = max(1, round(50 * self.alpha_d))
        self.boss_spec2_dmg = max(1, round(40 * self.alpha_d))

        self._reset_state()

    def _reset_state(self) -> None:
        self.step_count = 0
        self.pos = {
            "knight": tuple(self.config.knight_position),
            "boss": tuple(self.config.boss_position),
        }
        self.hp = {
            "knight": self.knight_max_hp,
            "boss": self.boss_max_hp,
        }
        self.atk_cd = {"knight": 0, "boss": 0}
        self.knight_invuln = 0
        self.knight_dodge_cd = 0
        self.knight_heal_charges = self.heal_charges_max if self.enable_heal else 0
        # Special state z_t: active pending telegraph ("special1" beam or "special2" shockwave)
        self.pending_special: dict[str, Any] | None = None
        self.damage_dealt = {"knight": 0, "boss": 0}
        self.damage_received = {"knight": 0, "boss": 0}
        self.events_count = {
            "knight_dodge_used": 0,
            "knight_dodge_blocked": 0,
            "knight_heal_used": 0,
            "knight_heal_noop": 0,
            "knight_attack_attempt": 0,
            "knight_attack_hit": 0,
            "knight_attack_miss": 0,
            "knight_attack_noop": 0,
            "boss_special1_cast": 0,
            "boss_special1_hit": 0,
            "boss_special1_blocked": 0,
            "boss_special2_cast": 0,
            "boss_special2_hit": 0,
            "boss_special2_blocked": 0,
            "knight_blocked_movement": 0,
            "boss_blocked_movement": 0,
            "knight_escaped_special2": 0,
        }
        self.outcome: Outcome | None = None
        self.terminated = False
        self.truncated = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        del agent
        return spaces.Dict(
            {
                "positions": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
                "health": spaces.Box(0.0, float("inf"), shape=(4,), dtype=np.float32),
                "health_fraction": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
                "cooldowns": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
                "combat_stats": spaces.Box(0.0, float("inf"), shape=(12,), dtype=np.float32),
                "windups": spaces.Box(-1.0, float("inf"), shape=(8,), dtype=np.float32),
                "remaining_steps": spaces.Box(0.0, float(self.config.max_steps), shape=(1,), dtype=np.float32),
                "proposal_state": spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32),
                # Legality is state-derived and exposed to the policy so PPO
                # can use the same categorical distribution during sampling
                # and optimization. This does not reveal hidden game state.
                "action_mask": spaces.MultiBinary(8),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        del agent
        return spaces.Discrete(8)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        del seed, options
        self._reset_state()
        self.agents = self.possible_agents.copy()
        obs = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {"outcome": None, "step_count": 0} for agent in self.agents}
        return obs, infos

    def step(
        self, actions: Mapping[str, int]
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
            raise ValueError(f"Actions must be provided for {self.agents}.")

        k_act = int(actions["knight"])
        b_act = int(actions["boss"])
        if not (0 <= k_act < 8 and 0 <= b_act < 8):
            raise ValueError(f"Invalid proposal actions: knight={k_act}, boss={b_act}.")

        acting_agents = self.agents.copy()
        prev_health = dict(self.hp)
        prev_dist = abs(self.pos["knight"][0] - self.pos["boss"][0]) + abs(
            self.pos["knight"][1] - self.pos["boss"][1]
        )

        # Snapshot start-of-step timers
        k_cd_start = self.atk_cd["knight"]
        b_cd_start = self.atk_cd["boss"]
        k_dodge_cd_start = self.knight_dodge_cd
        pending_at_start = self.pending_special
        special2_threatened_before_response = False
        special2_dodge_available = (
            self.enable_dodge and k_dodge_cd_start == 0
        )
        if (
            pending_at_start is not None
            and pending_at_start["kind"] == "special2"
            and pending_at_start["remaining"] <= 1
        ):
            knight_r, knight_c = self.pos["knight"]
            origin_r, origin_c = pending_at_start["origin"]
            special2_threatened_before_response = (
                max(abs(knight_r - origin_r), abs(knight_c - origin_c))
                <= self.special2_radius
            )

        events: list[str] = []

        # 1. Handle Knight defensive/healing actions (u_K and q_K)
        if k_act == int(ProposalKnightAction.DODGE):
            if self.enable_dodge and k_dodge_cd_start == 0:
                self.knight_invuln = self.invincibility_frames
                self.knight_dodge_cd = self.dodge_cooldown
                self.events_count["knight_dodge_used"] += 1
                events.append("knight_dodge")
            else:
                events.append("knight_dodge_noop")
        elif k_act == int(ProposalKnightAction.HEAL):
            if self.enable_heal and self.knight_heal_charges > 0 and self.hp["knight"] < self.knight_max_hp:
                restored = min(self.heal_amount, self.knight_max_hp - self.hp["knight"])
                self.hp["knight"] += restored
                self.knight_heal_charges -= 1
                self.events_count["knight_heal_used"] += 1
                events.append("knight_heal")
            else:
                events.append("knight_heal_noop")
                self.events_count["knight_heal_noop"] += 1

        # 2. Simultaneous movement resolution
        pos_start = dict(self.pos)
        targets = {}
        for agent, aid in (("knight", k_act), ("boss", b_act)):
            r, c = pos_start[agent]
            if aid in _MOVE_DELTAS:
                dr, dc = _MOVE_DELTAS[aid]
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.config.rows and 0 <= nc < self.config.columns:
                    targets[agent] = (nr, nc)
                else:
                    targets[agent] = (r, c)
                    blocked_event = f"{agent}_blocked_movement"
                    events.append(blocked_event)
                    self.events_count[blocked_event] += 1
            else:
                targets[agent] = (r, c)

        special2_escape_available = False
        knight_approach_available = False
        if special2_threatened_before_response:
            origin_r, origin_c = pending_at_start["origin"]
            for dr, dc in _MOVE_DELTAS.values():
                candidate = (
                    pos_start["knight"][0] + dr,
                    pos_start["knight"][1] + dc,
                )
                in_bounds = (
                    0 <= candidate[0] < self.config.rows
                    and 0 <= candidate[1] < self.config.columns
                )
                blocked_by_collision = (
                    candidate == targets["boss"]
                    or (
                        candidate == pos_start["boss"]
                        and targets["boss"] == pos_start["knight"]
                    )
                )
                outside_shockwave = (
                    max(
                        abs(candidate[0] - origin_r),
                        abs(candidate[1] - origin_c),
                    )
                    > self.special2_radius
                )
                if in_bounds and not blocked_by_collision and outside_shockwave:
                    special2_escape_available = True
                    break

        if pending_at_start is None:
            start_distance = (
                abs(pos_start["knight"][0] - pos_start["boss"][0])
                + abs(pos_start["knight"][1] - pos_start["boss"][1])
            )
            for dr, dc in _MOVE_DELTAS.values():
                candidate = (
                    pos_start["knight"][0] + dr,
                    pos_start["knight"][1] + dc,
                )
                in_bounds = (
                    0 <= candidate[0] < self.config.rows
                    and 0 <= candidate[1] < self.config.columns
                )
                blocked_by_collision = (
                    candidate == targets["boss"]
                    or (
                        candidate == pos_start["boss"]
                        and targets["boss"] == pos_start["knight"]
                    )
                )
                candidate_distance = (
                    abs(candidate[0] - pos_start["boss"][0])
                    + abs(candidate[1] - pos_start["boss"][1])
                )
                if (
                    in_bounds
                    and not blocked_by_collision
                    and candidate_distance < start_distance
                ):
                    knight_approach_available = True
                    break

        if targets["knight"] == targets["boss"]:
            pass  # Same-square collision cancels both
        elif targets["knight"] == pos_start["boss"] and targets["boss"] == pos_start["knight"]:
            pass  # Swapping is blocked
        else:
            for agent in PROPOSAL_AGENTS:
                cand = targets[agent]
                other = "boss" if agent == "knight" else "knight"
                other_start = pos_start[other]
                other_target = targets[other]
                if cand != other_start or other_target != other_start:
                    if cand != self.pos[agent]:
                        self.pos[agent] = cand
                        events.append(f"{agent}_moved")

        knight_move_distance_delta = (
            abs(self.pos["knight"][0] - pos_start["boss"][0])
            + abs(self.pos["knight"][1] - pos_start["boss"][1])
            - prev_dist
        )

        # 3. Resolve Boss pending special attack z_t started earlier. A geometric
        # connection is recorded as a *pending* hit: whether it lands as a real
        # "hit" or is negated as a "block" is only decided in step 5, once the
        # Knight's invincibility frames (u_K) have been applied.
        dmg_to = {"knight": 0, "boss": 0}
        pending_special_hit: str | None = None
        if pending_at_start is not None:
            if pending_at_start["remaining"] > 1:
                self.pending_special = {
                    **pending_at_start,
                    "remaining": pending_at_start["remaining"] - 1,
                }
            else:
                kr, kc = self.pos["knight"]
                if pending_at_start["kind"] == "special1":
                    br, bc = pending_at_start["origin"]
                    dr, dc = pending_at_start["dir"]
                    hit = False
                    for step_r in range(1, self.special1_range + 1):
                        if (kr, kc) == (br + dr * step_r, bc + dc * step_r):
                            hit = True
                            break
                    if hit:
                        dmg_to["knight"] += self.boss_spec1_dmg
                        pending_special_hit = "boss_special1_hit"
                    else:
                        events.append("boss_special1_missed")
                else:  # special2 expanding shockwave centered at cast origin
                    sr, sc = pending_at_start["origin"]
                    if max(abs(kr - sr), abs(kc - sc)) <= self.special2_radius:
                        dmg_to["knight"] += self.boss_spec2_dmg
                        pending_special_hit = "boss_special2_hit"
                    else:
                        events.append("boss_special2_missed")
                        if special2_threatened_before_response:
                            events.append("knight_escaped_special2")
                            self.events_count["knight_escaped_special2"] += 1
                self.pending_special = None

        # 4. Resolve newly initiated attacks and Boss special charges
        started_atk_cd: dict[str, int] = {}
        kr, kc = self.pos["knight"]
        br, bc = self.pos["boss"]
        dist = abs(kr - br) + abs(kc - bc)
        knight_chebyshev_dist = max(abs(kr - br), abs(kc - bc))

        if (
            k_act == int(ProposalKnightAction.WAIT)
            and pending_at_start is None
            and k_cd_start == 0
            and knight_chebyshev_dist <= self.knight_atk_range
        ):
            events.append("knight_waited_with_attack_ready")
        elif (
            k_act == int(ProposalKnightAction.WAIT)
            and pending_at_start is None
            and knight_approach_available
            and not (
                k_cd_start > 0
                and knight_chebyshev_dist <= self.knight_atk_range
            )
        ):
            events.append("knight_waited_with_approach_available")

        if k_act == int(ProposalKnightAction.ATTACK) and k_cd_start == 0:
            self.events_count["knight_attack_attempt"] += 1
            if knight_chebyshev_dist <= self.knight_atk_range:
                dmg_to["boss"] += self.knight_atk_dmg
                started_atk_cd["knight"] = 1
                events.append("knight_attack_hit")
                self.events_count["knight_attack_hit"] += 1
            else:
                events.append("knight_attack_miss")
                self.events_count["knight_attack_miss"] += 1
        elif k_act == int(ProposalKnightAction.ATTACK):
            events.append("knight_attack_noop")
            self.events_count["knight_attack_noop"] += 1

        pending_attack_hit: str | None = None
        if b_act == int(ProposalBossAction.ATTACK) and b_cd_start == 0:
            if dist <= 1:
                dmg_to["knight"] += self.boss_atk_dmg
                started_atk_cd["boss"] = 1
                pending_attack_hit = "boss_attack_hit"
            else:
                events.append("boss_attack_miss")
        elif b_act == int(ProposalBossAction.SPECIAL_1) and b_cd_start == 0 and self.pending_special is None:
            dr = kr - pos_start["boss"][0]
            dc = kc - pos_start["boss"][1]
            if abs(dr) >= abs(dc):
                cdir = (1 if dr > 0 else (-1 if dr < 0 else 0), 0)
            else:
                cdir = (0, 1 if dc > 0 else -1)
            if cdir != (0, 0) and max(abs(dr), abs(dc)) <= self.special1_range:
                self.pending_special = {
                    "kind": "special1",
                    "origin": pos_start["boss"],
                    "dir": cdir,
                    "remaining": 1,
                }
                started_atk_cd["boss"] = 2
                self.events_count["boss_special1_cast"] += 1
                events.append("boss_special1_windup")
            else:
                events.append("boss_special1_out_of_range")
        elif b_act == int(ProposalBossAction.SPECIAL_2) and b_cd_start == 0 and self.pending_special is None:
            self.pending_special = {
                "kind": "special2",
                "origin": self.pos["boss"],
                "remaining": 1,
            }
            started_atk_cd["boss"] = 2
            self.events_count["boss_special2_cast"] += 1
            events.append("boss_special2_windup")

        # 5. Apply damage (with Knight invincibility u_K negation). Hit events are
        # classified here so that a blow negated by u_K is reported as a block
        # rather than a hit; otherwise a perfect dodge would emit both a bonus
        # (knight_dodged_damage) and a penalty (boss_special*_hit).
        negated = dmg_to["knight"] > 0 and self.knight_invuln > 0
        if negated:
            events.append("knight_dodged_damage")
            self.events_count["knight_dodge_blocked"] += 1
            dmg_to["knight"] = 0

        if pending_attack_hit is not None:
            events.append("boss_attack_blocked" if negated else pending_attack_hit)
        if pending_special_hit is not None:
            if negated:
                blocked_event = pending_special_hit.replace("_hit", "_blocked")
                events.append(blocked_event)
                self.events_count[blocked_event] += 1
            else:
                events.append(pending_special_hit)
                self.events_count[pending_special_hit] += 1

        if (
            k_act == int(ProposalKnightAction.WAIT)
            and special2_threatened_before_response
            and (special2_dodge_available or special2_escape_available)
            and pending_special_hit == "boss_special2_hit"
            and not negated
        ):
            events.append("knight_waited_through_avoidable_special2")

        for target, attacker in (("knight", "boss"), ("boss", "knight")):
            actual = min(self.hp[target], dmg_to[target])
            self.hp[target] -= actual
            self.damage_dealt[attacker] += actual
            self.damage_received[target] += actual

        # 6. Tick cooldowns & invincibility
        self.atk_cd["knight"] = started_atk_cd.get("knight", max(0, k_cd_start - 1))
        self.atk_cd["boss"] = started_atk_cd.get("boss", max(0, b_cd_start - 1))
        if not (k_act == int(ProposalKnightAction.DODGE) and self.enable_dodge and k_dodge_cd_start == 0):
            self.knight_dodge_cd = max(0, k_dodge_cd_start - 1)
        if self.knight_invuln > 0:
            self.knight_invuln -= 1

        self.step_count += 1
        k_alive = self.hp["knight"] > 0
        b_alive = self.hp["boss"] > 0
        if not k_alive and not b_alive:
            self.outcome = Outcome.DRAW
            self.terminated = True
        elif not b_alive:
            self.outcome = Outcome.KNIGHT_WIN
            self.terminated = True
        elif not k_alive:
            self.outcome = Outcome.BOSS_WIN
            self.terminated = True
        elif self.step_count >= self.config.max_steps:
            self.outcome = Outcome.DRAW
            self.truncated = True

        curr_dist = abs(self.pos["knight"][0] - self.pos["boss"][0]) + abs(
            self.pos["knight"][1] - self.pos["boss"][1]
        )
        rewards = self._compute_rewards(
            prev_health=prev_health,
            prev_dist=prev_dist,
            curr_dist=curr_dist,
            events=events,
            knight_move_distance_delta=knight_move_distance_delta,
        )
        obs = {agent: self._observe(agent) for agent in acting_agents}
        terms = {agent: self.terminated for agent in acting_agents}
        truncs = {agent: self.truncated for agent in acting_agents}
        infos = {
            agent: {
                "outcome": self.outcome.value if self.outcome else None,
                "step_count": self.step_count,
                "events": tuple(events),
                "damage_dealt": self.damage_dealt[agent],
                "damage_received": self.damage_received[agent],
                "manhattan_distance": curr_dist,
                "knight_move_distance_delta": knight_move_distance_delta,
                "proposal_events": dict(self.events_count),
            }
            for agent in acting_agents
        }
        if self.terminated or self.truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def _compute_rewards(
        self,
        *,
        prev_health: dict[str, int],
        prev_dist: int,
        curr_dist: int,
        events: list[str],
        knight_move_distance_delta: int,
    ) -> dict[str, float]:
        common = {
            "prev_knight_health": prev_health["knight"],
            "curr_knight_health": self.hp["knight"],
            "prev_boss_health": prev_health["boss"],
            "curr_boss_health": self.hp["boss"],
            "knight_max_health": self.knight_max_hp,
            "boss_max_health": self.boss_max_hp,
            "shaping_weight": self.shaping_weight,
            "gamma": self.gamma,
            "truncated": self.truncated,
        }

        if self.reward_mode == "terminal":
            return terminal_rewards(self.outcome)

        if self.reward_mode == "shaped":
            return shaped_rewards(self.outcome, **common)

        if self.reward_mode == "draw_penalty":
            return draw_penalty_rewards(
                self.outcome,
                **common,
                penalty=self.draw_penalty,
            )

        if self.reward_mode == "tactical":
            return tactical_rewards(
                self.outcome,
                **common,
                events=events,
                knight_move_distance_delta=knight_move_distance_delta,
                draw_penalty=self.draw_penalty,
            )

        if self.reward_mode == "combat":
            return combat_rewards(
                self.outcome,
                **common,
                prev_distance=prev_dist,
                curr_distance=curr_dist,
            )

        raise ValueError(f"Unsupported reward mode: {self.reward_mode}")

    def _observe(self, agent: str) -> dict[str, np.ndarray]:
        opponent = "boss" if agent == "knight" else "knight"
        r_scale = max(1, self.config.rows - 1)
        c_scale = max(1, self.config.columns - 1)
        own_r, own_c = self.pos[agent]
        opp_r, opp_c = self.pos[opponent]
        own_hp = self.hp[agent]
        opp_hp = self.hp[opponent]
        own_max = self.knight_max_hp if agent == "knight" else self.boss_max_hp
        opp_max = self.boss_max_hp if agent == "knight" else self.knight_max_hp

        positions = np.array(
            [own_r / r_scale, own_c / c_scale, opp_r / r_scale, opp_c / c_scale],
            dtype=np.float32,
        )
        health = np.array([own_hp, opp_hp, own_max, opp_max], dtype=np.float32)
        health_fraction = np.array(
            [own_hp / own_max, opp_hp / opp_max], dtype=np.float32
        )
        cooldowns = np.array(
            [
                self.atk_cd[agent] / (1.0 if agent == "knight" else 2.0),
                0.0,
                self.atk_cd[opponent] / (2.0 if agent == "knight" else 1.0),
                0.0,
            ],
            dtype=np.float32,
        )
        k_stats = (self.knight_atk_dmg, 0, self.knight_atk_range, 0, 1, 0)
        b_stats = (self.boss_atk_dmg, max(self.boss_spec1_dmg, self.boss_spec2_dmg), 1, self.special1_range, 1, 2)
        own_st, opp_st = (k_stats, b_stats) if agent == "knight" else (b_stats, k_stats)
        combat_stats = np.array([*own_st, *opp_st], dtype=np.float32)

        windups = np.zeros(8, dtype=np.float32)
        if self.pending_special is not None:
            offset = 0 if agent == "boss" else 4
            if self.pending_special["kind"] == "special1":
                dr, dc = self.pending_special["dir"]
                windups[offset : offset + 4] = (1.0, float(dr), float(dc), float(self.pending_special["remaining"]))
            else:
                sr, sc = self.pending_special["origin"]
                windups[offset : offset + 4] = (1.0, sr / r_scale, sc / c_scale, float(self.pending_special["remaining"]))

        spec1_act = 1.0 if (self.pending_special and self.pending_special["kind"] == "special1") else 0.0
        spec2_act = 1.0 if (self.pending_special and self.pending_special["kind"] == "special2") else 0.0
        sr, sc = (
            self.pending_special["origin"]
            if self.pending_special is not None
            else self.pos["boss"]
        )
        proposal_state = np.array(
            [
                self.knight_invuln / max(1, self.invincibility_frames),
                self.knight_heal_charges / max(1, self.heal_charges_max),
                spec1_act,
                spec2_act,
                sr / r_scale,
                sc / c_scale,
                self.knight_dodge_cd / max(1, self.dodge_cooldown),
                1.0 if self.enable_dodge else 0.0,
                1.0 if self.enable_heal else 0.0,
            ],
            dtype=np.float32,
        )
        return {
            "positions": positions,
            "health": health,
            "health_fraction": health_fraction,
            "cooldowns": cooldowns,
            "combat_stats": combat_stats,
            "windups": windups,
            "remaining_steps": np.array(
                [max(0, self.config.max_steps - self.step_count)], dtype=np.float32
            ),
            "proposal_state": proposal_state,
            "action_mask": self._action_mask(agent),
        }

    def _action_mask(self, agent: str) -> np.ndarray:
        """Return actions that can produce a valid state transition now.

        Movement masks only use board boundaries. Whether a move collides
        depends on the other agent's simultaneous action, so it is left to the
        existing collision rules. Attack range is likewise not masked because
        the opponent can move during the same transition.
        """
        r, c = self.pos[agent]
        mask = np.ones(8, dtype=np.int8)
        for action, (dr, dc) in _MOVE_DELTAS.items():
            mask[action] = int(
                0 <= r + dr < self.config.rows
                and 0 <= c + dc < self.config.columns
            )

        if agent == "knight":
            mask[int(ProposalKnightAction.ATTACK)] = int(self.atk_cd[agent] == 0)
            mask[int(ProposalKnightAction.DODGE)] = int(
                self.enable_dodge and self.knight_dodge_cd == 0
            )
            mask[int(ProposalKnightAction.HEAL)] = int(
                self.enable_heal
                and self.knight_heal_charges > 0
                and self.hp[agent] < self.knight_max_hp
            )
        else:
            can_attack = self.atk_cd[agent] == 0 and self.pending_special is None
            mask[int(ProposalBossAction.ATTACK)] = int(can_attack)
            mask[int(ProposalBossAction.SPECIAL_1)] = int(can_attack)
            mask[int(ProposalBossAction.SPECIAL_2)] = int(can_attack)
        # WAIT is the guaranteed fallback, including fully blocked corners.
        mask[int(ProposalCommonAction.WAIT)] = 1
        return mask

    def close(self) -> None:
        self.agents = []

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (
            f"step {self.step_count}/{self.config.max_steps} | "
            f"K pos={self.pos['knight']} hp={self.hp['knight']}/{self.knight_max_hp} u_K={self.knight_invuln} q_K={self.knight_heal_charges} | "
            f"B pos={self.pos['boss']} hp={self.hp['boss']}/{self.boss_max_hp} spec={self.pending_special}"
        )


def _step_toward(own: tuple[int, int], opp: tuple[int, int]) -> int:
    dr = opp[0] - own[0]
    dc = opp[1] - own[1]
    if abs(dr) >= abs(dc):
        return int(ProposalCommonAction.DOWN if dr > 0 else ProposalCommonAction.UP)
    return int(ProposalCommonAction.RIGHT if dc > 0 else ProposalCommonAction.LEFT)


class ProposalHeuristicBoss(BaseAgent):
    """Scripted Boss per Proposal Section 4.1: chases, attacks, uses specials probabilistically."""

    def __init__(
        self,
        *,
        rows: int = 9,
        columns: int = 9,
        seed: int | None = None,
        special1_prob: float = 0.25,
        special2_prob: float = 0.25,
        special1_range: int = 3,
        special2_radius: int = 2,
    ) -> None:
        super().__init__(rows=rows, columns=columns, seed=seed)
        self.special1_prob = special1_prob
        self.special2_prob = special2_prob
        self.special1_range = special1_range
        self.special2_radius = special2_radius

    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

    def clone_for_eval(self, *, rows: int, columns: int, seed: int) -> "ProposalHeuristicBoss":
        return ProposalHeuristicBoss(
            rows=rows,
            columns=columns,
            seed=seed,
            special1_prob=self.special1_prob,
            special2_prob=self.special2_prob,
            special1_range=self.special1_range,
            special2_radius=self.special2_radius,
        )

    def act(self, observation: Observation) -> int:
        pos = observation["positions"]
        r_s = max(1, self.rows - 1)
        c_s = max(1, self.columns - 1)
        own = (round(float(pos[0]) * r_s), round(float(pos[1]) * c_s))
        opp = (round(float(pos[2]) * r_s), round(float(pos[3]) * c_s))
        dist = abs(own[0] - opp[0]) + abs(own[1] - opp[1])
        cheb = max(abs(own[0] - opp[0]), abs(own[1] - opp[1]))
        cd = float(observation["cooldowns"][0])
        pstate = observation.get("proposal_state")
        pending = (pstate is not None) and (float(pstate[2]) > 0.5 or float(pstate[3]) > 0.5)

        if cd <= 1e-6 and not pending and cheb <= self.special2_radius:
            u = float(self._rng.random())
            if u < self.special1_prob:
                return int(ProposalBossAction.SPECIAL_1)
            if u < self.special1_prob + self.special2_prob:
                return int(ProposalBossAction.SPECIAL_2)
            if dist <= 1:
                return int(ProposalBossAction.ATTACK)
        if own == opp:
            return int(ProposalBossAction.WAIT)
        return _step_toward(own, opp)


class ProposalHeuristicKnight(BaseAgent):
    """Scripted tactical Knight baseline: attacks in range, dodges Boss specials, heals low HP."""

    def clone_for_eval(self, *, rows: int, columns: int, seed: int) -> "ProposalHeuristicKnight":
        return ProposalHeuristicKnight(rows=rows, columns=columns, seed=seed)

    def act(self, observation: Observation) -> int:
        pos = observation["positions"]
        r_s = max(1, self.rows - 1)
        c_s = max(1, self.columns - 1)
        own = (round(float(pos[0]) * r_s), round(float(pos[1]) * c_s))
        opp = (round(float(pos[2]) * r_s), round(float(pos[3]) * c_s))
        dist = abs(own[0] - opp[0]) + abs(own[1] - opp[1])
        cheb = max(abs(own[0] - opp[0]), abs(own[1] - opp[1]))
        hp_frac = float(observation["health_fraction"][0])
        atk_cd = float(observation["cooldowns"][0])
        attack_range = int(round(float(observation["combat_stats"][2])))
        pstate = observation["proposal_state"]
        u_k = float(pstate[0])
        q_k = float(pstate[1])
        spec1 = float(pstate[2]) > 0.5
        spec2 = float(pstate[3]) > 0.5
        dodge_cd = float(pstate[6])
        can_dodge = float(pstate[7]) > 0.5
        can_heal = float(pstate[8]) > 0.5

        if can_dodge and (spec1 or spec2) and u_k <= 1e-6 and dodge_cd <= 1e-6:
            return int(ProposalKnightAction.DODGE)
        if can_heal and q_k > 1e-6 and hp_frac <= 0.45:
            return int(ProposalKnightAction.HEAL)
        if atk_cd <= 1e-6 and cheb <= attack_range:
            return int(ProposalKnightAction.ATTACK)
        if own == opp:
            return int(ProposalKnightAction.WAIT)
        return _step_toward(own, opp)


class ProposalRandomAgent(BaseAgent):
    """Uniformly random baseline over all 8 proposal actions."""

    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

    def clone_for_eval(self, *, rows: int, columns: int, seed: int) -> "ProposalRandomAgent":
        return ProposalRandomAgent(rows=rows, columns=columns, seed=seed)

    def act(self, observation: Observation) -> int:
        mask = np.asarray(observation.get("action_mask", np.ones(8)), dtype=bool)
        choices = np.flatnonzero(mask)
        if choices.size == 0:
            raise ValueError("Proposal observation has no valid actions.")
        return int(self._rng.choice(choices))
