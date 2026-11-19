"""PettingZoo Parallel API adapter for the standalone game engine."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from knight_boss.game.config import GameConfig
from knight_boss.game.engine import AGENTS, GameEngine
from knight_boss.game.types import Action
from knight_boss.marl.rewards import (
    REWARD_MODES,
    combat_rewards,
    draw_penalty_rewards,
    shaped_rewards,
    terminal_rewards,
)


class KnightBossParallelEnv(ParallelEnv[str, dict[str, np.ndarray], int]):
    """Two-player simultaneous game with terminal zero-sum rewards."""

    metadata = {"name": "knight_boss_v0", "render_modes": ["ansi"]}

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
    ) -> None:
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'.")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}.")
        if not isfinite(shaping_weight) or shaping_weight < 0:
            raise ValueError("shaping_weight must be non-negative and finite.")
        if not isfinite(draw_penalty) or draw_penalty < 0:
            raise ValueError("draw_penalty must be non-negative and finite.")
        self.config = config or GameConfig()
        self.alpha_h = self.config.alpha_h if alpha_h is None else alpha_h
        self.alpha_d = self.config.alpha_d if alpha_d is None else alpha_d
        if (
            not isfinite(self.alpha_h)
            or not isfinite(self.alpha_d)
            or self.alpha_h <= 0
            or self.alpha_d <= 0
        ):
            raise ValueError("alpha_h and alpha_d must be positive.")
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.shaping_weight = shaping_weight
        self.draw_penalty = draw_penalty
        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        self._observation_spaces = {
            agent: self._make_observation_space() for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(len(Action)) for agent in self.possible_agents
        }
        self.engine = GameEngine(
            self.config, alpha_h=self.alpha_h, alpha_d=self.alpha_d
        )

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict]]:
        self.np_random = np.random.default_rng(seed)
        for index, agent in enumerate(self.possible_agents):
            agent_seed = None if seed is None else seed + index
            self.action_space(agent).seed(agent_seed)
        if options:
            alpha_h = float(options.get("alpha_h", self.alpha_h))
            alpha_d = float(options.get("alpha_d", self.alpha_d))
            if (
                not isfinite(alpha_h)
                or not isfinite(alpha_d)
                or alpha_h <= 0
                or alpha_d <= 0
            ):
                raise ValueError("alpha_h and alpha_d must be positive.")
            self.alpha_h, self.alpha_d = alpha_h, alpha_d
        self.engine = GameEngine(
            self.config, alpha_h=self.alpha_h, alpha_d=self.alpha_d
        )
        self.agents = self.possible_agents.copy()
        observations = {
            agent: self._observe(agent) for agent in self.agents
        }
        infos = {agent: {"outcome": None, "step_count": 0} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        if not self.agents:
            raise RuntimeError("Episode is complete; call reset() before stepping.")
        if set(actions) != set(self.agents):
            raise ValueError("An action must be supplied for each live agent.")
        parsed_actions: dict[str, Action] = {}
        for agent, action in actions.items():
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action for {agent}: {action!r}")
            parsed_actions[agent] = Action(int(action))

        acting_agents = self.agents.copy()
        prev_knight_rec = self.engine.damage_received["knight"]
        prev_boss_rec = self.engine.damage_received["boss"]
        prev_k_pos = self.engine.fighters["knight"].position
        prev_b_pos = self.engine.fighters["boss"].position
        prev_distance = abs(prev_k_pos[0] - prev_b_pos[0]) + abs(prev_k_pos[1] - prev_b_pos[1])
        result = self.engine.step(parsed_actions)
        observations = {agent: self._observe(agent) for agent in acting_agents}

        if self.reward_mode == "combat":
            curr_k_pos = self.engine.fighters["knight"].position
            curr_b_pos = self.engine.fighters["boss"].position
            curr_distance = abs(curr_k_pos[0] - curr_b_pos[0]) + abs(curr_k_pos[1] - curr_b_pos[1])
            rewards = combat_rewards(
                result.outcome,
                truncated=result.truncated,
                knight_damage_received_delta=(
                    self.engine.damage_received["knight"] - prev_knight_rec
                ),
                boss_damage_received_delta=(
                    self.engine.damage_received["boss"] - prev_boss_rec
                ),
                knight_max_health=self.engine.fighters["knight"].max_health,
                boss_max_health=self.engine.fighters["boss"].max_health,
                prev_distance=prev_distance,
                curr_distance=curr_distance,
                shaping_weight=self.shaping_weight,
            )
        elif self.reward_mode == "draw_penalty":
            rewards = draw_penalty_rewards(
                result.outcome,
                truncated=result.truncated,
                knight_damage_received_delta=(
                    self.engine.damage_received["knight"] - prev_knight_rec
                ),
                boss_damage_received_delta=(
                    self.engine.damage_received["boss"] - prev_boss_rec
                ),
                knight_max_health=self.engine.fighters["knight"].max_health,
                boss_max_health=self.engine.fighters["boss"].max_health,
                shaping_weight=self.shaping_weight,
                penalty=self.draw_penalty,
            )
        elif self.reward_mode == "shaped":
            rewards = shaped_rewards(
                result.outcome,
                knight_damage_received_delta=(
                    self.engine.damage_received["knight"] - prev_knight_rec
                ),
                boss_damage_received_delta=(
                    self.engine.damage_received["boss"] - prev_boss_rec
                ),
                knight_max_health=self.engine.fighters["knight"].max_health,
                boss_max_health=self.engine.fighters["boss"].max_health,
                shaping_weight=self.shaping_weight,
            )
        else:
            rewards = terminal_rewards(result.outcome)

        terminations = {
            agent: result.terminated for agent in acting_agents
        }
        truncations = {
            agent: result.truncated for agent in acting_agents
        }
        infos = {
            agent: {
                "outcome": result.outcome.value if result.outcome else None,
                "step_count": self.engine.step_count,
                "events": result.events,
                "damage_dealt": self.engine.stats.damage_dealt[agent],
                "damage_received": self.engine.stats.damage_received[agent],
            }
            for agent in acting_agents
        }
        if result.terminated or result.truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        board = [["." for _ in range(self.engine.board.columns)] for _ in range(self.engine.board.rows)]
        for agent, marker in (("knight", "K"), ("boss", "B")):
            row, column = self.engine.fighters[agent].position
            board[row][column] = marker
        lines = [" ".join(row) for row in board]
        lines.append(
            f"Knight {self.engine.fighters['knight'].health}/"
            f"{self.engine.fighters['knight'].max_health} | "
            f"Boss {self.engine.fighters['boss'].health}/"
            f"{self.engine.fighters['boss'].max_health} | "
            f"step {self.engine.step_count}/{self.config.max_steps}"
        )
        if self.engine.outcome is not None:
            lines.append(f"Outcome: {self.engine.outcome.value}")
        return "\n".join(lines)

    def close(self) -> None:
        self.agents = []

    def _observe(self, agent: str) -> dict[str, np.ndarray]:
        opponent = "boss" if agent == "knight" else "knight"
        own_fighter = self.engine.fighters[agent]
        opponent_fighter = self.engine.fighters[opponent]
        board = self.engine.board
        position_scale = np.array(
            [max(1, board.rows - 1), max(1, board.columns - 1)],
            dtype=np.float32,
        )
        positions = np.array(
            [
                own_fighter.position[0] / position_scale[0],
                own_fighter.position[1] / position_scale[1],
                opponent_fighter.position[0] / position_scale[0],
                opponent_fighter.position[1] / position_scale[1],
            ],
            dtype=np.float32,
        )
        health = np.array(
            [
                own_fighter.health,
                opponent_fighter.health,
                own_fighter.max_health,
                opponent_fighter.max_health,
            ],
            dtype=np.float32,
        )
        health_fraction = np.array(
            [
                own_fighter.health / own_fighter.max_health,
                opponent_fighter.health / opponent_fighter.max_health,
            ],
            dtype=np.float32,
        )
        cooldowns = np.array(
            [
                own_fighter.basic_cooldown
                / max(1, own_fighter.basic_cooldown_duration),
                own_fighter.heavy_cooldown
                / max(1, own_fighter.heavy_cooldown_duration),
                opponent_fighter.basic_cooldown
                / max(1, opponent_fighter.basic_cooldown_duration),
                opponent_fighter.heavy_cooldown
                / max(1, opponent_fighter.heavy_cooldown_duration),
            ],
            dtype=np.float32,
        )
        combat_stats = np.array(
            [
                own_fighter.basic_damage,
                own_fighter.heavy_damage,
                own_fighter.basic_range,
                own_fighter.heavy_range,
                own_fighter.basic_cooldown_duration,
                own_fighter.heavy_cooldown_duration,
                opponent_fighter.basic_damage,
                opponent_fighter.heavy_damage,
                opponent_fighter.basic_range,
                opponent_fighter.heavy_range,
                opponent_fighter.basic_cooldown_duration,
                opponent_fighter.heavy_cooldown_duration,
            ],
            dtype=np.float32,
        )
        windups = np.zeros(8, dtype=np.float32)
        for offset, fighter_name in ((0, agent), (4, opponent)):
            pending = self.engine.pending_heavy.get(fighter_name)
            if pending is not None:
                windups[offset : offset + 4] = (
                    1.0,
                    pending.direction[0],
                    pending.direction[1],
                    pending.remaining_steps,
                )
        remaining_steps = np.array(
            [max(0, self.config.max_steps - self.engine.step_count)],
            dtype=np.float32,
        )
        return {
            "positions": positions,
            "health": health,
            "health_fraction": health_fraction,
            "cooldowns": cooldowns,
            "combat_stats": combat_stats,
            "windups": windups,
            "remaining_steps": remaining_steps,
        }

    def _make_observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "positions": spaces.Box(
                    low=0.0, high=1.0, shape=(4,), dtype=np.float32
                ),
                "health": spaces.Box(
                    low=0.0, high=np.inf, shape=(4,), dtype=np.float32
                ),
                "health_fraction": spaces.Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "cooldowns": spaces.Box(
                    low=0.0, high=1.0, shape=(4,), dtype=np.float32
                ),
                "combat_stats": spaces.Box(
                    low=0.0, high=np.inf, shape=(12,), dtype=np.float32
                ),
                "windups": spaces.Box(
                    low=-1.0, high=np.inf, shape=(8,), dtype=np.float32
                ),
                "remaining_steps": spaces.Box(
                    low=0.0,
                    high=float(self.config.max_steps),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )


def parallel_env(**kwargs) -> KnightBossParallelEnv:
    """PettingZoo-style environment factory."""
    return KnightBossParallelEnv(**kwargs)
