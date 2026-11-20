"""Reward functions for the Knight-vs-Boss competitive game.

Health shaping uses discounted potential-based reward shaping.

Knight potential:
    Phi_K(s) = w * (knight_hp_fraction - boss_hp_fraction)

Boss potential:
    Phi_B(s) = -Phi_K(s)

For non-terminal transitions:
    F_K(s,s') = gamma * Phi_K(s') - Phi_K(s)

For terminal and truncated transitions:
    F_K(s,s_terminal) = -Phi_K(s)

The terminal-state potential is zero. Consequently, the discounted sum
of health-shaping rewards over an episode is -Phi_K(s_0), independent
of the actions taken. This preserves the optimal policies for a fixed
initial-state distribution.

The convention matches the PPO rollout implementation, which treats
both termination and truncation as episode boundaries.
"""

from __future__ import annotations

from math import isfinite

from knight_boss.game.types import Outcome

REWARD_MODES = ("terminal", "shaped", "draw_penalty", "combat")
PROPOSAL_REWARD_MODES = (*REWARD_MODES, "tactical")


def _is_indecisive(outcome: Outcome | None, truncated: bool) -> bool:
    """True when an episode ends in a draw or timeout."""
    return bool(truncated) or outcome is Outcome.DRAW


def terminal_rewards(outcome: Outcome | None) -> dict[str, float]:
    """Win/loss/draw payoff. Draws and timeouts both pay zero."""
    if outcome is None or outcome is Outcome.DRAW:
        return {"knight": 0.0, "boss": 0.0}
    if outcome is Outcome.KNIGHT_WIN:
        return {"knight": 1.0, "boss": -1.0}
    if outcome is Outcome.BOSS_WIN:
        return {"knight": -1.0, "boss": 1.0}
    raise ValueError(f"Unsupported outcome: {outcome!r}")


def _health_shaping(
    *,
    prev_knight_health: float,
    curr_knight_health: float,
    prev_boss_health: float,
    curr_boss_health: float,
    knight_max_health: float,
    boss_max_health: float,
    shaping_weight: float,
    gamma: float,
    episode_done: bool,
) -> float:
    """Calculate Knight's discounted potential-based health shaping."""
    if knight_max_health <= 0 or boss_max_health <= 0:
        raise ValueError("Maximum health values must be positive.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not isfinite(shaping_weight) or shaping_weight < 0:
        raise ValueError("shaping_weight must be non-negative and finite.")

    prev_phi = shaping_weight * (
        prev_knight_health / knight_max_health
        - prev_boss_health / boss_max_health
    )

    # Terminal and truncated transitions enter an absorbing zero-potential
    # state, matching the episode-boundary handling in PPO.
    if episode_done:
        return -prev_phi

    curr_phi = shaping_weight * (
        curr_knight_health / knight_max_health
        - curr_boss_health / boss_max_health
    )
    return gamma * curr_phi - prev_phi


def shaped_rewards(
    outcome: Outcome | None,
    *,
    prev_knight_health: float,
    curr_knight_health: float,
    prev_boss_health: float,
    curr_boss_health: float,
    knight_max_health: float,
    boss_max_health: float,
    shaping_weight: float = 0.5,
    gamma: float = 0.995,
    truncated: bool = False,
) -> dict[str, float]:
    """Terminal payoff plus zero-sum discounted potential-based health shaping.

    Actual HP changes are used, so healing restores health potential.
    """
    base = terminal_rewards(outcome)
    delta = _health_shaping(
        prev_knight_health=prev_knight_health,
        curr_knight_health=curr_knight_health,
        prev_boss_health=prev_boss_health,
        curr_boss_health=curr_boss_health,
        knight_max_health=knight_max_health,
        boss_max_health=boss_max_health,
        shaping_weight=shaping_weight,
        gamma=gamma,
        episode_done=(outcome is not None or truncated),
    )
    return {
        "knight": float(base["knight"] + delta),
        "boss": float(base["boss"] - delta),
    }


def draw_penalty_rewards(
    outcome: Outcome | None,
    *,
    truncated: bool = False,
    prev_knight_health: float,
    curr_knight_health: float,
    prev_boss_health: float,
    curr_boss_health: float,
    knight_max_health: float,
    boss_max_health: float,
    shaping_weight: float = 0.5,
    gamma: float = 0.995,
    penalty: float = 0.75,
) -> dict[str, float]:
    """Health-shaped rewards with a common-payoff draw/timeout penalty."""
    base = shaped_rewards(
        outcome,
        truncated=truncated,
        prev_knight_health=prev_knight_health,
        curr_knight_health=curr_knight_health,
        prev_boss_health=prev_boss_health,
        curr_boss_health=curr_boss_health,
        knight_max_health=knight_max_health,
        boss_max_health=boss_max_health,
        shaping_weight=shaping_weight,
        gamma=gamma,
    )
    if not isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be non-negative and finite.")

    if _is_indecisive(outcome, truncated):
        return {
            "knight": float(base["knight"] - penalty),
            "boss": float(base["boss"] - penalty),
        }
    return base


def combat_rewards(
    outcome: Outcome | None,
    *,
    truncated: bool = False,
    prev_knight_health: float,
    curr_knight_health: float,
    prev_boss_health: float,
    curr_boss_health: float,
    knight_max_health: float,
    boss_max_health: float,
    prev_distance: int,
    curr_distance: int,
    shaping_weight: float = 0.5,
    gamma: float = 0.995,
    step_penalty: float = 0.005,
    distance_weight: float = 0.01,
    timeout_penalty: float = 0.5,
) -> dict[str, float]:
    """Health shaping plus symmetric anti-stall and approach signals."""
    base = shaped_rewards(
        outcome,
        truncated=truncated,
        prev_knight_health=prev_knight_health,
        curr_knight_health=curr_knight_health,
        prev_boss_health=prev_boss_health,
        curr_boss_health=curr_boss_health,
        knight_max_health=knight_max_health,
        boss_max_health=boss_max_health,
        shaping_weight=shaping_weight,
        gamma=gamma,
    )

    engagement = distance_weight * (prev_distance - curr_distance) - step_penalty
    if _is_indecisive(outcome, truncated):
        engagement -= timeout_penalty

    return {
        "knight": float(base["knight"] + engagement),
        "boss": float(base["boss"] + engagement),
    }


def tactical_rewards(
    outcome: Outcome | None,
    *,
    events: tuple[str, ...] | list[str],
    knight_move_distance_delta: int,
    truncated: bool = False,
    prev_knight_health: float,
    curr_knight_health: float,
    prev_boss_health: float,
    curr_boss_health: float,
    knight_max_health: float,
    boss_max_health: float,
    shaping_weight: float = 0.5,
    gamma: float = 0.995,
    draw_penalty: float = 0.75,
) -> dict[str, float]:
    """Reward accurate attacks and effective defense, on top of draw penalties.

    Bonuses are attached to outcomes rather than selecting an action blindly:
    a hit earns a small bonus, a miss/no-op costs a small amount, and a dodge
    or sidestep earns a bonus only when it avoids a pending Boss special or
    blocks incoming damage. Knight-only distance shaping encourages pursuit;
    retreat costs less than a successful dodge or evasion bonus.
    """
    base = draw_penalty_rewards(
        outcome,
        truncated=truncated,
        prev_knight_health=prev_knight_health,
        curr_knight_health=curr_knight_health,
        prev_boss_health=prev_boss_health,
        curr_boss_health=curr_boss_health,
        knight_max_health=knight_max_health,
        boss_max_health=boss_max_health,
        shaping_weight=shaping_weight,
        gamma=gamma,
        penalty=draw_penalty,
    )

    event_set = set(events)
    tactical_delta = 0.0
    knight_distance_delta = 0.0
    if "knight_attack_hit" in event_set:
        tactical_delta += 0.05
    if "knight_attack_miss" in event_set or "knight_attack_noop" in event_set:
        tactical_delta -= 0.025
    if "knight_waited_with_attack_ready" in event_set:
        tactical_delta -= 0.05
    if "knight_waited_with_approach_available" in event_set:
        tactical_delta -= 0.025
    if "knight_waited_through_avoidable_special2" in event_set:
        tactical_delta -= 0.05
    if "knight_dodged_damage" in event_set:
        tactical_delta += 0.05
    elif "knight_moved" in event_set and "boss_attack_miss" in event_set:
        tactical_delta += 0.04
    elif "knight_dodge" in event_set:
        tactical_delta -= 0.025
    if "knight_dodge_noop" in event_set or "knight_heal_noop" in event_set:
        tactical_delta -= 0.025
    if "boss_special1_hit" in event_set or "boss_special2_hit" in event_set:
        tactical_delta -= 0.10

    # The pursuit signal applies only to the Knight. A one-cell retreat costs
    # 0.02, less than a successful dodge (0.05) or evasion (0.04).
    if knight_move_distance_delta > 0:
        knight_distance_delta -= 0.02 * knight_move_distance_delta
    elif knight_move_distance_delta < 0:
        knight_distance_delta += 0.01 * -knight_move_distance_delta

    return {
        "knight": float(base["knight"] + tactical_delta + knight_distance_delta),
        "boss": float(base["boss"] - tactical_delta),
    }
