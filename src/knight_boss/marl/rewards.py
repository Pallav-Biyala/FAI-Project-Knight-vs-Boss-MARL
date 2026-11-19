"""Terminal-only competitive rewards and optional zero-sum shaping."""

from __future__ import annotations

from knight_boss.game.types import Outcome

REWARD_MODES = ("terminal", "shaped", "draw_penalty", "combat")


def terminal_rewards(outcome: Outcome | None) -> dict[str, float]:
    if outcome is None or outcome is Outcome.DRAW:
        return {"knight": 0.0, "boss": 0.0}
    if outcome is Outcome.KNIGHT_WIN:
        return {"knight": 1.0, "boss": -1.0}
    if outcome is Outcome.BOSS_WIN:
        return {"knight": -1.0, "boss": 1.0}
    raise ValueError(f"Unsupported outcome: {outcome!r}")


def shaped_rewards(
    outcome: Outcome | None,
    *,
    knight_damage_received_delta: int,
    boss_damage_received_delta: int,
    knight_max_health: int,
    boss_max_health: int,
    shaping_weight: float = 0.5,
) -> dict[str, float]:
    """Zero-sum normalized health-delta shaping added to terminal rewards.

    This is an experimental reward formulation intended for separate comparison
    when pure terminal rewards stagnate against much stronger opponents.
    """
    base = terminal_rewards(outcome)
    knight_loss_frac = knight_damage_received_delta / max(1, knight_max_health)
    boss_loss_frac = boss_damage_received_delta / max(1, boss_max_health)
    delta = shaping_weight * (boss_loss_frac - knight_loss_frac)
    return {
        "knight": float(base["knight"] + delta),
        "boss": float(base["boss"] - delta),
    }


def draw_penalty_rewards(
    outcome: Outcome | None,
    *,
    truncated: bool = False,
    knight_damage_received_delta: int,
    boss_damage_received_delta: int,
    knight_max_health: int,
    boss_max_health: int,
    shaping_weight: float = 0.5,
    penalty: float = 0.75,
) -> dict[str, float]:
    """HP-delta shaped rewards with an explicit penalty strictly on draws/timeouts.

    Unlike ``combat``, applies zero per-step living penalty and zero distance
    bias during active play, penalizing both agents only when an episode ends
    in a draw or truncation timeout.
    """
    base = shaped_rewards(
        outcome,
        knight_damage_received_delta=knight_damage_received_delta,
        boss_damage_received_delta=boss_damage_received_delta,
        knight_max_health=knight_max_health,
        boss_max_health=boss_max_health,
        shaping_weight=shaping_weight,
    )
    if outcome is Outcome.DRAW or truncated:
        return {
            "knight": float(base["knight"] - penalty),
            "boss": float(base["boss"] - penalty),
        }
    return base


def combat_rewards(
    outcome: Outcome | None,
    *,
    truncated: bool = False,
    knight_damage_received_delta: int,
    boss_damage_received_delta: int,
    knight_max_health: int,
    boss_max_health: int,
    prev_distance: int,
    curr_distance: int,
    shaping_weight: float = 0.5,
    step_penalty: float = 0.005,
    distance_weight: float = 0.01,
    timeout_penalty: float = 0.5,
) -> dict[str, float]:
    """Combat-engagement shaping adding symmetric anti-stall and approach signals.

    Extends zero-sum health-delta shaping with a small symmetric time penalty,
    distance-closing potential, and timeout draw penalty so self-play agents do
    not stall into passive 300-step evasion draws.
    """
    base = shaped_rewards(
        outcome,
        knight_damage_received_delta=knight_damage_received_delta,
        boss_damage_received_delta=boss_damage_received_delta,
        knight_max_health=knight_max_health,
        boss_max_health=boss_max_health,
        shaping_weight=shaping_weight,
    )
    engagement = distance_weight * (prev_distance - curr_distance) - step_penalty
    if truncated:
        engagement -= timeout_penalty
    return {
        "knight": float(base["knight"] + engagement),
        "boss": float(base["boss"] + engagement),
    }
