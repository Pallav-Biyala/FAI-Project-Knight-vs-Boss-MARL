"""Evaluation result records and binomial confidence intervals."""

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_index: int
    evaluation_seed: int
    outcome: str
    timeout: bool
    episode_length: int
    knight_damage_dealt: int
    knight_damage_received: int
    boss_damage_dealt: int
    boss_damage_received: int
    invalid_knight_moves: int = 0
    special2_casts: int = 0
    special2_hits: int = 0
    special2_blocks: int = 0
    special2_escapes: int = 0
    knight_heal_noops: int = 0
    knight_heals: int = 0
    knight_attack_attempts: int = 0
    knight_attack_hits: int = 0


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    confidence_level: float
    knight_wins: int
    boss_wins: int
    draws: int
    timeouts: int
    knight_win_rate: float
    boss_win_rate: float
    draw_rate: float
    timeout_rate: float
    knight_win_ci_lower: float
    knight_win_ci_upper: float
    boss_win_ci_lower: float
    boss_win_ci_upper: float
    draw_ci_lower: float
    draw_ci_upper: float
    average_episode_length: float
    average_knight_damage_dealt: float
    average_knight_damage_received: float
    average_boss_damage_dealt: float
    average_boss_damage_received: float
    average_invalid_knight_moves: float = 0.0
    average_special2_casts: float = 0.0
    average_special2_hits: float = 0.0
    average_special2_blocks: float = 0.0
    average_special2_escapes: float = 0.0
    average_knight_heal_noops: float = 0.0
    average_knight_heals: float = 0.0
    average_knight_attack_attempts: float = 0.0
    average_knight_attack_hits: float = 0.0
    special2_hit_rate: float = 0.0
    special2_block_rate: float = 0.0
    invalid_knight_moves_per_step: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 1:
        raise ValueError("trials must be positive.")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * ((proportion * (1.0 - proportion) / trials) + z * z / (4.0 * trials**2))
        ** 0.5
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == trials else min(1.0, center + margin)
    return lower, upper


def summarize_episodes(
    episodes: list[EpisodeMetrics], *, confidence: float = 0.95
) -> EvaluationSummary:
    if not episodes:
        raise ValueError("At least one evaluation episode is required.")
    total = len(episodes)
    special2_casts = sum(item.special2_casts for item in episodes)
    knight_wins = sum(item.outcome == "knight_win" for item in episodes)
    boss_wins = sum(item.outcome == "boss_win" for item in episodes)
    draws = sum(item.outcome == "draw" for item in episodes)
    timeouts = sum(item.timeout for item in episodes)
    knight_ci = wilson_interval(knight_wins, total, confidence=confidence)
    boss_ci = wilson_interval(boss_wins, total, confidence=confidence)
    draw_ci = wilson_interval(draws, total, confidence=confidence)
    return EvaluationSummary(
        episodes=total,
        confidence_level=confidence,
        knight_wins=knight_wins,
        boss_wins=boss_wins,
        draws=draws,
        timeouts=timeouts,
        knight_win_rate=knight_wins / total,
        boss_win_rate=boss_wins / total,
        draw_rate=draws / total,
        timeout_rate=timeouts / total,
        knight_win_ci_lower=knight_ci[0],
        knight_win_ci_upper=knight_ci[1],
        boss_win_ci_lower=boss_ci[0],
        boss_win_ci_upper=boss_ci[1],
        draw_ci_lower=draw_ci[0],
        draw_ci_upper=draw_ci[1],
        average_episode_length=sum(item.episode_length for item in episodes) / total,
        average_knight_damage_dealt=(
            sum(item.knight_damage_dealt for item in episodes) / total
        ),
        average_knight_damage_received=(
            sum(item.knight_damage_received for item in episodes) / total
        ),
        average_boss_damage_dealt=(
            sum(item.boss_damage_dealt for item in episodes) / total
        ),
        average_boss_damage_received=(
            sum(item.boss_damage_received for item in episodes) / total
        ),
        average_invalid_knight_moves=(
            sum(item.invalid_knight_moves for item in episodes) / total
        ),
        average_special2_casts=sum(item.special2_casts for item in episodes) / total,
        average_special2_hits=sum(item.special2_hits for item in episodes) / total,
        average_special2_blocks=sum(item.special2_blocks for item in episodes) / total,
        average_special2_escapes=sum(item.special2_escapes for item in episodes) / total,
        average_knight_heal_noops=sum(item.knight_heal_noops for item in episodes) / total,
        average_knight_heals=sum(item.knight_heals for item in episodes) / total,
        average_knight_attack_attempts=(
            sum(item.knight_attack_attempts for item in episodes) / total
        ),
        average_knight_attack_hits=(
            sum(item.knight_attack_hits for item in episodes) / total
        ),
        special2_hit_rate=(
            sum(item.special2_hits for item in episodes) / special2_casts
            if special2_casts
            else 0.0
        ),
        special2_block_rate=(
            sum(item.special2_blocks for item in episodes) / special2_casts
            if special2_casts
            else 0.0
        ),
        invalid_knight_moves_per_step=(
            sum(item.invalid_knight_moves for item in episodes)
            / sum(item.episode_length for item in episodes)
            if sum(item.episode_length for item in episodes)
            else 0.0
        ),
    )
