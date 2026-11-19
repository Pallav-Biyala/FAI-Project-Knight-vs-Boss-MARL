"""Unit tests for potential-based health shaping and reward modes."""

import pytest

from knight_boss.game.types import Outcome
from knight_boss.marl.rewards import (
    REWARD_MODES,
    combat_rewards,
    draw_penalty_rewards,
    shaped_rewards,
    terminal_rewards,
)

KNIGHT_HP = 150
BOSS_HP = 150


def _common(**overrides):
    values = {
        "prev_knight_health": 150,
        "curr_knight_health": 150,
        "prev_boss_health": 150,
        "curr_boss_health": 150,
        "knight_max_health": KNIGHT_HP,
        "boss_max_health": BOSS_HP,
        "shaping_weight": 0.5,
        "gamma": 0.995,
    }
    values.update(overrides)
    return values


def _phi(knight_hp, boss_hp, weight=0.5):
    return weight * (knight_hp / KNIGHT_HP - boss_hp / BOSS_HP)


def test_reward_modes_cover_all_four_supported_modes():
    assert set(REWARD_MODES) == {"terminal", "shaped", "draw_penalty", "combat"}


def test_terminal_rewards_assign_win_loss_and_neutral_draw():
    assert terminal_rewards(Outcome.KNIGHT_WIN) == {"knight": 1.0, "boss": -1.0}
    assert terminal_rewards(Outcome.BOSS_WIN) == {"knight": -1.0, "boss": 1.0}
    assert terminal_rewards(Outcome.DRAW) == {"knight": 0.0, "boss": 0.0}
    assert terminal_rewards(None) == {"knight": 0.0, "boss": 0.0}


def test_shaped_rewards_are_zero_sum_for_every_outcome():
    for outcome in (None, Outcome.DRAW, Outcome.KNIGHT_WIN, Outcome.BOSS_WIN):
        rewards = shaped_rewards(
            outcome,
            **_common(
                prev_knight_health=120,
                curr_knight_health=100,
                prev_boss_health=130,
                curr_boss_health=110,
            ),
        )
        assert rewards["knight"] + rewards["boss"] == pytest.approx(0.0)


def test_nonterminal_shaping_uses_discounted_potential_difference():
    common = _common(
        prev_knight_health=120,
        curr_knight_health=100,
        prev_boss_health=140,
        curr_boss_health=110,
        gamma=0.9,
        shaping_weight=0.5,
    )
    expected = 0.9 * _phi(100, 110) - _phi(120, 140)

    rewards = shaped_rewards(None, **common)

    assert rewards["knight"] == pytest.approx(expected)
    assert rewards["boss"] == pytest.approx(-expected)


def test_healing_restores_health_potential():
    # Knight starts damaged, then heals from 100 to 130 HP.
    damaged = shaped_rewards(
        None,
        **_common(
            prev_knight_health=150,
            curr_knight_health=100,
        ),
    )
    healed = shaped_rewards(
        None,
        **_common(
            prev_knight_health=100,
            curr_knight_health=130,
        ),
    )

    assert healed["knight"] > damaged["knight"]
    assert healed["knight"] > 0.0
    assert healed["boss"] == pytest.approx(-healed["knight"])


def test_terminal_transition_uses_zero_next_potential():
    common = _common(
        prev_knight_health=120,
        curr_knight_health=150,
        prev_boss_health=150,
        curr_boss_health=0,
    )
    expected_shaping = -_phi(120, 150)

    rewards = shaped_rewards(Outcome.KNIGHT_WIN, **common)

    assert rewards["knight"] == pytest.approx(1.0 + expected_shaping)
    assert rewards["boss"] == pytest.approx(-1.0 - expected_shaping)


def test_truncated_transition_uses_zero_next_potential():
    common = _common(
        prev_knight_health=120,
        curr_knight_health=150,
        prev_boss_health=150,
        curr_boss_health=100,
    )
    expected_shaping = -_phi(120, 150)

    rewards = shaped_rewards(None, truncated=True, **common)

    assert rewards["knight"] == pytest.approx(expected_shaping)
    assert rewards["boss"] == pytest.approx(-expected_shaping)


def test_discounted_episode_shaping_telescopes():
    gamma = 0.995
    states = [
        (150, 150),
        (130, 120),
        (110, 90),
    ]

    rewards = []
    for index in range(len(states) - 1):
        prev_k, prev_b = states[index]
        curr_k, curr_b = states[index + 1]
        result = shaped_rewards(
            None,
            **_common(
                prev_knight_health=prev_k,
                curr_knight_health=curr_k,
                prev_boss_health=prev_b,
                curr_boss_health=curr_b,
                gamma=gamma,
            ),
        )
        rewards.append(result["knight"])

    prev_k, prev_b = states[-1]
    terminal = shaped_rewards(
        Outcome.KNIGHT_WIN,
        **_common(
            prev_knight_health=prev_k,
            curr_knight_health=0,
            prev_boss_health=prev_b,
            curr_boss_health=0,
            gamma=gamma,
        ),
    )
    rewards.append(terminal["knight"] - 1.0)

    discounted_total = sum(
        (gamma**index) * reward for index, reward in enumerate(rewards)
    )
    assert discounted_total == pytest.approx(0.0, abs=1e-9)


def test_terminal_win_cannot_be_overturned_by_the_whole_shaping_budget():
    weight = 0.5
    win_worst_case = 1.0 - weight
    loss_best_case = -1.0 + weight
    assert win_worst_case > loss_best_case


@pytest.mark.parametrize("truncated", [False, True])
def test_draw_penalty_treats_draw_and_timeout_identically(truncated):
    rewards = draw_penalty_rewards(
        Outcome.DRAW,
        truncated=truncated,
        **_common(),
        penalty=0.75,
    )
    assert rewards == {"knight": -0.75, "boss": -0.75}


def test_draw_penalty_leaves_decisive_outcomes_unchanged_when_potential_is_zero():
    for outcome in (Outcome.KNIGHT_WIN, Outcome.BOSS_WIN):
        rewards = draw_penalty_rewards(
            outcome,
            truncated=False,
            **_common(),
        )
        assert rewards == terminal_rewards(outcome)


def test_draw_stays_better_than_a_loss_under_the_default_penalty():
    draw = draw_penalty_rewards(
        Outcome.DRAW,
        truncated=True,
        **_common(),
        penalty=0.75,
    )
    loss = terminal_rewards(Outcome.BOSS_WIN)
    assert draw["knight"] > loss["knight"]


@pytest.mark.parametrize("truncated", [False, True])
def test_combat_penalizes_draw_and_timeout_identically(truncated):
    rewards = combat_rewards(
        Outcome.DRAW,
        truncated=truncated,
        **_common(),
        prev_distance=4,
        curr_distance=4,
        timeout_penalty=0.5,
        step_penalty=0.005,
    )
    assert rewards["knight"] == pytest.approx(-0.505)
    assert rewards["boss"] == pytest.approx(-0.505)


def test_draw_and_timeout_are_consistent_between_draw_penalty_and_combat():
    def compare(is_truncated):
        common = _common()
        dp = draw_penalty_rewards(
            Outcome.DRAW,
            truncated=is_truncated,
            **common,
            penalty=0.5,
        )
        cb = combat_rewards(
            Outcome.DRAW,
            truncated=is_truncated,
            **common,
            prev_distance=4,
            curr_distance=4,
            timeout_penalty=0.5,
            step_penalty=0.0,
            distance_weight=0.0,
        )
        return dp["knight"], cb["knight"]

    assert compare(False) == pytest.approx(compare(True))


def test_combat_charges_a_living_cost_while_standing_still():
    rewards = combat_rewards(
        None,
        **_common(),
        prev_distance=4,
        curr_distance=4,
    )
    assert rewards["knight"] == pytest.approx(-0.005)
    assert rewards["boss"] == pytest.approx(-0.005)


def test_combat_makes_a_full_stall_worse_than_a_decisive_loss():
    per_step = combat_rewards(
        None,
        **_common(),
        prev_distance=4,
        curr_distance=4,
    )["knight"]
    final_step = combat_rewards(
        None,
        truncated=True,
        **_common(),
        prev_distance=4,
        curr_distance=4,
    )["knight"]

    full_stall = 300 * per_step + final_step
    assert full_stall < terminal_rewards(Outcome.BOSS_WIN)["knight"]


def test_combat_time_cost_cannot_invert_win_and_loss():
    step_penalty = 0.005
    max_steps = 300
    worst_case_win = 1.0 - step_penalty * max_steps
    best_case_loss = -1.0
    assert worst_case_win > best_case_loss


def test_combat_approach_term_is_symmetric():
    retreat = combat_rewards(
        None,
        **_common(),
        prev_distance=1,
        curr_distance=2,
    )
    approach = combat_rewards(
        None,
        **_common(),
        prev_distance=6,
        curr_distance=5,
    )

    assert retreat["knight"] == retreat["boss"] == pytest.approx(-0.015)
    assert approach["knight"] == approach["boss"] == pytest.approx(0.005)
