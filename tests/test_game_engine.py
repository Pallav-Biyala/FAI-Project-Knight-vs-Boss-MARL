from dataclasses import replace

import pytest

from knight_boss.game.board import Board
from knight_boss.game.config import FighterConfig, GameConfig
from knight_boss.game.engine import GameEngine
from knight_boss.game.types import Action, Outcome


def adjacent_config(**overrides):
    base = GameConfig(knight_position=(4, 4), boss_position=(4, 5))
    return replace(base, **overrides)


def wait_actions():
    return {"knight": Action.WAIT, "boss": Action.WAIT}


def test_board_boundaries_and_distance():
    board = Board(9, 9)
    assert board.in_bounds((0, 0))
    assert not board.in_bounds((-1, 0))
    assert not board.in_bounds((9, 8))
    assert board.manhattan_distance((1, 2), (4, 6)) == 7


def test_boundary_move_does_not_change_position():
    engine = GameEngine(GameConfig(knight_position=(0, 0), boss_position=(8, 8)))
    engine.step({"knight": Action.UP, "boss": Action.WAIT})
    assert engine.fighters["knight"].position == (0, 0)


def test_same_empty_cell_collision_cancels_both_moves():
    engine = GameEngine(GameConfig(knight_position=(4, 3), boss_position=(4, 5)))
    engine.step({"knight": Action.RIGHT, "boss": Action.LEFT})
    assert engine.fighters["knight"].position == (4, 3)
    assert engine.fighters["boss"].position == (4, 5)


def test_exchange_positions_is_rejected():
    engine = GameEngine(adjacent_config())
    engine.step({"knight": Action.RIGHT, "boss": Action.LEFT})
    assert engine.fighters["knight"].position == (4, 4)
    assert engine.fighters["boss"].position == (4, 5)


def test_blocked_move_does_not_crash_or_overlap():
    engine = GameEngine(adjacent_config())
    engine.step({"knight": Action.RIGHT, "boss": Action.WAIT})
    assert engine.fighters["knight"].position == (4, 4)
    assert engine.fighters["boss"].position == (4, 5)


def test_basic_attack_applies_configured_damage():
    engine = GameEngine(adjacent_config())
    engine.step({"knight": Action.BASIC_ATTACK, "boss": Action.WAIT})
    assert engine.fighters["boss"].health == 130
    assert engine.stats.damage_dealt["knight"] == 20


def test_basic_attack_cooldown_blocks_next_action_then_expires():
    engine = GameEngine(adjacent_config())
    engine.step({"knight": Action.BASIC_ATTACK, "boss": Action.WAIT})
    assert engine.fighters["knight"].basic_cooldown == 1
    engine.step({"knight": Action.BASIC_ATTACK, "boss": Action.WAIT})
    assert engine.fighters["boss"].health == 130
    assert engine.fighters["knight"].basic_cooldown == 0
    engine.step({"knight": Action.BASIC_ATTACK, "boss": Action.WAIT})
    assert engine.fighters["boss"].health == 110


def test_heavy_attack_requires_cardinal_line_and_winds_up():
    config = GameConfig(knight_position=(3, 3), boss_position=(4, 4))
    engine = GameEngine(config)
    result = engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    assert "knight_heavy_out_of_range" in result.events
    assert engine.fighters["boss"].health == 150

    engine = GameEngine(GameConfig(knight_position=(4, 3), boss_position=(4, 5)))
    result = engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    assert "knight_heavy_windup" in result.events
    assert engine.fighters["boss"].health == 150


def test_heavy_attack_hits_if_target_remains_in_committed_direction():
    engine = GameEngine(GameConfig(knight_position=(4, 3), boss_position=(4, 5)))
    engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    result = engine.step(wait_actions())
    assert "knight_heavy_hit" in result.events
    assert engine.fighters["boss"].health == 110


def test_heavy_attack_misses_after_target_reacts_out_of_line():
    engine = GameEngine(GameConfig(knight_position=(4, 3), boss_position=(4, 5)))
    engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    result = engine.step({"knight": Action.WAIT, "boss": Action.UP})
    assert "knight_heavy_missed" in result.events
    assert engine.fighters["boss"].health == 150


def test_heavy_attack_cooldown_does_not_expire_on_announcement_step():
    engine = GameEngine(GameConfig(knight_position=(4, 3), boss_position=(4, 5)))
    engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    assert engine.fighters["knight"].heavy_cooldown == 3
    engine.step(wait_actions())
    assert engine.fighters["knight"].heavy_cooldown == 2


def test_heavy_attack_cooldown_blocks_then_becomes_available():
    engine = GameEngine(GameConfig(knight_position=(4, 3), boss_position=(4, 5)))
    engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    engine.step(wait_actions())
    engine.step(wait_actions())
    result = engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    assert "knight_heavy_windup" not in result.events
    result = engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    assert "knight_heavy_windup" in result.events


def test_configured_heavy_windup_waits_the_requested_steps():
    config = GameConfig(
        knight_position=(4, 3), boss_position=(4, 5), heavy_windup_steps=2
    )
    engine = GameEngine(config)
    engine.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    result = engine.step(wait_actions())
    assert "knight_heavy_hit" not in result.events
    assert engine.fighters["boss"].health == 150
    result = engine.step(wait_actions())
    assert "knight_heavy_hit" in result.events
    assert engine.fighters["boss"].health == 110


def test_simultaneous_attacks_both_deal_damage():
    engine = GameEngine(adjacent_config())
    engine.step({"knight": Action.BASIC_ATTACK, "boss": Action.BASIC_ATTACK})
    assert engine.fighters["knight"].health == 130
    assert engine.fighters["boss"].health == 130


def test_simultaneous_death_is_a_draw():
    one_hit = FighterConfig(max_health=20, basic_damage=20)
    config = adjacent_config(knight=one_hit)
    engine = GameEngine(config)
    result = engine.step({"knight": Action.BASIC_ATTACK, "boss": Action.BASIC_ATTACK})
    assert result.terminated
    assert not result.truncated
    assert result.outcome is Outcome.DRAW


def test_wins_and_timeout_are_distinguished():
    engine = GameEngine(adjacent_config(), alpha_h=0.05)
    result = engine.step({"knight": Action.BASIC_ATTACK, "boss": Action.WAIT})
    assert result.outcome is Outcome.KNIGHT_WIN
    assert result.terminated and not result.truncated

    engine = GameEngine(adjacent_config(max_steps=1))
    result = engine.step(wait_actions())
    assert result.outcome is Outcome.DRAW
    assert result.truncated and not result.terminated


def test_alpha_parameters_are_independent():
    engine = GameEngine(alpha_h=2.0, alpha_d=1.5)
    assert engine.fighters["boss"].max_health == 300
    assert engine.fighters["boss"].basic_damage == 30
    assert engine.fighters["boss"].heavy_damage == 60


def test_boss_strength_is_derived_from_knight_base_statistics():
    knight = FighterConfig(max_health=200, basic_damage=11, heavy_damage=31)
    config = GameConfig(knight=knight, alpha_h=1.5, alpha_d=2.0)
    engine = GameEngine(config)
    boss = engine.fighters["boss"]
    assert boss.max_health == 300
    assert boss.basic_damage == 22
    assert boss.heavy_damage == 62


def test_yaml_configuration_loads(tmp_path):
    config_file = tmp_path / "game.yaml"
    config_file.write_text(
        "board:\n  rows: 9\n  columns: 9\nalpha_h: 2\nalpha_d: 0.5\n",
        encoding="utf-8",
    )
    config = GameConfig.from_yaml(config_file)
    engine = GameEngine(config)
    assert engine.fighters["boss"].max_health == 300
    assert engine.fighters["boss"].basic_damage == 10


def test_invalid_action_fails_explicitly():
    engine = GameEngine()
    with pytest.raises(ValueError, match="Invalid action"):
        engine.step({"knight": 99, "boss": Action.WAIT})


def test_completed_episode_cannot_be_advanced():
    engine = GameEngine(adjacent_config(max_steps=1))
    engine.step(wait_actions())
    with pytest.raises(RuntimeError, match="completed episode"):
        engine.step(wait_actions())
