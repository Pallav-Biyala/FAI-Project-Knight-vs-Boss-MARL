from knight_boss.game.config import GameConfig
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.visualization.battle_replay import (
    BattleReplayRecorder,
    load_replay,
    render_replay_state,
)


def test_record_load_and_render_complete_battle(tmp_path):
    env = KnightBossParallelEnv(
        GameConfig(
            knight_position=(4, 4),
            boss_position=(4, 5),
            max_steps=10,
        ),
        alpha_h=0.05,
        render_mode="ansi",
    )
    env.reset(seed=5)
    recorder = BattleReplayRecorder(
        env.engine,
        seed=5,
        policies={"knight": "heuristic", "boss": "heuristic"},
    )
    observations, rewards, terminations, truncations, infos = env.step(
        {"knight": 4, "boss": 6}
    )
    recorder.record_step(
        env.engine,
        {"knight": 4, "boss": 6},
        rewards,
        infos,
        terminations,
        truncations,
    )
    path = recorder.save(tmp_path / "battle.json")
    replay = load_replay(path)
    assert replay["alpha_h"] == 0.05
    assert len(replay["frames"]) == 1
    assert replay["frames"][0]["actions"]["knight"] == "BASIC_ATTACK"
    assert replay["frames"][0]["state"]["outcome"] == "knight_win"
    initial = render_replay_state(replay, replay["initial_state"])
    final = render_replay_state(
        replay,
        replay["frames"][0]["state"],
        actions=replay["frames"][0]["actions"],
        events=replay["frames"][0]["events"],
    )
    assert "K" in initial and "B" in initial
    assert "range 2" in initial
    assert "Actions: Knight=BASIC_ATTACK" in final
    assert "Outcome: knight_win" in final
