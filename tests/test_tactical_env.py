from pettingzoo.test import parallel_api_test
import pytest

from knight_boss.game.config import GameConfig
from knight_boss.game.types import Action, Outcome
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.tactical_env import TacticalHeuristicAgent, TacticalParallelEnv
from knight_boss.training.config import PPOConfig
from knight_boss.training.trainer import MultiAgentPPOTrainer
from knight_boss.evaluation.evaluator import evaluate_policies


@pytest.mark.parametrize(
    ("enable_dodge", "enable_heal", "expected_actions"),
    [
        (False, False, 7),
        (True, False, 8),
        (False, True, 8),
        (True, True, 9),
    ],
)
def test_tactical_env_passes_parallel_api_test_for_all_2x2_combinations(
    enable_dodge: bool, enable_heal: bool, expected_actions: int
):
    env = TacticalParallelEnv(
        GameConfig(),
        enable_dodge=enable_dodge,
        enable_heal=enable_heal,
    )
    assert env.action_space("knight").n == expected_actions
    obs, _ = env.reset(seed=5)
    assert observation_to_vector(obs["knight"]).size == 41
    parallel_api_test(env, num_cycles=50)


def test_dodge_negates_damage_and_respects_cooldown():
    env = TacticalParallelEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        enable_dodge=True,
        enable_heal=False,
        dodge_cooldown=3,
    )
    obs, _ = env.reset(seed=0)
    assert env.dodge_action_id == 7

    # Boss basic attacks while Knight dodges -> 0 damage dealt/received.
    obs, _, _, _, info = env.step({"knight": 7, "boss": Action.BASIC_ATTACK})
    assert env.engine.fighters["knight"].health == 150
    assert env.engine.damage_received["knight"] == 0
    assert "knight_dodged_hit" in info["knight"]["events"]
    assert obs["knight"]["tactical_stats"][0] == pytest.approx(1.0)

    # Next step Dodge is on cooldown (cd decrements from 3 -> 2), so Boss hits when Boss cools down.
    env.step({"knight": Action.WAIT, "boss": Action.WAIT})  # boss cd 1->0, knight dodge cd 3->2
    obs, _, _, _, _ = env.step({"knight": 7, "boss": Action.BASIC_ATTACK})  # dodge blocked by cd!
    assert env.engine.fighters["knight"].health == 130


def test_heal_restores_hp_respects_charges_and_saves_from_fatal_net_positive():
    env = TacticalParallelEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        enable_dodge=True,
        enable_heal=True,
        heal_amount=30,
        heal_cooldown=2,
        heal_charges=2,
    )
    env.reset(seed=0)
    assert env.heal_action_id == 8

    # Bring Knight to 20 HP cleanly: take basic attacks (20 dmg) + waits.
    for _ in range(6):
        env.step({"knight": Action.WAIT, "boss": Action.BASIC_ATTACK})
        env.step({"knight": Action.WAIT, "boss": Action.WAIT})
    # Wait one extra turn so Knight has 30 HP? Actually 6*20 = 120 dmg -> Knight HP = 30.
    assert env.engine.fighters["knight"].health == 30

    # Take one more basic attack (20 dmg) -> 10 HP, Boss basic cooldown resets next step.
    env.step({"knight": Action.WAIT, "boss": Action.BASIC_ATTACK})
    assert env.engine.fighters["knight"].health == 10
    env.step({"knight": Action.WAIT, "boss": Action.WAIT})

    # Now Knight heals (+30) on the same step Boss hits for 20 -> simultaneous net HP = 10 - 20 + 30 = 20!
    obs, _, term, _, _ = env.step({"knight": 8, "boss": Action.BASIC_ATTACK})
    assert not term["knight"]
    assert env.engine.outcome is None
    assert env.engine.fighters["knight"].health == 20
    assert obs["knight"]["tactical_stats"][4] == pytest.approx(0.5)  # 1 of 2 charges left


def test_tactical_ppo_trainer_and_evaluator_work_end_to_end():
    cfg = GameConfig(max_steps=40)
    env = TacticalParallelEnv(
        cfg, enable_dodge=True, enable_heal=True, reward_mode="shaped"
    )
    ppo = PPOConfig(
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=1,
        batch_size=8,
        rollout_steps=16,
        hidden_sizes=(16,),
        max_grad_norm=0.5,
        device="cpu",
    )
    trainer = MultiAgentPPOTrainer(env, ppo, seed=1)
    trainer.policies["boss"] = TacticalHeuristicAgent(
        dodge_action_id=env.dodge_action_id,
        heal_action_id=env.heal_action_id,
        seed=2,
    )
    trainer.collect_rollout(16, trainable_agents=("knight",))
    _, summary = evaluate_policies(
        trainer.policies,
        cfg,
        episodes=3,
        seed=100,
        env_factory=lambda: TacticalParallelEnv(
            cfg, enable_dodge=True, enable_heal=True
        ),
    )
    assert summary.episodes == 3


def test_tactical_heuristic_triggers_dodge_on_heavy_windup_and_heal_low_hp():
    env = TacticalParallelEnv(
        GameConfig(knight_position=(4, 3), boss_position=(4, 5)),
        enable_dodge=True,
        enable_heal=True,
        heal_cooldown=2,
        heal_charges=2,
    )
    obs, _ = env.reset(seed=0)
    heuristic_boss = TacticalHeuristicAgent(
        dodge_action_id=env.dodge_action_id,
        heal_action_id=env.heal_action_id,
        seed=0,
    )
    # Knight starts heavy attack in range (distance 2, aligned) -> Boss sees opponent windup.
    obs, _, _, _, _ = env.step({"knight": Action.HEAVY_ATTACK, "boss": Action.WAIT})
    assert obs["boss"]["windups"][4] == pytest.approx(1.0)
    assert obs["boss"]["windups"][7] == pytest.approx(1.0)
    chosen = heuristic_boss.act(obs["boss"])
    assert chosen == env.dodge_action_id

    # Lower Boss HP below 50% and clear Knight pending heavy -> Boss should cast HEAL.
    env.reset(seed=1)
    env.engine.fighters["boss"].health = 60  # 60 / 150 = 0.40 <= 0.50
    obs = {
        "knight": env._observe("knight"),
        "boss": env._observe("boss"),
    }
    assert heuristic_boss.act(obs["boss"]) == env.heal_action_id
