from pettingzoo.test import parallel_api_test
import pytest

from knight_boss.evaluation.evaluator import evaluate_policies
from knight_boss.game.config import GameConfig
from knight_boss.game.types import Outcome
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.proposal_env import (
    ProposalBossAction,
    ProposalCombatEnv,
    ProposalCommonAction,
    ProposalHeuristicBoss,
    ProposalHeuristicKnight,
    ProposalKnightAction,
    ProposalRandomAgent,
)
from knight_boss.training.config import PPOConfig
from knight_boss.training.trainer import MultiAgentPPOTrainer


@pytest.mark.parametrize(
    ("enable_dodge", "enable_heal"),
    [(True, True), (False, False)],
)
def test_proposal_env_passes_pettingzoo_parallel_api(
    enable_dodge: bool, enable_heal: bool
):
    env = ProposalCombatEnv(
        GameConfig(),
        enable_dodge=enable_dodge,
        enable_heal=enable_heal,
    )
    obs, _ = env.reset(seed=7)
    assert observation_to_vector(obs["knight"]).size == 44
    parallel_api_test(env, num_cycles=60)


def test_knight_dodge_grants_u_k_invincibility_frames_across_two_steps():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        invincibility_frames=2,
        dodge_cooldown=4,
    )
    obs, _ = env.reset(seed=0)
    assert obs["knight"]["proposal_state"][0] == pytest.approx(0.0)

    # Step 0: Knight uses DODGE while Boss lands ATTACK -> negated (u_K=2 -> 1 at end).
    obs, rewards, _, _, info = env.step(
        {"knight": ProposalKnightAction.DODGE, "boss": ProposalBossAction.ATTACK}
    )
    assert env.hp["knight"] == 150
    assert env.knight_invuln == 1
    assert "knight_dodged_damage" in info["knight"]["events"]
    assert obs["knight"]["proposal_state"][0] == pytest.approx(0.5)
    assert rewards["knight"] == pytest.approx(0.0)

    # Step 1: Boss WAITS (basic cooldown ticks 1->0), Knight WAITS (u_K ticks 1->0).
    env.step({"knight": ProposalCommonAction.WAIT, "boss": ProposalCommonAction.WAIT})
    assert env.knight_invuln == 0

    # Step 2: Knight DODGE is still on cooldown (cd=2), so Boss basic attack deals 20 dmg!
    env.step({"knight": ProposalKnightAction.DODGE, "boss": ProposalBossAction.ATTACK})
    assert env.hp["knight"] == 130


def test_knight_heal_charges_q_k_cap_and_exhaustion():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        heal_amount=30,
        heal_charges=3,
    )
    env.reset(seed=0)
    env.hp["knight"] = 50

    for expected_hp, expected_charges in [(80, 2), (110, 1), (140, 0)]:
        env.step({"knight": ProposalKnightAction.HEAL, "boss": ProposalCommonAction.WAIT})
        assert env.hp["knight"] == expected_hp
        assert env.knight_heal_charges == expected_charges

    # 4th HEAL with 0 charges left must be a no-op.
    _, _, _, _, info = env.step(
        {"knight": ProposalKnightAction.HEAL, "boss": ProposalCommonAction.WAIT}
    )
    assert env.hp["knight"] == 140
    assert "knight_heal_noop" in info["knight"]["events"]


def test_boss_special2_shockwave_phase_z_t_hits_and_shares_cooldown():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 6)),
        alpha_d=1.5,
    )
    env.reset(seed=0)
    # Distance is 2; Boss initiates SPECIAL_2 shockwave windup -> shared cooldown c_B = 2.
    obs, _, _, _, info = env.step(
        {"knight": ProposalCommonAction.WAIT, "boss": ProposalBossAction.SPECIAL_2}
    )
    assert env.atk_cd["boss"] == 2
    assert "boss_special2_windup" in info["knight"]["events"]
    assert obs["knight"]["proposal_state"][3] == pytest.approx(1.0)

    # Cannot immediately cast normal ATTACK while shared cooldown c_B > 0; shockwave fires!
    # At alpha_d=1.5, special2 dmg = round(40 * 1.5) = 60.
    env.step(
        {"knight": ProposalCommonAction.UP, "boss": ProposalBossAction.ATTACK}
    )
    # Knight stepped UP from (4,4) to (3,4), still within Chebyshev <= 2 of (4,6) -> hit!
    assert env.hp["knight"] == 150 - 60


def test_proposal_baselines_ppo_trainer_and_evaluator_work_end_to_end():
    cfg = GameConfig(max_steps=40)
    env = ProposalCombatEnv(cfg, alpha_h=1.5, alpha_d=1.5, reward_mode="shaped")
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
    trainer.policies["boss"] = ProposalHeuristicBoss(seed=2)
    trainer.collect_rollout(16, trainable_agents=("knight",))

    _, summary = evaluate_policies(
        {"knight": trainer.policies["knight"], "boss": ProposalHeuristicBoss(seed=3)},
        cfg,
        episodes=3,
        seed=10,
        env_factory=lambda: ProposalCombatEnv(cfg, alpha_h=1.5, alpha_d=1.5),
    )
    assert summary.episodes == 3

    _, calib = evaluate_policies(
        {"knight": ProposalHeuristicKnight(seed=4), "boss": ProposalHeuristicBoss(seed=5)},
        cfg,
        episodes=3,
        seed=20,
        env_factory=lambda: ProposalCombatEnv(cfg, alpha_h=1.0, alpha_d=1.0),
    )
    assert calib.episodes == 3

    _, rand_sum = evaluate_policies(
        {"knight": ProposalRandomAgent(seed=6), "boss": ProposalHeuristicBoss(seed=7)},
        cfg,
        episodes=3,
        seed=30,
        env_factory=lambda: ProposalCombatEnv(cfg, alpha_h=1.0, alpha_d=1.0),
    )
    assert rand_sum.episodes == 3


def test_proposal_draw_penalty_applies_only_on_timeout_or_draw():
    with pytest.raises(ValueError, match="draw_penalty"):
        ProposalCombatEnv(reward_mode="draw_penalty", draw_penalty=-0.1)

    env = ProposalCombatEnv(
        GameConfig(max_steps=2),
        reward_mode="draw_penalty",
        draw_penalty=1.0,
    )
    env.reset(seed=0)
    _, r1, term1, trunc1, _ = env.step(
        {"knight": ProposalCommonAction.WAIT, "boss": ProposalCommonAction.WAIT}
    )
    assert term1["knight"] is False and trunc1["knight"] is False
    assert r1["knight"] == pytest.approx(0.0)
    assert r1["boss"] == pytest.approx(0.0)

    _, r2, _, trunc2, _ = env.step(
        {"knight": ProposalCommonAction.WAIT, "boss": ProposalCommonAction.WAIT}
    )
    assert trunc2["knight"] is True
    assert env.outcome is Outcome.DRAW
    assert r2["knight"] == pytest.approx(-1.0)
    assert r2["boss"] == pytest.approx(-1.0)
