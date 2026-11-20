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
    assert obs["knight"]["action_mask"].shape == (8,)
    assert env.observation_space("knight").contains(obs["knight"])
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


def test_dodged_special_is_classified_as_block_not_hit_under_tactical_reward():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        reward_mode="tactical",
    )
    env.reset(seed=0)
    # Boss telegraphs SPECIAL_2 (shared cooldown c_B = 2) while the Knight waits.
    env.step(
        {"knight": ProposalCommonAction.WAIT, "boss": ProposalBossAction.SPECIAL_2}
    )
    # Knight DODGEs on the firing step, so u_K negates the shockwave on contact.
    _, rewards, _, _, info = env.step(
        {"knight": ProposalKnightAction.DODGE, "boss": ProposalCommonAction.WAIT}
    )
    events = info["knight"]["events"]
    assert "knight_dodged_damage" in events
    assert "boss_special2_blocked" in events
    assert "boss_special2_hit" not in events
    assert env.hp["knight"] == env.knight_max_hp
    assert rewards["knight"] >= 0.0
    assert info["knight"]["proposal_events"]["boss_special2_hit"] == 0
    assert info["knight"]["proposal_events"]["boss_special2_blocked"] == 1


def test_unblocked_special_still_emits_hit_and_penalty_under_tactical_reward():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        reward_mode="tactical",
    )
    env.reset(seed=0)
    env.step(
        {"knight": ProposalCommonAction.WAIT, "boss": ProposalBossAction.SPECIAL_2}
    )
    _, rewards, _, _, info = env.step(
        {"knight": ProposalCommonAction.WAIT, "boss": ProposalCommonAction.WAIT}
    )
    events = info["knight"]["events"]
    assert "boss_special2_hit" in events
    assert "knight_dodged_damage" not in events
    assert env.hp["knight"] < env.knight_max_hp
    assert rewards["knight"] < 0.0
    assert info["knight"]["proposal_events"]["boss_special2_hit"] == 1


def test_blocked_boundary_moves_are_masked_and_reported_without_moving():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(0, 0), boss_position=(8, 8))
    )
    observations, _ = env.reset(seed=0)
    mask = observations["knight"]["action_mask"]
    assert mask[ProposalCommonAction.UP] == 0
    assert mask[ProposalCommonAction.LEFT] == 0
    assert mask[ProposalCommonAction.WAIT] == 1

    for _ in range(3):
        _, _, _, _, info = env.step(
            {"knight": ProposalCommonAction.UP, "boss": ProposalCommonAction.WAIT}
        )
        assert env.pos["knight"] == (0, 0)
        assert "knight_blocked_movement" in info["knight"]["events"]
    assert info["knight"]["proposal_events"]["knight_blocked_movement"] == 3


def test_heal_mask_disables_full_health_and_no_charges_but_allows_injured_heal():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5)),
        heal_charges=1,
    )
    observations, _ = env.reset(seed=0)
    assert observations["knight"]["action_mask"][ProposalKnightAction.HEAL] == 0

    env.hp["knight"] -= 20
    observation = env._observe("knight")
    assert observation["action_mask"][ProposalKnightAction.HEAL] == 1
    env.step({"knight": ProposalKnightAction.HEAL, "boss": ProposalCommonAction.WAIT})
    assert env.hp["knight"] == env.knight_max_hp
    assert env.knight_heal_charges == 0
    assert env._observe("knight")["action_mask"][ProposalKnightAction.HEAL] == 0


def test_special2_warning_exposes_threat_and_allows_dodge_or_escape():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(0, 2), boss_position=(0, 0)),
        special2_radius=2,
    )
    env.reset(seed=0)
    observations, _, _, _, _ = env.step(
        {"knight": ProposalCommonAction.WAIT, "boss": ProposalBossAction.SPECIAL_2}
    )
    knight_obs = observations["knight"]
    assert knight_obs["proposal_state"][3] == pytest.approx(1.0)
    assert knight_obs["windups"][7] == pytest.approx(1.0)
    assert knight_obs["action_mask"][ProposalKnightAction.DODGE] == 1
    assert knight_obs["action_mask"][ProposalCommonAction.RIGHT] == 1

    observations, _, _, _, info = env.step(
        {"knight": ProposalCommonAction.RIGHT, "boss": ProposalCommonAction.WAIT}
    )
    assert env.hp["knight"] == env.knight_max_hp
    assert "knight_escaped_special2" in info["knight"]["events"]
    assert info["knight"]["proposal_events"]["knight_escaped_special2"] == 1


def test_attack_and_defensive_choices_remain_available_together():
    env = ProposalCombatEnv(
        GameConfig(knight_position=(4, 4), boss_position=(4, 5))
    )
    observations, _ = env.reset(seed=0)
    mask = observations["knight"]["action_mask"]
    assert mask[ProposalKnightAction.ATTACK] == 1
    assert mask[ProposalKnightAction.DODGE] == 1
    assert mask[ProposalKnightAction.HEAL] == 0
