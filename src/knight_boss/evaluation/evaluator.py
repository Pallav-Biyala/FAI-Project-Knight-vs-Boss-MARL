"""Independent match evaluation for trained policies and baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from knight_boss.agents import HeuristicAgent, RandomAgent
from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.evaluation.metrics import (
    EpisodeMetrics,
    EvaluationSummary,
    summarize_episodes,
)
from knight_boss.game.config import GameConfig
from knight_boss.game.types import Action
from knight_boss.marl.parallel_env import KnightBossParallelEnv


def evaluate_policies(
    policies: Mapping[str, object],
    game_config: GameConfig,
    *,
    episodes: int,
    seed: int,
    alpha_h: float | None = None,
    alpha_d: float | None = None,
    deterministic: bool = False,
    confidence: float = 0.95,
    env_factory: Callable[[], object] | None = None,
) -> tuple[list[EpisodeMetrics], EvaluationSummary]:
    """Evaluate one fixed policy pair without any policy updates."""
    if set(policies) != {"knight", "boss"}:
        raise ValueError("Policies must be provided for exactly knight and boss.")
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    env = (
        env_factory()
        if env_factory is not None
        else KnightBossParallelEnv(game_config, alpha_h=alpha_h, alpha_d=alpha_d)
    )
    _check_action_space_matches(policies, env)
    episode_results: list[EpisodeMetrics] = []

    # Clone once upfront so the original input policies are never modified or reseeded
    cloned_policies = {
        agent: _clone_policy(
            policies[agent],
            rows=game_config.rows,
            columns=game_config.columns,
            seed=seed * 2 + index,
        )
        for index, agent in enumerate(("knight", "boss"))
    }

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        for index, agent in enumerate(("knight", "boss")):
            agent_policy = cloned_policies[agent]
            if hasattr(agent_policy, "reset"):
                agent_policy.reset(seed=episode_seed * 2 + index)

        observations, _ = env.reset(seed=episode_seed)
        final_terminations: dict[str, bool] = {}
        final_truncations: dict[str, bool] = {}
        final_infos: dict[str, dict] = {}
        while env.agents:
            actions = {
                agent: _select_action(
                    cloned_policies[agent],
                    observations[agent],
                    deterministic=deterministic,
                )
                for agent in env.agents
            }
            (
                observations,
                _,
                final_terminations,
                final_truncations,
                final_infos,
            ) = env.step(actions)
        knight_info = final_infos["knight"]
        boss_info = final_infos["boss"]
        proposal_events = knight_info.get("proposal_events", {})
        episode_results.append(
            EpisodeMetrics(
                episode_index=episode_index,
                evaluation_seed=episode_seed,
                outcome=knight_info["outcome"],
                timeout=final_truncations["knight"],
                episode_length=knight_info["step_count"],
                knight_damage_dealt=knight_info["damage_dealt"],
                knight_damage_received=knight_info["damage_received"],
                boss_damage_dealt=boss_info["damage_dealt"],
                boss_damage_received=boss_info["damage_received"],
                invalid_knight_moves=proposal_events.get(
                    "knight_blocked_movement", 0
                ),
                special2_casts=proposal_events.get("boss_special2_cast", 0),
                special2_hits=proposal_events.get("boss_special2_hit", 0),
                special2_blocks=proposal_events.get("boss_special2_blocked", 0),
                special2_escapes=proposal_events.get("knight_escaped_special2", 0),
                knight_heal_noops=proposal_events.get("knight_heal_noop", 0),
                knight_heals=proposal_events.get("knight_heal_used", 0),
                knight_attack_attempts=(
                    proposal_events.get("knight_attack_attempt", 0)
                    + proposal_events.get("knight_attack_noop", 0)
                ),
                knight_attack_hits=proposal_events.get("knight_attack_hit", 0),
            )
        )
    env.close()
    return episode_results, summarize_episodes(
        episode_results, confidence=confidence
    )


def _check_action_space_matches(
    policies: Mapping[str, object], env: object
) -> None:
    """Fail fast when a policy and the evaluation environment disagree.

    Proposal and tactical policies expose more actions than the base
    environment. Evaluating them without a matching ``env_factory`` would
    silently score a different game (or crash mid-episode) and invalidate the
    results, so this is checked up front.
    """
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return
    for agent in ("knight", "boss"):
        policy = policies[agent]
        if not isinstance(policy, PPOAgent):
            continue
        expected = action_space(agent).n
        if policy.n_actions != expected:
            raise ValueError(
                f"Policy for {agent} has {policy.n_actions} actions but the "
                f"evaluation environment exposes {expected}; pass an "
                f"env_factory matching the training environment."
            )


def _clone_policy(policy: object, *, rows: int, columns: int, seed: int) -> object:
    if isinstance(policy, PPOAgent):
        clone = PPOAgent(
            policy.observation_size,
            policy.config,
            seed=seed,
            n_actions=policy.n_actions,
            respect_action_masks=policy.respect_action_masks,
        )
        clone.policy.load_state_dict(policy.policy.state_dict())
        return clone
    if hasattr(policy, "clone_for_eval"):
        return policy.clone_for_eval(rows=rows, columns=columns, seed=seed)
    if isinstance(policy, RandomAgent):
        return RandomAgent(rows=rows, columns=columns, seed=seed)
    if isinstance(policy, HeuristicAgent):
        return HeuristicAgent(rows=rows, columns=columns, seed=seed)
    raise TypeError(
        "Evaluation supports PPOAgent, RandomAgent, and HeuristicAgent policies."
    )


def _select_action(
    policy: object, observation: Mapping[str, np.ndarray], *, deterministic: bool
) -> int:
    if isinstance(policy, PPOAgent):
        return int(policy.act(observation, deterministic=deterministic))
    return int(policy.act(observation))
