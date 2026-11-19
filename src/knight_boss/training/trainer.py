"""Centralized simultaneous rollout collection and independent PPO updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from knight_boss.agents.ppo_agent import PPOAgent
from knight_boss.game.engine import AGENTS
from knight_boss.game.types import Action
from knight_boss.marl.observations import observation_to_vector
from knight_boss.marl.parallel_env import KnightBossParallelEnv
from knight_boss.training.config import PPOConfig
from knight_boss.training.rollout import RolloutBatch, RolloutBuffer


class MultiAgentPPOTrainer:
    """Collect joint actions first, then update each trainable policy separately."""

    def __init__(
        self,
        env: KnightBossParallelEnv,
        config: PPOConfig,
        *,
        seed: int,
        policies: Mapping[str, object] | None = None,
    ) -> None:
        self.env = env
        self.config = config
        self.seed = seed
        torch.manual_seed(seed)
        self.observations, _ = env.reset(seed=seed)
        if policies is None:
            observation_size = observation_to_vector(
                self.observations[AGENTS[0]]
            ).size
            self.policies: dict[str, object] = {
                agent: PPOAgent(
                    observation_size,
                    config,
                    seed=seed + index,
                    n_actions=env.action_space(agent).n,
                )
                for index, agent in enumerate(AGENTS)
            }
        else:
            if set(policies) != set(AGENTS):
                raise ValueError(f"Policies must be provided for exactly {AGENTS}.")
            self.policies = dict(policies)
        self.total_environment_steps = 0
        self.completed_episodes = 0
        self.last_rollout_rewards: dict[str, float] = {}
        self.last_rollout_episodes: list[dict[str, object]] = []

    def collect_rollout(
        self,
        steps: int | None = None,
        *,
        trainable_agents: Sequence[str] | None = None,
        stop_after_episodes: int | None = None,
        stop_at_episode_end: bool = False,
    ) -> dict[str, RolloutBatch]:
        rollout_steps = self.config.rollout_steps if steps is None else steps
        if rollout_steps < 1:
            raise ValueError("Rollout steps must be positive.")
        if (
            stop_after_episodes is not None
            and stop_after_episodes <= self.completed_episodes
        ):
            raise ValueError("stop_after_episodes must be greater than completed episodes.")
        trainable = self._trainable_agents(trainable_agents)
        buffers = {agent: RolloutBuffer() for agent in trainable}
        completed_in_rollout: list[dict[str, object]] = []

        steps_collected = 0
        episode_ended = False

        while steps_collected < rollout_steps or (
            stop_at_episode_end and not episode_ended
        ):
            episode_ended = False
            current_observations = self.observations
            pending_records: dict[str, tuple[np.ndarray, int, float, float]] = {}
            actions: dict[str, int] = {}
            for agent in AGENTS:
                policy = self.policies[agent]
                observation = current_observations[agent]
                if isinstance(policy, PPOAgent):
                    action, log_probability, value = policy.act_with_info(observation)
                elif hasattr(policy, "act"):
                    action = int(policy.act(observation))
                    log_probability, value = 0.0, 0.0
                else:
                    raise TypeError(f"Policy for {agent} must implement act().")
                actions[agent] = int(action)
                if agent in buffers:
                    if not isinstance(policy, PPOAgent):
                        raise TypeError(
                            f"Trainable policy for {agent} must be a PPOAgent."
                        )
                    pending_records[agent] = (
                        observation_to_vector(observation),
                        int(action),
                        log_probability,
                        value,
                    )

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(actions)
            for agent, (vector, action, log_probability, value) in pending_records.items():
                done = terminations[agent] or truncations[agent]
                buffers[agent].add(
                    vector,
                    action,
                    rewards[agent],
                    done,
                    value,
                    log_probability,
                )

            self.total_environment_steps += 1
            steps_collected += 1
            if self.env.agents:
                self.observations = next_observations
            else:
                knight_info = infos["knight"]
                completed_in_rollout.append(
                    {
                        "outcome": knight_info["outcome"],
                        "timeout": truncations["knight"],
                        "episode_length": knight_info["step_count"],
                        "knight_damage_dealt": knight_info["damage_dealt"],
                        "knight_damage_received": knight_info["damage_received"],
                        "boss_damage_dealt": infos["boss"]["damage_dealt"],
                        "boss_damage_received": infos["boss"]["damage_received"],
                    }
                )
                self.completed_episodes += 1
                self.observations, _ = self.env.reset(
                    seed=self.seed + self.completed_episodes
                )
                episode_ended = True
                if (
                    stop_after_episodes is not None
                    and self.completed_episodes >= stop_after_episodes
                ):
                    break

        batches: dict[str, RolloutBatch] = {}
        for agent, buffer in buffers.items():
            last_value = (
                self.policies[agent].value(self.observations[agent])
                if self.env.agents
                else 0.0
            )
            batches[agent] = buffer.compute(
                last_value=last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
        self.last_rollout_rewards = {
            agent: float(batch.rewards.sum()) for agent, batch in batches.items()
        }
        self.last_rollout_episodes = completed_in_rollout
        return batches

    def train_rollout(
        self,
        steps: int | None = None,
        *,
        trainable_agents: Sequence[str] | None = None,
        stop_after_episodes: int | None = None,
        stop_at_episode_end: bool = False,
    ) -> dict[str, dict[str, float]]:
        batches = self.collect_rollout(
            steps,
            trainable_agents=trainable_agents,
            stop_after_episodes=stop_after_episodes,
            stop_at_episode_end=stop_at_episode_end,
        )
        metrics: dict[str, dict[str, float]] = {}
        for agent, batch in batches.items():
            policy = self.policies[agent]
            if not isinstance(policy, PPOAgent):
                raise TypeError(f"Trainable policy for {agent} must be a PPOAgent.")
            metrics[agent] = policy.update(
                observations=batch.observations,
                actions=batch.actions,
                old_log_probabilities=batch.old_log_probabilities,
                advantages=batch.advantages,
                returns=batch.returns,
            )
        return metrics

    def _trainable_agents(
        self, trainable_agents: Sequence[str] | None
    ) -> tuple[str, ...]:
        selected = tuple(AGENTS if trainable_agents is None else trainable_agents)
        if len(set(selected)) != len(selected) or not set(selected).issubset(AGENTS):
            raise ValueError("trainable_agents must contain unique known agents.")
        return selected
