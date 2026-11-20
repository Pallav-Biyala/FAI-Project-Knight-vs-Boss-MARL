"""Per-agent rollout storage and generalized advantage estimation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    old_log_probabilities: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    action_masks: np.ndarray | None = None


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.old_log_probabilities: list[float] = []
        self.action_masks: list[np.ndarray | None] = []

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        old_log_probability: float,
        action_mask: np.ndarray | None = None,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.old_log_probabilities.append(float(old_log_probability))
        self.action_masks.append(
            None
            if action_mask is None
            else np.asarray(action_mask, dtype=np.bool_).copy()
        )

    def compute(
        self, *, last_value: float, gamma: float, gae_lambda: float
    ) -> RolloutBatch:
        if not self.rewards:
            raise ValueError("Cannot compute advantages for an empty rollout.")
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.bool_)
        values = np.asarray(self.values, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            next_value = last_value if index == len(rewards) - 1 else values[index + 1]
            nonterminal = 0.0 if dones[index] else 1.0
            delta = rewards[index] + gamma * next_value * nonterminal - values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[index] = gae
        action_masks = None
        if any(mask is not None for mask in self.action_masks):
            shapes = {
                mask.shape for mask in self.action_masks if mask is not None
            }
            if len(shapes) != 1 or any(mask is None for mask in self.action_masks):
                raise ValueError("A rollout must use consistently shaped action masks.")
            action_masks = np.stack(self.action_masks).astype(np.bool_)
        return RolloutBatch(
            observations=np.stack(self.observations).astype(np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=rewards,
            dones=dones,
            values=values,
            old_log_probabilities=np.asarray(
                self.old_log_probabilities, dtype=np.float32
            ),
            advantages=advantages,
            returns=advantages + values,
            action_masks=action_masks,
        )
