"""Independent actor-critic PPO policy for one agent."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from knight_boss.game.types import Action
from knight_boss.marl.observations import action_count, observation_to_vector
from knight_boss.training.config import PPOConfig


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, n_actions: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(width, hidden_size), nn.Tanh()))
            width = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.policy_head = nn.Linear(width, n_actions)
        self.value_head = nn.Linear(width, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observations)
        return self.policy_head(encoded), self.value_head(encoded).squeeze(-1)


class PPOAgent:
    """One independently parameterized PPO actor, critic, and optimizer."""

    def __init__(
        self,
        observation_size: int,
        config: PPOConfig,
        *,
        seed: int,
        n_actions: int | None = None,
        respect_action_masks: bool = True,
    ) -> None:
        self.observation_size = observation_size
        self.n_actions = n_actions or action_count()
        self.respect_action_masks = bool(respect_action_masks)
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.policy = ActorCritic(
                observation_size,
                self.n_actions,
                config.hidden_sizes,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self._torch_rng = torch.Generator(device=self.device)
        self._torch_rng.manual_seed(seed)
        self._numpy_rng = np.random.default_rng(seed)
        self.update_count = 0

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal RNG state for reproducible evaluation rollouts."""
        if seed is not None:
            self._torch_rng.manual_seed(seed)
            self._numpy_rng = np.random.default_rng(seed)

    def act(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        deterministic: bool = False,
    ) -> Action:
        vector = observation_to_vector(observation)
        if vector.size != self.observation_size:
            raise ValueError(
                f"Expected observation size {self.observation_size}, got {vector.size}."
            )
        tensor = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.policy(tensor)
            logits = self._masked_logits(
                logits,
                observation.get("action_mask") if self.respect_action_masks else None,
            )
            if deterministic:
                action_tensor = torch.argmax(logits, dim=-1)
            else:
                probabilities = torch.softmax(logits, dim=-1)
                action_tensor = torch.multinomial(
                    probabilities, 1, generator=self._torch_rng
                ).squeeze(-1)
        action_id = int(action_tensor.item())
        if self.n_actions == len(Action):
            return Action(action_id)
        return action_id

    def act_with_info(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        vector = observation_to_vector(observation)
        if vector.size != self.observation_size:
            raise ValueError(
                f"Expected observation size {self.observation_size}, got {vector.size}."
            )
        tensor = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, value = self.policy(tensor)
            logits = self._masked_logits(
                logits,
                observation.get("action_mask") if self.respect_action_masks else None,
            )
            log_probabilities = torch.log_softmax(logits, dim=-1)
            if deterministic:
                action_tensor = torch.argmax(logits, dim=-1)
            else:
                probabilities = log_probabilities.exp()
                action_tensor = torch.multinomial(
                    probabilities, 1, generator=self._torch_rng
                ).squeeze(-1)
            log_probability = log_probabilities[action_tensor]
        return (
            int(action_tensor.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    def value(self, observation: Mapping[str, np.ndarray]) -> float:
        vector = observation_to_vector(observation)
        tensor = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, value = self.policy(tensor)
        return float(value.item())

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        old_log_probabilities: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        action_masks: np.ndarray | None = None,
    ) -> dict[str, float]:
        count = len(observations)
        if count == 0:
            raise ValueError("Cannot update PPO from an empty rollout.")
        if not (
            len(actions)
            == len(old_log_probabilities)
            == len(advantages)
            == len(returns)
            == count
        ):
            raise ValueError("PPO rollout arrays must have matching lengths.")
        if observations.shape[1] != self.observation_size:
            raise ValueError("PPO rollout observation size does not match the policy.")

        device = self.device
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device)
        old_log_tensor = torch.as_tensor(
            old_log_probabilities, dtype=torch.float32, device=device
        )
        advantage_tensor = torch.as_tensor(
            advantages, dtype=torch.float32, device=device
        )
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        action_mask_tensor = None
        if action_masks is not None:
            action_mask_tensor = torch.as_tensor(
                action_masks, dtype=torch.bool, device=device
            )
            if action_mask_tensor.shape != (count, self.n_actions):
                raise ValueError("PPO action masks must have shape (samples, n_actions).")
            if not torch.all(action_mask_tensor.any(dim=1)):
                raise ValueError("Every PPO sample must allow at least one action.")
            if not torch.all(
                action_mask_tensor.gather(1, action_tensor.unsqueeze(1))
            ):
                raise ValueError("A rollout contains an action disallowed by its mask.")
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8
        )
        usable_count = (
            count - (count % self.config.batch_size)
            if count >= self.config.batch_size
            else count
        )
        effective_epochs = max(
            1,
            round(self.config.epochs * min(1.0, count / self.config.rollout_steps)),
        )
        metrics: list[tuple[float, float, float, float]] = []

        for _ in range(effective_epochs):
            order = self._numpy_rng.permutation(count)
            for start in range(0, usable_count, self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                logits, values = self.policy(obs_tensor[indices])
                if action_mask_tensor is not None:
                    logits = self._masked_logits(logits, action_mask_tensor[indices])
                distribution = Categorical(logits=logits)
                new_log_probabilities = distribution.log_prob(action_tensor[indices])
                ratios = torch.exp(new_log_probabilities - old_log_tensor[indices])
                unclipped = ratios * advantage_tensor[indices]
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantage_tensor[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(
                    values, returns_tensor[indices]
                )
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                metrics.append(
                    (
                        float(policy_loss.detach().cpu()),
                        float(value_loss.detach().cpu()),
                        float(entropy.detach().cpu()),
                        float(loss.detach().cpu()),
                    )
                )

        self.update_count += 1
        means = np.mean(np.asarray(metrics, dtype=np.float64), axis=0)
        return {
            "policy_loss": float(means[0]),
            "value_loss": float(means[1]),
            "entropy": float(means[2]),
            "total_loss": float(means[3]),
        }

    def _masked_logits(
        self, logits: torch.Tensor, action_mask: np.ndarray | torch.Tensor | None
    ) -> torch.Tensor:
        if action_mask is None:
            return logits
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
        if mask.shape != logits.shape:
            raise ValueError(
                f"Action mask shape {tuple(mask.shape)} does not match logits "
                f"{tuple(logits.shape)}."
            )
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("Every state must allow at least one action.")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
