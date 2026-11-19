"""PPO hyperparameter configuration."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 0.0003
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    epochs: int = 10
    batch_size: int = 64
    rollout_steps: int = 1024
    hidden_sizes: tuple[int, ...] = (128, 128)
    max_grad_norm: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "clip_range": self.clip_range,
            "max_grad_norm": self.max_grad_norm,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1].")
        if (
            not isfinite(self.entropy_coef)
            or not isfinite(self.value_coef)
            or self.entropy_coef < 0
            or self.value_coef < 0
        ):
            raise ValueError("Loss coefficients must be non-negative.")
        if self.epochs < 1 or self.batch_size < 1 or self.rollout_steps < 1:
            raise ValueError("epochs, batch_size, and rollout_steps must be positive.")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer widths.")
        if not self.device:
            raise ValueError("device must be specified.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PPOConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            data: Any = yaml.safe_load(config_file)
        if not isinstance(data, dict):
            raise ValueError("PPO configuration must be a YAML mapping.")
        return cls(
            learning_rate=float(data["learning_rate"]),
            gamma=float(data["gamma"]),
            gae_lambda=float(data["gae_lambda"]),
            clip_range=float(data["clip_range"]),
            entropy_coef=float(data["entropy_coef"]),
            value_coef=float(data["value_coef"]),
            epochs=int(data["epochs"]),
            batch_size=int(data["batch_size"]),
            rollout_steps=int(data["rollout_steps"]),
            hidden_sizes=tuple(int(size) for size in data["hidden_sizes"]),
            max_grad_norm=float(data["max_grad_norm"]),
            device=str(data["device"]),
        )
