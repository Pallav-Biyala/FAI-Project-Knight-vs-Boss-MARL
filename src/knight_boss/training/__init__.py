"""PPO training and checkpointing."""

from .config import PPOConfig

__all__ = ["MultiAgentPPOTrainer", "PPOConfig"]


def __getattr__(name: str):
    if name == "MultiAgentPPOTrainer":
        from .trainer import MultiAgentPPOTrainer

        return MultiAgentPPOTrainer
    raise AttributeError(name)
