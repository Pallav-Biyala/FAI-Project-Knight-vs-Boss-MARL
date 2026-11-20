"""Stable vector encoding for PPO policies."""

from collections.abc import Mapping

import numpy as np

from knight_boss.game.types import Action

OBSERVATION_FIELDS = (
    "positions",
    "health",
    "health_fraction",
    "cooldowns",
    "combat_stats",
    "windups",
    "remaining_steps",
)


# Fixed per-field scales that bring raw quantities to roughly [0, 3]. The raw
# observation carries absolute health (up to 3 x 150), damage values (up to
# 100) and the remaining step count (up to 300); feeding those unscaled into a
# tanh MLP saturates its units. Scaling is applied only when building the
# network input; the environment's observation dict is unchanged, so scripted
# agents and the existing tests are unaffected. Checkpoints trained before
# this scaling was introduced are not compatible with it.
OBSERVATION_SCALES: dict[str, float] = {
    "health": 150.0,
    "combat_stats": 50.0,
    "remaining_steps": 300.0,
}


def observation_to_vector(observation: Mapping[str, np.ndarray]) -> np.ndarray:
    """Flatten the structured environment observation in a fixed field order."""
    missing = set(OBSERVATION_FIELDS) - set(observation)
    if missing:
        raise ValueError(f"Observation is missing fields: {sorted(missing)}")
    chunks = [
        np.asarray(observation[field], dtype=np.float32).reshape(-1)
        / np.float32(OBSERVATION_SCALES.get(field, 1.0))
        for field in OBSERVATION_FIELDS
    ]
    # Action masks control the policy distribution, not the actor/critic input.
    # Keeping them out preserves the observation architecture; PPO stores each
    # mask alongside its transition and reuses it when recomputing log-probs.
    for extra_field in sorted(
        set(observation) - set(OBSERVATION_FIELDS) - {"action_mask"}
    ):
        chunks.append(np.asarray(observation[extra_field], dtype=np.float32).reshape(-1))
    vector = np.concatenate(chunks)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Observation contains non-finite values.")
    return vector


def action_count() -> int:
    return len(Action)
