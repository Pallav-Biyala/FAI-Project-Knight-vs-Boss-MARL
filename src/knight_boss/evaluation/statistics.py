"""Across-training-seed estimates; seeds remain the independent units."""

from collections.abc import Iterable
from math import sqrt
from statistics import NormalDist, mean, stdev
from typing import Any

_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean_ci_across_seeds(values: Iterable[float]) -> dict[str, Any]:
    """Mean, sample standard deviation, and t-based CI across training seeds."""
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("At least one independent training seed is required.")
    average = mean(samples)
    if len(samples) == 1:
        return {
            "n_training_seeds": 1,
            "mean": average,
            "standard_deviation": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    sample_std = stdev(samples)
    degrees_of_freedom = len(samples) - 1
    critical = _T_975.get(
        degrees_of_freedom,
        NormalDist().inv_cdf(0.975),
    )
    margin = critical * sample_std / sqrt(len(samples))
    return {
        "n_training_seeds": len(samples),
        "mean": average,
        "standard_deviation": sample_std,
        "ci_lower": average - margin,
        "ci_upper": average + margin,
    }
