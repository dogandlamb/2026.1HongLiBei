from __future__ import annotations

import numpy as np


def rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_equivalent_diameter_um(
    rng: np.random.Generator,
    d_min_um: float,
    d_max_um: float,
    mode: str = "uniform",
) -> float:
    if mode == "uniform":
        return float(rng.uniform(d_min_um, d_max_um))
    if mode == "triangular":
        return float(rng.triangular(d_min_um, (d_min_um + d_max_um) / 2.0, d_max_um))
    raise ValueError(f"Unknown diameter mode: {mode}")


def random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    # Shoemake method
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([q4, q1, q2, q3], dtype=float)  # (w, x, y, z)
