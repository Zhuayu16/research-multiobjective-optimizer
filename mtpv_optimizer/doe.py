from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def central_composite_design(
    names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    integer_indices: tuple[int, ...] = (),
    center_points: int = 6,
    extra_space_filling_points: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a face-centered CCD plus optional space-filling interior points."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if len(names) != len(lower) or len(names) != len(upper):
        raise ValueError("变量名与上下限数量不一致。")
    if np.any(upper <= lower):
        raise ValueError("每个变量的上限必须大于下限。")

    dimension = len(names)
    coded: list[list[float]] = [list(point) for point in product((-1.0, 1.0), repeat=dimension)]
    for axis in range(dimension):
        for level in (-1.0, 1.0):
            point = [0.0] * dimension
            point[axis] = level
            coded.append(point)
    coded.extend([[0.0] * dimension for _ in range(center_points)])

    coded_array = np.asarray(coded, dtype=float)
    center = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0
    values = center + coded_array * half_range
    if extra_space_filling_points > 0:
        rng = np.random.default_rng(seed)
        lhs = np.empty((extra_space_filling_points, dimension), dtype=float)
        for axis in range(dimension):
            strata = (np.arange(extra_space_filling_points) + rng.random(extra_space_filling_points)) / extra_space_filling_points
            lhs[:, axis] = strata[rng.permutation(extra_space_filling_points)]
        interior = lower + lhs * (upper - lower)
        values = np.vstack((values, interior))
    if integer_indices:
        values[:, integer_indices] = np.rint(values[:, integer_indices])
    frame = pd.DataFrame(values, columns=names).drop_duplicates().reset_index(drop=True)
    frame.insert(0, "算例编号", np.arange(1, len(frame) + 1))
    return frame
