from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.spatial import ConvexHull, QhullError


class SampleDomain:
    """Scaled sample hull intersected with the requested variable bounds."""

    def __init__(self, samples, lower, upper, integer_indices=()):
        self.samples = np.asarray(samples, dtype=float)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.integer_indices = tuple(integer_indices)
        self.origin = self.samples.min(axis=0)
        self.span = self.samples.max(axis=0) - self.origin
        if np.any(self.span <= 0):
            raise ValueError("Convex hull requires varying inputs.")
        z = (self.samples - self.origin) / self.span
        if np.linalg.matrix_rank(z - z.mean(axis=0)) < z.shape[1]:
            raise ValueError("Input samples are rank deficient; remove dependent variables or add joint samples.")
        if z.shape[1] == 1:
            self.equations = np.array([[-1.0, 0.0], [1.0, -1.0]])
            return
        try:
            self.equations = ConvexHull(z).equations
        except QhullError as exc:
            raise ValueError("Cannot construct the input sample convex hull.") from exc

    def contains(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        z = (x - self.origin) / self.span
        hull = np.all(z @ self.equations[:, :-1].T + self.equations[:, -1] <= 1e-9, axis=1)
        bounds = np.all((x >= self.lower - 1e-12) & (x <= self.upper + 1e-12), axis=1)
        return hull & bounds & np.all(np.isfinite(x), axis=1)

    def sample(self, rng, count):
        accepted = []
        remaining = count
        for _ in range(100):
            x = rng.uniform(self.lower, self.upper, size=(max(256, remaining * 4), len(self.lower)))
            if self.integer_indices:
                x[:, self.integer_indices] = np.rint(x[:, self.integer_indices])
            valid = x[self.contains(x)][:remaining]
            accepted.extend(valid)
            remaining -= len(valid)
            if not remaining:
                return np.asarray(accepted)
        raise ValueError("Requested bounds have insufficient feasible hull volume; widen bounds or add samples.")

    def repair(self, child, parent):
        if self.contains(child)[0]:
            return child
        if self.integer_indices or getattr(self, "discrete_repair", False):
            return parent.copy()
        # Intersect the parent-child segment with each scaled hull half-space.
        z = (parent - self.origin) / self.span
        direction = (child - parent) / self.span
        denominator = self.equations[:, :-1] @ direction
        residual = self.equations[:, :-1] @ z + self.equations[:, -1]
        outward = denominator > 1e-14
        fraction = min(1.0, float(np.min(-residual[outward] / denominator[outward]))) if outward.any() else 1.0
        repaired = parent + max(0.0, fraction * (1.0 - 1e-10)) * (child - parent)
        return repaired if self.contains(repaired)[0] else parent.copy()

    def linear_constraint(self):
        a = self.equations[:, :-1] / self.span
        b = a @ self.origin - self.equations[:, -1] + 1e-10
        return LinearConstraint(a, -np.inf, b)
