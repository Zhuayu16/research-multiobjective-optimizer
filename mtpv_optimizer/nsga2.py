from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import ModelBundle, decision_scores, normalized_weights, objective_signs


@dataclass
class NSGA2Result:
    variables: np.ndarray
    objectives: np.ndarray
    utility: np.ndarray
    recommended_index: int
    run_summary: list[dict] | None = None
    history: list[dict] = field(default_factory=list)
    evaluation_audit: dict = field(default_factory=dict)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(fitness: np.ndarray, violations=None) -> tuple[list[np.ndarray], np.ndarray]:
    count = len(fitness)
    matrix = np.all(fitness[:, None, :] <= fitness[None, :, :], axis=2) & np.any(fitness[:, None, :] < fitness[None, :, :], axis=2)
    if violations is not None:
        v = np.asarray(violations, float)
        feasible = v <= 1e-10
        matrix &= feasible[:, None] & feasible[None, :]
        matrix |= feasible[:, None] & ~feasible[None, :]
        matrix |= ~feasible[:, None] & ~feasible[None, :] & (v[:, None] < v[None, :])
    dominated = [np.flatnonzero(row).tolist() for row in matrix]
    domination_count = matrix.sum(axis=0)
    rank = np.full(count, -1, dtype=int)
    first: list[int] = []
    for p in range(count):
        if domination_count[p] == 0:
            rank[p] = 0
            first.append(p)

    fronts: list[np.ndarray] = []
    current = first
    level = 0
    while current:
        fronts.append(np.asarray(current, dtype=int))
        following: list[int] = []
        for p in current:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = level + 1
                    following.append(q)
        level += 1
        current = following
    return fronts, rank


def _crowding(fitness: np.ndarray, front: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(front), dtype=float)
    if len(front) <= 2:
        distance[:] = np.inf
        return distance
    local = fitness[front]
    for objective in range(local.shape[1]):
        order = np.argsort(local[:, objective])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = local[order[-1], objective] - local[order[0], objective]
        if span > 0:
            distance[order[1:-1]] += (
                local[order[2:], objective] - local[order[:-2], objective]
            ) / span
    return distance


def _rank_and_crowding(fitness: np.ndarray, violations=None) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    fronts, rank = _non_dominated_sort(fitness, violations)
    crowding = np.zeros(len(fitness), dtype=float)
    for front in fronts:
        crowding[front] = _crowding(fitness, front)
    return rank, crowding, fronts


def _tournament(rng: np.random.Generator, rank: np.ndarray, crowding: np.ndarray) -> int:
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] < rank[b]:
        return int(a)
    if rank[b] < rank[a]:
        return int(b)
    if crowding[a] > crowding[b]:
        return int(a)
    if crowding[b] > crowding[a]:
        return int(b)
    return int(a if rng.random() < 0.5 else b)


def _repair(x: np.ndarray, lower: np.ndarray, upper: np.ndarray, integer_indices: tuple[int, ...]) -> np.ndarray:
    x = np.clip(x, lower, upper)
    if integer_indices:
        indices = list(integer_indices)
        x[indices] = np.clip(np.rint(x[indices]), np.ceil(lower[indices]), np.floor(upper[indices]))
    return x


def _sbx(
    rng: np.random.Generator,
    first: np.ndarray,
    second: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    probability: float = 0.9,
    eta: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = first.copy(), second.copy()
    if rng.random() > probability:
        return c1, c2
    for j in range(len(first)):
        if rng.random() > 0.5 or abs(first[j] - second[j]) < 1e-14:
            continue
        x1, x2 = sorted((first[j], second[j]))
        rand = rng.random()
        beta = 1.0 + 2.0 * (x1 - lower[j]) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
        beta = 1.0 + 2.0 * (upper[j] - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        if rng.random() < 0.5:
            child1, child2 = child2, child1
        c1[j], c2[j] = child1, child2
    return np.clip(c1, lower, upper), np.clip(c2, lower, upper)


def _mutate(
    rng: np.random.Generator,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eta: float = 20.0,
) -> np.ndarray:
    result = x.copy()
    probability = 1.0 / len(result)
    for j in range(len(result)):
        if rng.random() > probability:
            continue
        span = upper[j] - lower[j]
        if span <= 0:
            continue
        delta1 = (result[j] - lower[j]) / span
        delta2 = (upper[j] - result[j]) / span
        rand = rng.random()
        mutation_power = 1.0 / (eta + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            value = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
            deltaq = value ** mutation_power - 1.0
        else:
            xy = 1.0 - delta2
            value = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
            deltaq = 1.0 - value ** mutation_power
        result[j] += deltaq * span
    return np.clip(result, lower, upper)


def _fitness(bundle: ModelBundle, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = bundle.predict(population)
    minimized = predicted * objective_signs(getattr(bundle, "directions", None), predicted.shape[1])
    invalid = np.any(~np.isfinite(predicted), axis=1)
    if getattr(bundle, "require_positive", True):
        invalid |= np.any(predicted <= 0.0, axis=1)
    minimized[invalid] = 1e30
    return predicted, minimized


def _violations(bundle, population, predicted):
    evaluator = getattr(bundle, "constraint_violation", None)
    violation = np.zeros(len(population)) if evaluator is None else evaluator(population, predicted)
    violation = np.asarray(violation, float).copy()
    invalid = np.any(~np.isfinite(predicted), axis=1)
    if getattr(bundle, "require_positive", True):
        invalid |= np.any(predicted <= 0, axis=1)
    violation[invalid | ~np.isfinite(violation)] = np.inf
    return violation


def _bundle_repair(bundle, row):
    row = _repair(row, bundle.lower, bundle.upper, bundle.integer_indices)
    for j, choices in getattr(bundle, "allowed_values", {}).items():
        row[j] = choices[np.argmin(np.abs(np.asarray(choices) - row[j]))]
    return row


def _run_once(
    bundle: ModelBundle,
    population_size: int = 150,
    generations: int = 100,
    seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
    trace=None,
    audit=None,
) -> tuple[np.ndarray, np.ndarray]:
    if population_size < 20:
        raise ValueError("种群规模至少为 20。")
    if generations < 10:
        raise ValueError("迭代次数至少为 10。")
    rng = np.random.default_rng(seed)
    lower, upper = bundle.lower, bundle.upper
    domain = getattr(bundle, "domain", None)
    population = domain.sample(rng, population_size) if domain is not None else rng.uniform(lower, upper, size=(population_size, len(lower)))
    for index, row in enumerate(population):
        population[index] = _bundle_repair(bundle, row)
    # Seed with measured designs so narrow feasible regions are not lost.
    if getattr(bundle, "seed_measured", False) and bundle.training_x is not None:
        seeds = bundle.training_x
        seeds = seeds[np.all((seeds >= lower) & (seeds <= upper), axis=1)]
        if domain is not None:
            seeds = seeds[domain.contains(seeds)]
        if len(seeds):
            chosen = rng.permutation(len(seeds))[:population_size]
            population[:len(chosen)] = seeds[chosen]
    if domain is not None:
        valid = domain.contains(population)
        if not valid.any():
            raise ValueError("离散取值与样本凸包没有找到共同可行的初始设计。")
        if not valid.all():
            population[~valid] = population[rng.choice(np.flatnonzero(valid), size=int((~valid).sum()))]

    def evaluate(values):
        if audit is not None:
            audit["objective_vector_rows"] += len(values)
            audit["objective_prediction_batches"] += 1
        return _fitness(bundle, values)

    for generation in range(generations):
        predicted, fitness = evaluate(population)
        rank, crowding, _ = _rank_and_crowding(fitness, _violations(bundle, population, predicted))
        children: list[np.ndarray] = []
        while len(children) < population_size:
            p1 = population[_tournament(rng, rank, crowding)]
            p2 = population[_tournament(rng, rank, crowding)]
            c1, c2 = _sbx(rng, p1, p2, lower, upper)
            c1 = _bundle_repair(bundle, _mutate(rng, c1, lower, upper))
            children.append(domain.repair(c1, p1) if domain is not None else c1)
            if len(children) < population_size:
                c2 = _bundle_repair(bundle, _mutate(rng, c2, lower, upper))
                children.append(domain.repair(c2, p2) if domain is not None else c2)

        combined = np.vstack((population, np.asarray(children)))
        combined_predicted, combined_fitness = evaluate(combined)
        combined_violation = _violations(bundle, combined, combined_predicted)
        _, _, fronts = _rank_and_crowding(combined_fitness, combined_violation)
        selected: list[int] = []
        for front in fronts:
            remaining = population_size - len(selected)
            if len(front) <= remaining:
                selected.extend(front.tolist())
            else:
                distance = _crowding(combined_fitness, front)
                order = np.argsort(-distance, kind="stable")
                selected.extend(front[order[:remaining]].tolist())
                break
        population = combined[np.asarray(selected, dtype=int)]
        if trace:
            trace(generation + 1, combined_predicted[selected], combined_violation[selected])
        if progress and (generation == 0 or (generation + 1) % 5 == 0 or generation + 1 == generations):
            progress(generation + 1, generations)

    predicted, final_fitness = evaluate(population)
    violations = _violations(bundle, population, predicted)
    fronts, _ = _non_dominated_sort(final_fitness, violations)
    pareto_indices = fronts[0]
    pareto_x = population[pareto_indices]
    pareto_y = predicted[pareto_indices]
    physically_valid = violations[pareto_indices] <= 1e-10
    pareto_x = pareto_x[physically_valid]
    pareto_y = pareto_y[physically_valid]
    if len(pareto_x) == 0:
        raise ValueError("搜索没有找到满足数据域及工程约束的方案；请检查约束、边界或补充可行样本。")

    return pareto_x, pareto_y


def optimize_nsga2(
    bundle: ModelBundle,
    population_size: int = 150,
    generations: int = 100,
    seed: int = 42,
    weights: np.ndarray | None = None,
    runs: int = 3,
    decision_method: str = "TOPSIS",
    progress: Callable[[int, int], None] | None = None,
    anchors: tuple[np.ndarray, np.ndarray] | None = None,
) -> NSGA2Result:
    if runs < 1 or runs > 20:
        raise ValueError("独立运行次数应在 1–20 之间。")
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    directions = getattr(bundle, "directions", None)
    weights = normalized_weights(weights, len(bundle.target_names))
    summaries = []
    history = []
    evaluation_audit = dict(objective_vector_rows=0, objective_prediction_batches=0,
        population_size=population_size, generations=generations, runs=runs,
        semantics="Rows requested from the objective surrogate, including repeated evaluations; not CFD solves or unique designs.")
    total_steps = generations * runs
    for run_index in range(runs):
        callback = None
        if progress:
            callback = lambda current, _total, offset=run_index * generations: progress(offset + current, total_steps)
        run_seed = seed + 7919 * run_index
        def trace(generation, y, violations):
            feasible = violations <= 1e-10
            valid = y[feasible]
            if len(valid):
                front = _non_dominated_sort(valid * objective_signs(directions, valid.shape[1]))[0][0]
                reference = decision_scores(valid, weights, "IDEAL", anchors, directions) if anchors is not None else None
            else:
                front, reference = [], None
            history.append(dict(seed=run_seed, generation=generation, feasible_count=int(feasible.sum()),
                nondominated_count=len(front), reference_ideal_utility=float(reference.max()) if reference is not None else None))
        run_x, run_y = _run_once(
            bundle,
            population_size=population_size,
            generations=generations,
            seed=seed + 7919 * run_index,
            progress=callback,
            trace=trace,
            audit=evaluation_audit,
        )
        all_x.append(run_x)
        all_y.append(run_y)
        scores = decision_scores(run_y, weights, decision_method, anchors, directions)
        best = int(np.argmax(scores))
        summaries.append(dict(seed=seed + 7919 * run_index, pareto_count=len(run_y),
                              variables=run_x[best].tolist(), objectives=run_y[best].tolist(),
                              utility=float(scores[best]), maximum_power=float(run_y[:, min(1, run_y.shape[1] - 1)].max())))

    pareto_x = np.vstack(all_x)
    pareto_y = np.vstack(all_y)
    merged_fitness = pareto_y * objective_signs(directions, pareto_y.shape[1])
    fronts, _ = _non_dominated_sort(merged_fitness)
    pareto_x = pareto_x[fronts[0]]
    pareto_y = pareto_y[fronts[0]]

    combined_for_unique = np.column_stack((pareto_x, pareto_y))
    rounded = np.round(combined_for_unique, decimals=10)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    pareto_x = pareto_x[unique_indices]
    pareto_y = pareto_y[unique_indices]
    utility = decision_scores(pareto_y, weights, decision_method, anchors, directions)
    recommended = int(np.argmax(utility))
    order = np.argsort(-utility)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    recommended_sorted = int(inverse[recommended])
    return NSGA2Result(
        variables=pareto_x[order],
        objectives=pareto_y[order],
        utility=utility[order],
        recommended_index=recommended_sorted,
        run_summary=summaries,
        history=history,
        evaluation_audit=evaluation_audit,
    )
