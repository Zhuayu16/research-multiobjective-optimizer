from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from threadpoolctl import threadpool_limits

from .core import decision_scores, fit_surrogates, ideal_distances, metrics_frame, normalized_weights, validate_dataset
from .coupled import FuelDefinition, OBJECTIVES, RESPONSES, fit_coupled, objective_values
from .nsga2 import _non_dominated_sort, optimize_nsga2


@dataclass(frozen=True)
class RunConfig:
    variables: list[str]
    targets: list[str]
    lower: list[float]
    upper: list[float]
    integer_indices: tuple[int, ...] = ()
    mode: str = "generic"
    phi_column: str = "phi"
    flow_column: str = "qm_1e5_kg_s"
    pv_column: str = "P_MTPV_W"
    te_column: str = "P_MTEG_W"
    fraction_column: str | None = None
    flow_scale: float = 1e-5
    lhv_J_kg: float = 119960000.0
    stoichiometric_air_fuel_ratio: float = 34.32
    use_hull: bool = True
    model_mode: str = "Auto"
    seed: int = 20260929
    population_size: int = 90
    generations: int = 65
    runs: int = 2
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    decision_method: str = "IDEAL"
    anchor_minimum: list[float] | None = None
    anchor_maximum: list[float] | None = None
    check_endpoints: bool = True
    endpoint_iterations: int = 90
    holdout_batch: str | None = None
    sources: list[str] = field(default_factory=list)
    fixed_parameters: dict = field(default_factory=dict)


@dataclass
class RunOutput:
    config: dict
    bundle: object
    clean: pd.DataFrame
    predicted: pd.DataFrame
    observed: pd.DataFrame
    sensitivity: pd.DataFrame
    run_summary: pd.DataFrame
    endpoints: pd.DataFrame
    warnings: list[str]


def _front(values):
    cost = np.asarray(values).copy()
    cost[:, 1:] *= -1
    return _non_dominated_sort(cost)[0][0]


def _rank(frame, targets, weights, method, anchors):
    frame = frame.copy()
    values = frame[targets].to_numpy(float)
    frame["综合效用"] = decision_scores(values, weights, method, anchors)
    if method == "IDEAL":
        frame["理想点距离"] = ideal_distances(values, weights, anchors)
    frame = frame.sort_values("综合效用", ascending=False, kind="stable").reset_index(drop=True)
    frame.insert(0, "推荐排序", np.arange(1, len(frame) + 1))
    return frame


def _endpoint_checks(bundle, seed, iterations, anchors, weights, progress=None):
    domain = bundle.domain
    constraints = (domain.linear_constraint(),) if domain is not None else ()
    bounds = list(zip(bundle.lower, bundle.upper))
    rows = []
    for j, name in enumerate(("minimum_pressure", "maximum_power", "maximum_efficiency", "balanced_ideal")):
        if progress:
            progress(85 + 3 * j, name)
        def score(x):
            y = bundle.predict(x)[0]
            if not np.all(np.isfinite(y)) or np.any(y <= 0):
                return 1e30
            if j == 3:
                return float(ideal_distances([y], weights, anchors)[0])
            return float(y[j] * (1 if j == 0 else -1))
        def callback(x, convergence):
            if progress:
                progress(85 + 3 * j, name)
            return False
        integer_mask = np.zeros(len(bounds), bool)
        integer_mask[list(bundle.integer_indices)] = True
        solved = differential_evolution(score, bounds, constraints=constraints, seed=seed + (17 if j == 3 else j),
                                         popsize=16 if j == 3 else 12, maxiter=round(iterations * 130 / 90) if j == 3 else iterations,
                                         tol=1e-7 if j == 3 else 1e-6, polish=False,
                                         integrality=integer_mask, callback=callback)
        y = bundle.predict(solved.x)[0]
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError(f"Endpoint search found no feasible positive candidate: {name}")
        row = dict(zip(bundle.variable_names + bundle.target_names, np.r_[solved.x, y]))
        row.update(endpoint=name, seed=seed + (17 if j == 3 else j), converged=bool(solved.success),
                   iterations=int(solved.nit), evaluations=int(solved.nfev), status=str(solved.message))
        rows.append(row)
    return pd.DataFrame(rows)


def run_workflow(frame, config, progress=None):
    if config.mode not in {"generic", "coupled"}:
        raise ValueError("Unknown workflow mode.")
    weights = normalized_weights(config.weights)
    if config.decision_method not in {"IDEAL", "TOPSIS", "ARAS"}:
        raise ValueError("Unknown decision method.")
    work = frame.copy()
    if config.mode == "coupled":
        if config.phi_column not in config.variables or config.flow_column not in config.variables:
            raise ValueError("Coupled H2/air mode requires equivalence ratio and mixture mass flow as variables.")
        mapped = [config.targets[0], config.pv_column, config.te_column]
        if len(set(mapped + config.variables)) != len(mapped + config.variables):
            raise ValueError("Response and variable mappings must not overlap.")
        for canonical, source in zip(RESPONSES, mapped):
            work[canonical] = pd.to_numeric(frame[source], errors="raise")
        if config.fraction_column:
            work["Y_H2_in"] = pd.to_numeric(frame[config.fraction_column], errors="raise")
        responses, targets = RESPONSES, OBJECTIVES
    else:
        responses = targets = config.targets
    numeric = config.variables + responses
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    valid = np.all(np.isfinite(work[numeric].to_numpy(float)), axis=1)
    removed = int(np.sum(~valid))
    work = work.loc[valid].drop_duplicates(numeric).reset_index(drop=True)
    _, notes = validate_dataset(work, config.variables, responses, config.integer_indices)
    if removed:
        notes.append(f"已剔除 {removed} 行空值、非数值或无限值。")
    if np.any(work[responses].to_numpy(float) <= 0):
        raise ValueError("All three responses must be positive.")
    if config.integer_indices:
        for j in config.integer_indices:
            if np.ceil(config.lower[j]) > np.floor(config.upper[j]):
                raise ValueError("Integer bounds contain no admissible integer.")
    if progress:
        progress(2, "代理模型交叉验证")
    with threadpool_limits(limits=1):
        if config.mode == "coupled":
            fuel = FuelDefinition(config.variables.index(config.phi_column), config.variables.index(config.flow_column),
                                  config.flow_scale, config.lhv_J_kg, config.stoichiometric_air_fuel_ratio)
            mask = None
            if config.holdout_batch:
                if "batch" not in work:
                    raise ValueError("Response holdout requires a batch column.")
                mask = work["batch"].eq(config.holdout_batch).to_numpy()
                if not mask.any():
                    raise ValueError("Requested holdout batch is absent.")
            bundle = fit_coupled(work, config.variables, fuel, config.lower, config.upper,
                                 config.integer_indices, config.seed, config.model_mode, config.use_hull, mask)
            actual = objective_values(work[RESPONSES].to_numpy(float), work[config.variables].to_numpy(float), fuel)
            for j, column in enumerate(targets):
                if column in work:
                    work[column + "_input"] = work[column]
                work[column] = actual[:, j]
        else:
            bundle = fit_surrogates(work, config.variables, targets, np.array(config.lower), np.array(config.upper),
                                    config.integer_indices, config.seed, config.model_mode, config.use_hull)
            actual = work[targets].to_numpy(float)
        if (config.anchor_minimum is None) != (config.anchor_maximum is None):
            raise ValueError("Both anchor minima and maxima must be supplied together.")
        anchors = (actual.min(axis=0), actual.max(axis=0)) if config.anchor_minimum is None else (
            np.array(config.anchor_minimum), np.array(config.anchor_maximum))
        ideal_distances(actual, weights, anchors)
        if progress:
            progress(25, "多次 NSGA-II 搜索")
        result = optimize_nsga2(bundle, config.population_size, config.generations, config.seed, weights,
                                config.runs, config.decision_method,
                                progress=(lambda current, total: progress(25 + round(60 * current / total), "多次 NSGA-II 搜索")) if progress else None,
                                anchors=anchors)
        endpoints = _endpoint_checks(bundle, config.seed, config.endpoint_iterations, anchors, weights, progress) if config.check_endpoints else pd.DataFrame()
    predicted = pd.DataFrame(np.column_stack([result.variables, result.objectives]), columns=config.variables + targets)
    if not endpoints.empty:
        predicted = pd.concat([predicted, endpoints[config.variables + targets]], ignore_index=True)
    predicted = predicted.drop_duplicates(config.variables).reset_index(drop=True)
    predicted = predicted.iloc[_front(predicted[targets].to_numpy(float))]
    predicted = _rank(predicted, targets, weights, config.decision_method, anchors)
    predicted["结果类型"] = "代理模型预测"
    if config.mode == "coupled":
        components = bundle.predict_components(predicted[config.variables].to_numpy(float))
        predicted["P_MTPV_W"], predicted["P_MTEG_W"] = components[:, 1], components[:, 2]
    within = np.all((work[config.variables].to_numpy(float) >= bundle.lower) &
                    (work[config.variables].to_numpy(float) <= bundle.upper), axis=1)
    observed = work.loc[within].copy()
    if observed.empty:
        raise ValueError("Requested bounds contain no calculated design to rank.")
    observed = observed.iloc[_front(observed[targets].to_numpy(float))]
    observed = _rank(observed, targets, weights, config.decision_method, anchors)
    observed["结果类型"] = "输入表计算值"
    scenarios = [("等权", (1, 1, 1)), ("压降优先", (2, 1, 1)), ("功率优先", (1, 2, 1)), ("效率优先", (1, 1, 2)), ("当前权重", config.weights)]
    sensitivity = []
    for category, choices in (("输入表计算值", observed), ("代理模型预测", predicted)):
        for name, w in scenarios:
            ranked = _rank(choices.drop(columns="推荐排序"), targets, w, "IDEAL", anchors)
            chosen = ranked.iloc[0]
            row = {"结果类型": category, "偏好": name, "压降权重": normalized_weights(w)[0],
                   "功率权重": normalized_weights(w)[1], "效率权重": normalized_weights(w)[2]}
            for column in ["case"] + config.variables + targets + ["理想点距离"]:
                if column in chosen:
                    row[column] = chosen[column]
            sensitivity.append(row)
    summaries = []
    for run in result.run_summary:
        row = {k: run[k] for k in ("seed", "pareto_count", "utility", "maximum_power")}
        row.update(zip(config.variables + targets, run["variables"] + run["objectives"]))
        if config.decision_method == "IDEAL":
            row["理想点距离"] = float(ideal_distances([run["objectives"]], weights, anchors)[0])
        summaries.append(row)
    saved = asdict(config)
    saved.update(normalized_weights=weights.tolist(), anchor_minimum=anchors[0].tolist(), anchor_maximum=anchors[1].tolist(),
                 anchor_source="input_table_fixed_before_search" if config.anchor_minimum is None else "user_fixed",
                 sample_count=len(work), target_names=targets, selected_models={m.target: m.model_name for m in bundle.metrics},
                 cv="shuffled five-fold; model selection and reporting share folds", cv_seed=config.seed,
                 crossover_probability=0.9, crossover_eta=15, mutation_probability=1 / len(config.variables), mutation_eta=20)
    saved.update(endpoint_polish=False, endpoint_population_multiplier=12, endpoint_tol=1e-6,
                 balanced_population_multiplier=16, balanced_iterations=round(config.endpoint_iterations * 130 / 90), balanced_tol=1e-7)
    saved["clean_data_sha256"] = hashlib.sha256(work.to_json(orient="records", double_precision=15).encode("utf-8")).hexdigest()
    saved["source_sha256"] = {source: hashlib.sha256(Path(source).read_bytes()).hexdigest()
                              for source in config.sources if Path(source).is_file()}
    import scipy
    import sklearn
    saved["library_versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    if progress:
        progress(100, "完成")
    return RunOutput(saved, bundle, work, predicted, observed, pd.DataFrame(sensitivity), pd.DataFrame(summaries), endpoints, notes)


def export_workbook(output, filename):
    path = Path(filename)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in (("模型预测Pareto", output.predicted), ("已计算Pareto", output.observed),
                            ("固定尺度偏好对比", output.sensitivity), ("独立运行比较", output.run_summary),
                            ("端点搜索核对", output.endpoints), ("模型检验", metrics_frame(output.bundle)),
                            ("候选模型比较", output.bundle.candidate_metrics), ("交叉验证预测", output.bundle.validation_frame),
                            ("响应留出检验", getattr(output.bundle, "holdout_frame", pd.DataFrame())), ("源数据", output.clean)):
            table.to_excel(writer, index=False, sheet_name=name)
        for attribute, name in (("constraint_audit", "可行性与覆盖诊断"), ("cleaning_audit", "数据清理记录"), ("holdout_metrics", "独立留出指标")):
            table = getattr(output, attribute, None)
            if table is not None:
                table.to_excel(writer, index=False, sheet_name=name)
        pd.DataFrame([(key, json.dumps(value, ensure_ascii=False)) for key, value in output.config.items()],
                     columns=["设置", "值"]).to_excel(writer, index=False, sheet_name="运行设置")
    path.with_suffix(".json").write_text(json.dumps(output.config, ensure_ascii=False, indent=2), encoding="utf-8")
