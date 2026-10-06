"""Domain-neutral modelling and optimization; H2/air physics stays in coupled.py."""
from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from threadpoolctl import threadpool_limits

from .core import decision_scores, fit_surrogates, ideal_distances, normalized_weights, objective_signs
from .nsga2 import _non_dominated_sort, optimize_nsga2
from .problem import ConstraintEvaluator, Problem
from .workflow import RunOutput
from .scientific import decision_matrix, correlations


def _clean(frame, problem, notes):
    work = frame.copy().reset_index(drop=True)
    if "_source_row" in work:
        raise ValueError("_source_row 是保留列名，请重命名此数据列。")
    work["_source_row"] = np.arange(1, len(work) + 1)
    columns = problem.variable_names + problem.response_names
    for c in columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    valid = np.all(np.isfinite(work[columns].to_numpy(float)), axis=1)
    audit = work.loc[~valid, ["_source_row"]].copy()
    audit["处理"] = "剔除：变量或响应含空值/非数值/无限值"
    if (~valid).any():
        notes.append(f"已剔除 {(~valid).sum()} 行无效数据，原始行号已导出。")
    work = work.loc[valid].reset_index(drop=True)
    for variable in problem.variables:
        values = work[variable.column].to_numpy(float)
        if variable.kind == "integer" and not np.allclose(values, np.rint(values), atol=1e-8, rtol=0):
            raise ValueError(f"整数变量 {variable.column} 含非整数数据。")
        if variable.kind == "discrete" and not np.isclose(values[:, None], np.asarray(variable.allowed)[None, :], atol=1e-8, rtol=0).any(axis=1).all():
            raise ValueError(f"离散变量 {variable.column} 的数据含不在允许集合中的值。")
    if problem.holdout_column:
        if work[problem.holdout_column].isna().any():
            raise ValueError("留出列含空值，请补充明确的批次标识。")
        mask = work[problem.holdout_column].astype(str).eq(problem.holdout_value)
        if not mask.any() or mask.all():
            raise ValueError("留出值必须同时留下非空训练集和检验集。")
    else:
        mask = np.zeros(len(work), bool)
    train = _replicates(work.loc[~mask].copy(), problem, notes)
    holdout = _replicates(work.loc[mask].copy(), problem, notes)
    if len(train.drop_duplicates(problem.variable_names)) < 10:
        raise ValueError("训练集至少需要 10 个不同的变量组合；重复测量不能代替不同设计。")
    for v in problem.variables:
        minimum = 3 if v.kind == "continuous" else 2
        if train[v.column].nunique() < minimum:
            raise ValueError(f"变量 {v.column} 的训练取值不足，请取消固定变量或补充样本。")
    for target in problem.target_names:
        if train[target].nunique() < 3:
            raise ValueError(f"目标 {target} 在训练集中变化不足。")
    count = len(problem.variables)
    terms = (count + 1) * (count + 2) // 2
    if len(train) < 2 * terms:
        notes.append(f"二次响应面约含 {terms} 项，训练样本仅 {len(train)} 行；需结合验证误差评估。")
    return train.reset_index(drop=True), holdout.reset_index(drop=True), audit


def _replicates(frame, problem, notes):
    if frame.empty:
        return frame
    keys = problem.variable_names
    repeated = frame.duplicated(keys, keep=False)
    if not repeated.any():
        return frame
    if problem.duplicates == "reject":
        conflicting = frame.groupby(keys, dropna=False)[problem.response_names].nunique().gt(1).any(axis=1).any()
        if conflicting:
            raise ValueError("同一变量组合存在不同响应，请选择保留重复测量或按设计取均值。")
        result = frame.drop_duplicates(keys)
        notes.append(f"相同设计及响应去重 {len(frame) - len(result)} 行。")
        return result
    if problem.duplicates == "keep":
        notes.append("保留重复测量；随机验证将按设计组合分组，避免同一设计泄漏到训练和验证两侧。")
        return frame
    metadata = [c for c in (problem.group_column if problem.validation == "group" else "", problem.time_column if problem.validation == "time" else "") if c]
    if metadata and frame.groupby(keys, dropna=False)[metadata].nunique(dropna=False).gt(1).any().any():
        raise ValueError("同一设计跨分组或时间，无法直接取均值；请保留重复测量或先整理数据。")
    aggregated = frame.groupby(keys, sort=False, dropna=False).first().reset_index()
    means = frame.groupby(keys, sort=False, dropna=False)[problem.response_names].mean().reset_index()
    aggregated = aggregated.drop(columns=problem.response_names).merge(means, on=keys, validate="one_to_one", sort=False)
    counts = frame.groupby(keys, sort=False, dropna=False).size().reset_index(name="_replicate_count")
    aggregated = aggregated.merge(counts, on=keys, validate="one_to_one", sort=False)
    notes.append(f"重复测量取均值：{len(frame)} 行合为 {len(aggregated)} 个设计；保留重复次数。")
    return aggregated


def validation_splits(train, problem):
    """Return the reordered training table and fully recorded folds."""
    strategy = problem.validation
    if strategy == "time":
        values = train[problem.time_column]
        if values.isna().any():
            raise ValueError("时间列含空值。")
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            order = np.argsort(numeric.to_numpy(), kind="stable")
            time_values = numeric.to_numpy()[order]
        else:
            dates = pd.to_datetime(values, errors="coerce")
            if dates.isna().any():
                raise ValueError("时间列需为数字或可识别的日期。")
            order = np.argsort(dates.to_numpy(), kind="stable")
            time_values = dates.to_numpy()[order]
        train = train.iloc[order].reset_index(drop=True)
        # Equal timestamps form a block; no simultaneous samples cross a fold.
        time_groups = pd.factorize(time_values, sort=False)[0]
        groups = list(dict.fromkeys(time_groups))
        if len(groups) <= problem.folds:
            raise ValueError("不同时间块数必须大于验证折数。")
        folds = []
        for a, b in TimeSeriesSplit(n_splits=problem.folds).split(groups):
            folds.append((np.flatnonzero(np.isin(time_groups, np.asarray(groups)[a])), np.flatnonzero(np.isin(time_groups, np.asarray(groups)[b]))))
        label = "按时间块前向验证；初始训练块无验证预测"
    elif strategy == "group":
        if train[problem.group_column].isna().any():
            raise ValueError("验证分组列含空值。")
        groups = train[problem.group_column].astype(str).to_numpy()
        if train.assign(_validation_group=groups).groupby(problem.variable_names)["_validation_group"].nunique().gt(1).any():
            raise ValueError("相同设计跨验证分组，可能产生泄漏；请合并相关分组或整理重复设计。")
        if len(np.unique(groups)) < problem.folds:
            raise ValueError("不同验证分组数少于折数。")
        folds = list(GroupKFold(problem.folds).split(train, groups=groups))
        label = "按指定分组验证"
    elif train.duplicated(problem.variable_names).any():
        groups = pd.util.hash_pandas_object(train[problem.variable_names], index=False).to_numpy()
        if len(np.unique(groups)) < problem.folds:
            raise ValueError("不同设计数少于折数。")
        folds = list(GroupKFold(problem.folds).split(train, groups=groups))
        label = "按重复设计分组验证"
    else:
        if len(train) < problem.folds:
            raise ValueError("训练样本数少于折数。")
        folds = list(KFold(problem.folds, shuffle=True, random_state=problem.seed).split(train))
        label = "随机 K 折验证"
    if any(len(a) < 3 or len(b) < 1 for a, b in folds):
        raise ValueError("验证折的训练样本过少，请降低折数或补充数据。")
    return train, folds, label


def _rank(frame, problem, anchors, weights=None):
    result = frame.copy()
    y = result[problem.target_names].to_numpy(float)
    result["综合效用"] = decision_scores(y, problem.weights if weights is None else weights, problem.decision, anchors, problem.directions)
    result["理想点距离"] = ideal_distances(y, problem.weights if weights is None else weights, anchors, problem.directions,
        "coefficient" if problem.decision.upper() == "IDEAL-COEFFICIENT" else "variance")
    result = result.sort_values("综合效用", ascending=False, kind="stable").reset_index(drop=True)
    result.insert(0, "推荐排序", np.arange(1, len(result) + 1))
    return result


def run_general(frame, problem: Problem, progress=None):
    started = time.perf_counter()
    problem.validate(frame)
    notes = []
    train, holdout, audit = _clean(frame, problem, notes)
    train, folds, cv_label = validation_splits(train, problem)
    variables, targets = problem.variable_names, problem.target_names
    lower, upper = np.array([v.lower for v in problem.variables]), np.array([v.upper for v in problem.variables])
    integers = tuple(i for i, v in enumerate(problem.variables) if v.kind == "integer")
    if problem.domain == "hull" and len(variables) > 6:
        raise ValueError("六维以上建议采用边界框加最近样本距离，避免凸包计算规模失控。")
    if progress:
        progress(2, "训练集验证与模型比较")
    prepared = time.perf_counter()
    with threadpool_limits(limits=1):
        bundle = fit_surrogates(train, variables, problem.response_names, lower, upper, integers, problem.seed, problem.model, problem.domain == "hull", folds)
        fitted_at = time.perf_counter()
        bundle.target_names = targets
        if any(not np.isfinite(m.r2_cv) or m.r2_cv < .8 for m in bundle.metrics):
            notes.append("存在验证 R² < 0.8 或未定义的响应；预测候选只宜用于补充样本选点，应先核查模型误差。")
        bundle.directions = problem.directions
        bundle.require_positive = False
        bundle.seed_measured = True
        bundle.allowed_values = {i: np.array(sorted(v.allowed)) for i, v in enumerate(problem.variables) if v.kind == "discrete"}
        if bundle.domain is not None and bundle.allowed_values:
            # Segment projection cannot preserve a finite allowed set.
            bundle.domain.discrete_repair = True
        evaluator = ConstraintEvaluator(problem)
        train_x = train[variables].to_numpy(float)
        origin, span = train_x.min(axis=0), np.ptp(train_x, axis=0)
        tree = cKDTree((train_x - origin) / span)
        def distances(x):
            return tree.query((np.atleast_2d(x) - origin) / span)[0]
        def responses(x, y):
            if not problem.responses:
                return y
            auxiliary = np.column_stack([bundle.models[r.column].predict(x) for r in problem.responses])
            return np.column_stack([y, auxiliary])
        def violations(x, y):
            values = responses(x, y)
            value = evaluator(x, values)
            value[np.any(~np.isfinite(values), axis=1)] = np.inf
            if problem.max_distance is not None:
                value += np.maximum(distances(x) - problem.max_distance, 0)
            for j, choices in bundle.allowed_values.items():
                value += ~np.isclose(np.asarray(x)[:, j, None], choices[None, :], rtol=0, atol=1e-8).any(axis=1)
            return value
        bundle.constraint_violation = violations
        bundle.validation_frame = bundle.validation_frame.rename(columns={"CFD计算值": "实测值"})
        bundle.validation_frame["原始行号"] = train["_source_row"].to_numpy()[bundle.validation_frame["sample_index"].to_numpy(int)]
        bundle.validation_frame["用于验证"] = bundle.validation_frame["fold"].gt(0)
        # Model families are selected and fitted using training rows only.
        holdout_rows, holdout_metrics = [], []
        if not holdout.empty:
            test_x = holdout[variables].to_numpy(float)
            prediction = np.column_stack([bundle.models[c].predict(test_x) for c in problem.response_names])
            in_range = np.all((test_x >= train_x.min(axis=0)) & (test_x <= train_x.max(axis=0)), axis=1)
            if bundle.domain is not None:
                in_range &= bundle.domain.contains(test_x)
            if pd.MultiIndex.from_frame(holdout[variables]).isin(pd.MultiIndex.from_frame(train[variables])).any():
                notes.append("留出集含训练集中已出现的设计；该部分属于响应复验，不代表新设计泛化。")
            for j, target in enumerate(problem.response_names):
                truth, pred = holdout[target].to_numpy(float), prediction[:, j]
                relative = np.full(len(truth), np.nan)
                safe = np.abs(truth) > max(float(np.max(np.abs(truth))) * 1e-8, 1e-12)
                relative[safe] = 100 * np.abs((pred[safe] - truth[safe]) / truth[safe])
                holdout_rows.append(pd.DataFrame({"原始行号": holdout["_source_row"], "目标": target, "实测值": truth,
                    "冻结模型预测值": pred, "相对误差(%)": relative, "训练域内": in_range,
                    "训练组数": len(train), "留出组数": len(holdout), "检验类型": "留出响应未参与模型选型或训练"}))
                holdout_metrics.append({"目标": target, "留出RMSE": float(np.sqrt(np.mean((truth - pred) ** 2))),
                    "留出MAE": float(np.mean(np.abs(truth - pred))), "训练域内数": int(in_range.sum()), "留出数": len(truth)})
        bundle.holdout_frame = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame()
        actual = train[targets].to_numpy(float)
        anchors = (actual.min(axis=0), actual.max(axis=0))
        if progress:
            progress(25, "工程约束内多次 NSGA-II 搜索")
        search_started = time.perf_counter()
        result = optimize_nsga2(bundle, problem.population, problem.generations, problem.seed, problem.weights,
            problem.runs, problem.decision, (lambda current, total: progress(25 + round(70 * current / total), "工程约束内多次 NSGA-II 搜索")) if progress else None, anchors)
        search_finished = time.perf_counter()
    predicted = pd.DataFrame(np.column_stack([result.variables, result.objectives]), columns=variables + targets)
    predicted = _rank(predicted, problem, anchors)
    predicted["结果类型"] = "代理模型预测"
    predicted["最近样本距离"] = distances(predicted[variables].to_numpy(float))
    predicted["约束违反量"] = violations(predicted[variables].to_numpy(float), predicted[targets].to_numpy(float))
    for r in problem.responses:
        predicted[r.column] = bundle.models[r.column].predict(predicted[variables].to_numpy(float))
    all_observed = pd.concat([train.assign(_data_role="训练"), holdout.assign(_data_role="独立留出")], ignore_index=True)
    x, y = all_observed[variables].to_numpy(float), all_observed[targets].to_numpy(float)
    within = np.all((x >= lower) & (x <= upper), axis=1)
    if bundle.domain is not None:
        within &= bundle.domain.contains(x)
    # Measured feasibility uses measured auxiliary responses, never their surrogate.
    measured_y = all_observed[problem.response_names].to_numpy(float)
    violation = evaluator(x, measured_y)
    if problem.max_distance is not None:
        violation += np.maximum(distances(x) - problem.max_distance, 0)
    diagnostic = pd.concat([all_observed, evaluator.details(x, measured_y)], axis=1)
    diagnostic["搜索域内"] = within
    diagnostic["最近样本距离"] = distances(x)
    diagnostic["约束违反量"] = violation
    diagnostic["可行"] = within & (violation <= 1e-10)
    observed = diagnostic.loc[diagnostic["可行"]].copy()
    if not observed.empty:
        first = _non_dominated_sort(observed[targets].to_numpy(float) * objective_signs(problem.directions, len(targets)))[0][0]
        observed = _rank(observed.iloc[first], problem, anchors)
        observed["结果类型"] = "输入表实测/计算值"
    else:
        notes.append("输入数据没有满足当前全部约束的设计；预测候选仍需补充真实计算或试验验证。")
    sensitivity = []
    scenarios = [("等权", np.ones(len(targets))), ("当前权重", problem.weights)]
    for j, target in enumerate(targets):
        w = np.ones(len(targets)); w[j] = 2
        scenarios.append((f"{target} 优先", w))
    for category, choices in (("输入表", observed), ("预测", predicted)):
        if choices.empty:
            continue
        for label, weights in scenarios:
            chosen = _rank(choices.drop(columns="推荐排序"), problem, anchors, weights).iloc[0]
            row = {"结果类型": category, "偏好": label}
            row.update({f"{target} 权重": w for target, w in zip(targets, normalized_weights(weights, len(targets)))})
            row.update({c: chosen[c] for c in variables + targets + ["理想点距离"]})
            sensitivity.append(row)
    summaries = []
    for run in result.run_summary:
        row = {k: run[k] for k in ("seed", "pareto_count", "utility")}
        row.update(zip(variables + targets, run["variables"] + run["objectives"]))
        summaries.append(row)
    import scipy, sklearn
    from . import __version__
    saved = problem.to_dict()
    saved.update(mode="general", problem=problem.to_dict(), variables=variables, target_names=targets,
        software_version=__version__,
        directions=problem.directions,
        normalized_weights=normalized_weights(problem.weights, len(targets)).tolist(), decision_method=problem.decision,
        anchor_minimum=anchors[0].tolist(), anchor_maximum=anchors[1].tolist(), anchor_source="training_only_before_search",
        search_lower=bundle.lower.tolist(), search_upper=bundle.upper.tolist(),
        sample_count=len(train), holdout_count=len(holdout), cv=cv_label, cv_folds=len(folds),
        cv_fold_sizes=[{"train": len(a), "validation": len(b)} for a, b in folds], cv_seed=problem.seed,
        model_selection="最小CV NRMSE；选型与报告共用折，非嵌套验证", selected_models={m.target: m.model_name for m in bundle.metrics},
        library_versions={"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        clean_data_sha256=hashlib.sha256(train.to_json(orient="records", double_precision=15).encode("utf-8")).hexdigest(),
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in problem.sources if Path(p).is_file()})
    saved["calculation_audit"] = dict(training=getattr(bundle, "training_audit", {}), search=result.evaluation_audit,
        selected_model_parameters={name: {k: str(v) for k, v in model.get_params(deep=True).items()} for name, model in bundle.models.items()},
        runtime_seconds=dict(preprocessing=prepared-started, model_validation_and_fit=fitted_at-prepared,
            holdout_and_constraints=search_started-fitted_at, nsga2_search=search_finished-search_started),
        distance_formula="sqrt(sum((w*delta)^2))" if problem.decision.upper() == "IDEAL-COEFFICIENT" else "sqrt(sum(w*delta^2))",
        weights="Input weights divided by their sum", normalization="Frozen training minima and maxima; improvements beyond ideal have zero loss",
        reference_score="S_c = 1/(1+D); TOPSIS/ARAS actual utility is reported separately",
        direct_cfd_solves=0)
    notes.append("CV 模型比较与误差报告共用折；独立留出结果用于评估选型后的性能。最近样本距离只表示覆盖程度。")
    if any(np.min(np.abs(train[t])) <= max(np.max(np.abs(train[t])) * 1e-8, 1e-12) for t in targets):
        notes.append("响应含零值或接近零的值，相对误差置为空；请采用 RMSE、MAE 和 NRMSE。")
    output = RunOutput(saved, bundle, all_observed, predicted, observed, pd.DataFrame(sensitivity), pd.DataFrame(summaries), pd.DataFrame(), notes)
    output.constraint_audit, output.cleaning_audit, output.holdout_metrics = diagnostic, audit, pd.DataFrame(holdout_metrics)
    output.decision_matrix = decision_matrix(predicted, targets, problem.directions, problem.weights, anchors, problem.decision)
    output.observed_decision_matrix = decision_matrix(observed, targets, problem.directions, problem.weights, anchors, problem.decision)
    output.correlation = correlations(train, variables + problem.response_names)
    output.search_history = pd.DataFrame(result.history)
    saved["calculation_audit"]["runtime_seconds"].update(postprocessing=time.perf_counter()-search_finished, total=time.perf_counter()-started)
    if progress:
        progress(100, "完成")
    return output


def design_template(problem, samples=30, center_replicates=3):
    """LHS with retained centre replicates, for any supported numeric variable type."""
    from scipy.stats import qmc
    if not problem.variables or samples < 1 or center_replicates < 0:
        raise ValueError("至少设置一个变量，采样数为正、中心重复次数非负。")
    lower, upper = np.array([v.lower for v in problem.variables]), np.array([v.upper for v in problem.variables])
    if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
        raise ValueError("试验设计边界必须有限且上限大于下限。")
    lhs = qmc.LatinHypercube(len(lower), seed=problem.seed).random(samples)
    values = np.vstack([qmc.scale(lhs, lower, upper), np.tile((lower + upper) / 2, (center_replicates, 1))])
    for j, v in enumerate(problem.variables):
        if v.kind == "integer":
            if np.ceil(v.lower) > np.floor(v.upper):
                raise ValueError(f"{v.column} 边界内没有整数。")
            values[:, j] = np.clip(np.rint(values[:, j]), np.ceil(v.lower), np.floor(v.upper))
        if v.kind == "discrete":
            choices = np.asarray(v.allowed)
            if len(choices) < 2 or np.any(choices < v.lower) or np.any(choices > v.upper):
                raise ValueError(f"{v.column} 允许值不完整或超出边界。")
            values[:, j] = choices[np.argmin(np.abs(values[:, j, None] - choices[None, :]), axis=1)]
    table = pd.DataFrame(values, columns=problem.variable_names)
    table.insert(0, "case", [f"DOE{i:03d}" for i in range(1, len(table) + 1)])
    table["design_role"] = ["LHS"] * samples + ["center_replicate"] * center_replicates
    for target in problem.response_names:
        table[target] = np.nan
    return table
