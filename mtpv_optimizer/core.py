from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import warnings
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from .dat_parser import parse_dat_file
from .domain import SampleDomain


@dataclass(frozen=True)
class Metric:
    target: str
    model_name: str
    r2_cv: float
    rmse_cv: float
    nrmse_cv: float
    mae_cv: float
    r2_fit: float
    mape_cv_pct: float = np.nan
    max_re_cv_pct: float = np.nan


@dataclass
class ModelBundle:
    variable_names: list[str]
    target_names: list[str]
    models: dict[str, Any]
    metrics: list[Metric]
    candidate_metrics: pd.DataFrame
    validation_frame: pd.DataFrame
    lower: np.ndarray
    upper: np.ndarray
    integer_indices: tuple[int, ...] = ()
    domain: SampleDomain | None = None
    training_x: np.ndarray | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float)).copy()
        valid = np.all(np.isfinite(x), axis=1) & np.all((x >= self.lower) & (x <= self.upper), axis=1)
        if self.integer_indices:
            x[:, self.integer_indices] = np.rint(x[:, self.integer_indices])
            valid &= np.all((x >= self.lower) & (x <= self.upper), axis=1)
        if self.domain is not None:
            valid &= self.domain.contains(x)
        output = np.full((len(x), len(self.target_names)), np.nan)
        if valid.any():
            output[valid] = self._predict_targets(x[valid])
        return output

    def _predict_targets(self, x):
        return np.column_stack([self.models[name].predict(x) for name in self.target_names])


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
            raise ValueError("JSON must contain a nonempty list of case records.")
        return pd.DataFrame(records)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    if suffix == ".xls":
        return pd.read_excel(path)
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    if path.name.lower().endswith(".dat.h5"):
        raise ValueError(
            "Fluent .dat.h5 是二进制求解结果，不能直接作为优化表读取。"
            "请使用配套 case 在 Fluent 中导出报告，或导出文本 .dat。"
        )
    if suffix == ".dat":
        frame, _ = parse_dat_file(path)
        return frame
    raise ValueError(f"不支持的文件格式：{suffix}。请使用 xlsx、xls、csv、tsv 或文本 dat。")


def validate_dataset(
    frame: pd.DataFrame,
    variable_names: Iterable[str],
    target_names: Iterable[str],
    integer_indices: Iterable[int] = (),
) -> tuple[pd.DataFrame, list[str]]:
    variable_names = list(variable_names)
    target_names = list(target_names)
    columns = variable_names + target_names
    if len(set(columns)) != len(columns):
        raise ValueError("Variable and response columns must be distinct.")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError("缺少列：" + "、".join(missing))

    clean = frame.loc[:, columns].copy()
    for name in columns:
        clean[name] = pd.to_numeric(clean[name], errors="coerce")
    bad_rows = int((~np.isfinite(clean.to_numpy(float))).any(axis=1).sum())
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates().reset_index(drop=True)
    if clean.duplicated(variable_names).any():
        raise ValueError("Duplicate input designs have different responses; reconcile batches before fitting.")
    warnings: list[str] = []
    if bad_rows:
        warnings.append(f"已忽略 {bad_rows} 行空值或非数值数据。")
    if len(clean) < 10:
        raise ValueError(f"有效样本仅 {len(clean)} 组，至少需要 10 组，建议按试验设计补充。")

    integer_indices = set(integer_indices)
    for i, name in enumerate(variable_names):
        if i in integer_indices and not np.allclose(clean[name], np.rint(clean[name]), atol=1e-8, rtol=0):
            raise ValueError(f"Integer variable {name} contains noninteger values.")
        unique_count = clean[name].nunique()
        minimum_unique = 2 if i in integer_indices else 3
        if unique_count < minimum_unique:
            raise ValueError(
                f"变量“{name}”只有 {unique_count} 个不同取值，无法用于当前优化。"
                "如果入口质量流量始终为 4.5，请先补充不同流量工况，或暂时取消该变量。"
            )

    for name in target_names:
        if clean[name].nunique() < 3:
            raise ValueError(f"目标“{name}”变化不足，无法建立代理模型。")

    n_var = len(variable_names)
    quadratic_terms = 1 + 2 * n_var + n_var * (n_var - 1) // 2
    recommended = max(20, 2 * quadratic_terms)
    if len(clean) < recommended:
        warnings.append(
            f"二次响应面含约 {quadratic_terms} 个系数，当前仅 {len(clean)} 组样本；"
            f"建议至少 {recommended} 组，并以交叉验证结果判断是否可用。"
        )
    return clean, warnings


def _candidate_models(seed: int, n_variables: int = 3) -> dict[str, Any]:
    scaled_linear = Pipeline([("scale", StandardScaler()), ("regression", LinearRegression())])
    quadratic = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )
    mlp = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regression",
                TransformedTargetRegressor(
                    regressor=MLPRegressor(
                        hidden_layer_sizes=(32, 16),
                        activation="tanh",
                        solver="lbfgs",
                        alpha=1e-3,
                        max_iter=4000,
                        random_state=seed,
                    ),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )
    gpr = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regression",
                GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0, (1e-2, 1e2))
                    * Matern(length_scale=np.ones(n_variables), length_scale_bounds=(0.1, 50), nu=2.5),
                    alpha=1e-5,
                    normalize_y=True,
                    n_restarts_optimizer=1,
                    random_state=seed,
                ),
            ),
        ]
    )
    return {
        "Linear": scaled_linear,
        "Ridge-L2": Pipeline([("scale", StandardScaler()), ("regression", Ridge(alpha=1.0))]),
        "ElasticNet": Pipeline([("scale", StandardScaler()), ("regression", TransformedTargetRegressor(
            regressor=ElasticNet(alpha=0.003, l1_ratio=0.5, max_iter=10000, random_state=seed), transformer=StandardScaler()))]),
        "RSM-Quadratic": quadratic,
        "RSM-Cubic": Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scale", StandardScaler()), ("regression", Ridge(alpha=0.1))]),
        "KNN-Distance": Pipeline([("scale", StandardScaler()),
            ("regression", KNeighborsRegressor(n_neighbors=3, weights="distance", n_jobs=1))]),
        "AdaBoost": AdaBoostRegressor(n_estimators=120, learning_rate=0.05, random_state=seed),
        "MLP-NeuralNet": mlp,
        "GaussianProcess": gpr,
        "SVR-RBF": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "regression",
                    TransformedTargetRegressor(
                        regressor=SVR(C=20.0, epsilon=0.03, gamma="scale"),
                        transformer=StandardScaler(),
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=1.0, random_state=seed, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.035, max_depth=2, loss="huber", random_state=seed
        ),
    }


def fit_surrogates(
    frame: pd.DataFrame,
    variable_names: list[str],
    target_names: list[str],
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    integer_indices: tuple[int, ...] = (),
    seed: int = 42,
    model_mode: str = "Auto",
    use_hull: bool = False,
    cv_splits=None,
) -> ModelBundle:
    x = frame.loc[:, variable_names].to_numpy(dtype=float)
    if lower is None:
        lower = np.min(x, axis=0)
    if upper is None:
        upper = np.max(x, axis=0)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if np.any(upper <= lower):
        raise ValueError("每个变量的上限必须大于下限。")
    data_lower, data_upper = np.min(x, axis=0), np.max(x, axis=0)
    # Spin-box formatting may change the last displayed digits of a data bound.
    # Snap only numerical-scale differences, never a substantive extrapolation.
    tolerance = np.maximum(np.abs(data_lower), np.abs(data_upper)) * 1e-11
    lower = np.where(np.abs(lower - data_lower) <= tolerance, data_lower, lower)
    upper = np.where(np.abs(upper - data_upper) <= tolerance, data_upper, upper)
    if np.any(lower < data_lower) or np.any(upper > data_upper):
        raise ValueError("优化边界超出训练数据范围。为避免代理模型外推，请先补充相应 CFD 样本。")

    splits = min(5, max(2, len(frame) // 4))
    cv = list(KFold(n_splits=splits, shuffle=True, random_state=seed).split(x)) if cv_splits is None else list(cv_splits)
    fold_ids = np.zeros(len(x), dtype=int)
    for fold, (_, test) in enumerate(cv, 1):
        if np.any(fold_ids[test]):
            raise ValueError("Validation folds must not overlap.")
        fold_ids[test] = fold
    scored = fold_ids > 0
    if scored.sum() < 2:
        raise ValueError("Validation requires at least two held-out observations.")
    models: dict[str, Any] = {}
    metrics: list[Metric] = []
    candidate_rows: list[dict[str, object]] = []
    validation_rows: list[pd.DataFrame] = []
    candidates = _candidate_models(seed, len(variable_names))
    if model_mode != "Auto":
        if model_mode not in candidates:
            raise ValueError(f"未知代理模型：{model_mode}")
        candidates = {model_mode: candidates[model_mode]}
    fit_attempts = fit_completed = 0
    validation_timings = []
    for name in target_names:
        y = frame[name].to_numpy(dtype=float)
        scale = max(float(np.std(y, ddof=1)), 1e-12)
        evaluated: list[tuple[float, str, Any, np.ndarray, dict[str, float]]] = []
        for model_name, candidate in candidates.items():
            candidate_started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=UserWarning)
                    predicted_cv = np.full(len(x), np.nan)
                    for train, test in cv:
                        fit_attempts += 1
                        fitted = clone(candidate).fit(x[train], y[train])
                        fit_completed += 1
                        predicted_cv[test] = fitted.predict(x[test])
                truth, pred = y[scored], predicted_cv[scored]
                rmse = float(np.sqrt(mean_squared_error(truth, pred)))
                # Relative errors are undefined near zero; keep absolute metrics usable.
                relative_valid = np.abs(truth) > max(np.max(np.abs(truth)) * 1e-8, 1e-12)
                values = {
                    "r2": float(r2_score(truth, pred)),
                    "rmse": rmse,
                    "nrmse": rmse / scale,
                    "mae": float(mean_absolute_error(truth, pred)),
                    "mape": float(np.mean(np.abs((truth - pred) / truth)) * 100) if relative_valid.all() else np.nan,
                    "max_re": float(np.max(np.abs((truth - pred) / truth)) * 100) if relative_valid.all() else np.nan,
                }
                evaluated.append((values["nrmse"], model_name, candidate, predicted_cv, values))
                candidate_rows.append(
                    {
                        "目标": name,
                        "模型": model_name,
                        "交叉验证R2": values["r2"],
                        "交叉验证RMSE": values["rmse"],
                        "归一化RMSE": values["nrmse"],
                        "交叉验证MAE": values["mae"],
                        "MAPE(%)": values["mape"],
                        "最大相对误差(%)": values["max_re"],
                        "状态": "可用",
                    }
                )
            except Exception as exc:
                candidate_rows.append(
                    {
                        "目标": name,
                        "模型": model_name,
                        "交叉验证R2": np.nan,
                        "交叉验证RMSE": np.nan,
                        "归一化RMSE": np.nan,
                        "交叉验证MAE": np.nan,
                        "状态": f"失败: {exc}",
                    }
                )
            validation_timings.append(dict(target=name, model=model_name, seconds=time.perf_counter()-candidate_started))
        if not evaluated:
            raise ValueError(f"目标“{name}”的所有候选模型均训练失败。")
        _, best_name, best_template, predicted_cv, values = min(evaluated, key=lambda item: item[0])
        model = clone(best_template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fit_attempts += 1
            model.fit(x, y)
            fit_completed += 1
        predicted_fit = model.predict(x)
        metrics.append(
            Metric(
                target=name,
                model_name=best_name,
                r2_cv=values["r2"],
                rmse_cv=values["rmse"],
                nrmse_cv=values["nrmse"],
                mae_cv=values["mae"],
                r2_fit=float(r2_score(y, predicted_fit)),
                mape_cv_pct=values["mape"],
                max_re_cv_pct=values["max_re"],
            )
        )
        models[name] = model
        validation_rows.append(
            pd.DataFrame(
                {
                    "目标": name,
                    "模型": best_name,
                    "sample_index": np.arange(len(x)),
                    "fold": fold_ids,
                    "CFD计算值": y,
                    "交叉验证预测值": predicted_cv,
                }
            )
        )

    bundle = ModelBundle(
        variable_names=variable_names,
        target_names=target_names,
        models=models,
        metrics=metrics,
        candidate_metrics=pd.DataFrame(candidate_rows),
        validation_frame=pd.concat(validation_rows, ignore_index=True),
        lower=lower,
        upper=upper,
        integer_indices=integer_indices,
        domain=SampleDomain(x, lower, upper, integer_indices) if use_hull else None,
        training_x=x,
    )
    bundle.training_audit = dict(fit_attempts=fit_attempts, fits_completed=fit_completed,
        candidate_families=list(candidates), response_count=len(target_names), validation_folds=len(cv), validation_timings=validation_timings)
    return bundle


def objective_signs(directions, count):
    directions = ["min"] + ["max"] * (count - 1) if directions is None else list(directions)
    if len(directions) != count or any(d not in {"min", "max"} for d in directions):
        raise ValueError("Every objective needs a min or max direction.")
    return np.array([1.0 if d == "min" else -1.0 for d in directions])


def _objective_array(objectives):
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("A nonempty matrix of finite objectives is required.")
    return values


def normalized_weights(weights=None, n_objectives=3):
    weights = np.ones(n_objectives) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n_objectives,) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError(f"{n_objectives} finite, nonnegative weights with a positive sum are required.")
    return weights / weights.sum()


def aras_scores(objectives, weights=None, directions=None):
    values = _objective_array(objectives)
    count = values.shape[1]
    weights = normalized_weights(weights, count)
    shifted = values - np.minimum(values.min(axis=0), 0.0) + 1e-12
    costs = objective_signs(directions, count) > 0
    shifted[:, costs] = 1.0 / shifted[:, costs]
    score = (shifted / np.maximum(shifted.sum(axis=0), 1e-12)) @ weights
    return score / max(float(score.max()), 1e-12)


def topsis_scores(objectives, weights=None, directions=None):
    values = _objective_array(objectives)
    weights = normalized_weights(weights, values.shape[1])
    spans = np.maximum(np.ptp(values, axis=0), 1e-12)
    desirable = (values - values.min(axis=0)) / spans
    costs = objective_signs(directions, values.shape[1]) > 0
    desirable[:, costs] = 1.0 - desirable[:, costs]
    positive_distance = np.sqrt(np.sum(weights * (1.0 - desirable) ** 2, axis=1))
    negative_distance = np.sqrt(np.sum(weights * desirable**2, axis=1))
    return negative_distance / np.maximum(positive_distance + negative_distance, 1e-12)


def ideal_distances(objectives, weights=None, anchors=None, directions=None, weighting="variance"):
    values = _objective_array(objectives)
    count = values.shape[1]
    if anchors is None:
        raise ValueError("IDEAL ranking requires fixed minimum/maximum anchors.")
    low, high = (np.asarray(a, dtype=float) for a in anchors)
    if low.shape != (count,) or high.shape != (count,) or not np.all(np.isfinite([low, high])) or np.any(high <= low):
        raise ValueError("Each fixed anchor maximum must exceed its minimum.")
    cost = (values - low) / (high - low)
    benefits = objective_signs(directions, count) < 0
    cost[:, benefits] = 1.0 - cost[:, benefits]
    # The legacy manuscript retains signed distances. General projects count
    # improvements beyond the ideal anchor as zero loss, never as a penalty.
    if directions is not None:
        cost = np.maximum(cost, 0.0)
    if weighting not in {"variance", "coefficient"}:
        raise ValueError("Distance weighting must be variance or coefficient.")
    w = normalized_weights(weights, count)
    return np.sqrt(np.sum((w if weighting == "variance" else w**2) * cost**2, axis=1))


def decision_scores(objectives, weights=None, method="TOPSIS", anchors=None, directions=None):
    if method.upper() == "IDEAL":
        return 1.0 / (1.0 + ideal_distances(objectives, weights, anchors, directions))
    if method.upper() == "IDEAL-COEFFICIENT":
        return 1.0 / (1.0 + ideal_distances(objectives, weights, anchors, directions, "coefficient"))
    if method.upper() == "TOPSIS":
        return topsis_scores(objectives, weights, directions)
    if method.upper() == "ARAS":
        return aras_scores(objectives, weights, directions)
    raise ValueError(f"Unknown decision method: {method}")


def metrics_frame(bundle: ModelBundle) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "目标": m.target,
                "选用模型": m.model_name,
                "交叉验证R2": m.r2_cv,
                "交叉验证RMSE": m.rmse_cv,
                "归一化RMSE": m.nrmse_cv,
                "交叉验证MAE": m.mae_cv,
                "拟合R2": m.r2_fit,
                "MAPE(%)": m.mape_cv_pct,
                "最大相对误差(%)": m.max_re_cv_pct,
            }
            for m in bundle.metrics
        ]
    )
