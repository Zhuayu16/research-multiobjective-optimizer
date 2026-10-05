from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .core import Metric, ModelBundle, _candidate_models
from .domain import SampleDomain


PRESSURE = "dp_Pa"
POWER = "P_total_W"
EFFICIENCY = "eta_gross_pct"
RESPONSES = [PRESSURE, "P_MTPV_W", "P_MTEG_W"]
OBJECTIVES = [PRESSURE, POWER, EFFICIENCY]


@dataclass(frozen=True)
class FuelDefinition:
    phi_index: int
    flow_index: int
    flow_scale: float = 1e-5
    lhv_J_kg: float = 119960000.0
    stoichiometric_air_fuel_ratio: float = 34.32

    def fraction(self, x):
        phi = np.atleast_2d(x)[:, self.phi_index]
        return phi / (self.stoichiometric_air_fuel_ratio + phi)

    def power(self, x):
        x = np.atleast_2d(x)
        if np.any(x[:, [self.phi_index, self.flow_index]] <= 0):
            raise ValueError("Equivalence ratio and mixture mass flow must be positive.")
        fuel = x[:, self.flow_index] * self.flow_scale * self.fraction(x) * self.lhv_J_kg
        if np.any(~np.isfinite(fuel)) or np.any(fuel <= 0):
            raise ValueError("Mixture mass flow, equivalence ratio and fuel input must be positive.")
        return fuel


def objective_values(raw, x, fuel):
    total = raw[:, 1] + raw[:, 2]
    return np.column_stack([raw[:, 0], total, 100.0 * total / fuel.power(x)])


def error_metrics(truth, predicted):
    rmse = float(np.sqrt(mean_squared_error(truth, predicted)))
    re = np.abs((predicted - truth) / truth) * 100
    return dict(r2=float(r2_score(truth, predicted)), rmse=rmse,
                nrmse=rmse / max(float(np.std(truth, ddof=1)), 1e-12),
                mae=float(mean_absolute_error(truth, predicted)),
                mape=float(re.mean()), max_re=float(re.max()))


@dataclass
class CoupledBundle(ModelBundle):
    fuel: FuelDefinition | None = None
    selection_score: float = np.nan
    holdout_frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predict_components(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.column_stack([self.models[name].predict(x) for name in RESPONSES])

    def _predict_targets(self, x):
        raw = self.predict_components(x)
        output = objective_values(raw, x, self.fuel)
        output[np.any(raw <= 0, axis=1)] = np.nan
        return output


def fit_coupled(frame, variable_names, fuel, lower=None, upper=None,
                integer_indices=(), seed=20260929, model_mode="Auto",
                use_hull=True, holdout_mask=None):
    """Compare complete component-model families using the same OOF folds."""
    x = frame[variable_names].to_numpy(float)
    y = frame[RESPONSES].to_numpy(float)
    if np.any(y <= 0) or not np.all(np.isfinite(y)):
        raise ValueError("Pressure and both stage powers must be finite and positive.")
    if not np.all(np.isfinite([fuel.flow_scale, fuel.lhv_J_kg, fuel.stoichiometric_air_fuel_ratio])) or fuel.flow_scale <= 0 or fuel.lhv_J_kg <= 0 or fuel.stoichiometric_air_fuel_ratio <= 0:
        raise ValueError("Fuel constants and flow unit scale must be positive.")
    actual = objective_values(y, x, fuel)
    if "Y_H2_in" in frame:
        supplied = pd.to_numeric(frame["Y_H2_in"], errors="raise").to_numpy(float)
        if not np.allclose(supplied, fuel.fraction(x), rtol=1e-3, atol=1e-8):
            raise ValueError("Y_H2_in disagrees with the selected H2/air composition; check fuel and flow definitions.")
    for index, column in ((1, POWER), (2, EFFICIENCY)):
        if column in frame and not np.allclose(frame[column].to_numpy(float), actual[:, index], rtol=1e-3, atol=1e-6):
            raise ValueError(f"{column} is inconsistent with stage powers and fuel input.")
    lower = x.min(axis=0) if lower is None else np.asarray(lower, float)
    upper = x.max(axis=0) if upper is None else np.asarray(upper, float)
    if np.any(upper <= lower) or np.any(lower < x.min(axis=0)) or np.any(upper > x.max(axis=0)):
        raise ValueError("Search bounds must lie within the input sample ranges.")
    domain = SampleDomain(x, lower, upper, integer_indices) if use_hull else None
    templates = _candidate_models(seed, len(variable_names))
    # Auto in the research workflow compares the two manuscript model families.
    names = ["RSM-Quadratic", "GaussianProcess"] if model_mode == "Auto" else [model_mode]
    if any(name not in templates for name in names):
        raise ValueError(f"Unknown model mode: {model_mode}")
    splits = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(x))
    fold_ids = np.empty(len(x), int)
    for fold, (_, test) in enumerate(splits, 1):
        fold_ids[test] = fold
    candidates, predictions, evaluations = [], {}, {}
    for name in names:
        raw = np.empty_like(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for train, test in splits:
                for j in range(3):
                    raw[test, j] = clone(templates[name]).fit(x[train], y[train, j]).predict(x[test])
        predicted = objective_values(raw, x, fuel)
        values = [error_metrics(actual[:, j], predicted[:, j]) for j in range(3)]
        score = float(np.mean([values[0]["mape"], values[1]["mape"]]))
        evaluations[name] = (score, values)
        predictions[name] = predicted
        for target, metrics in zip(OBJECTIVES, values):
            candidates.append({"目标": target, "模型": name, "交叉验证R2": metrics["r2"],
                               "交叉验证RMSE": metrics["rmse"], "归一化RMSE": metrics["nrmse"],
                               "交叉验证MAE": metrics["mae"], "MAPE(%)": metrics["mape"],
                               "最大相对误差(%)": metrics["max_re"], "选型平均MAPE(%)": score, "状态": "可用"})
    chosen = min(evaluations, key=lambda name: evaluations[name][0])
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for j, target in enumerate(RESPONSES):
            models[target] = clone(templates[chosen]).fit(x, y[:, j])
    fitted = objective_values(np.column_stack([models[n].predict(x) for n in RESPONSES]), x, fuel)
    cases = frame["case"].astype(str).to_numpy() if "case" in frame else np.arange(1, len(x) + 1).astype(str)
    validation, metrics = [], []
    for j, target in enumerate(OBJECTIVES):
        v = evaluations[chosen][1][j]
        metrics.append(Metric(target, chosen, v["r2"], v["rmse"], v["nrmse"], v["mae"],
                              float(r2_score(actual[:, j], fitted[:, j])), v["mape"], v["max_re"]))
        validation.append(pd.DataFrame({"case": cases, "fold": fold_ids, "目标": target, "模型": chosen,
                                        "CFD计算值": actual[:, j], "交叉验证预测值": predictions[chosen][:, j],
                                        "相对误差(%)": 100 * np.abs(predictions[chosen][:, j] / actual[:, j] - 1)}))
    holdout = pd.DataFrame()
    if holdout_mask is not None and np.any(holdout_mask):
        mask = np.asarray(holdout_mask, bool)
        if mask.shape != (len(x),) or np.sum(~mask) < 10:
            raise ValueError("Holdout mask requires at least ten remaining training designs.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            raw = np.column_stack([clone(templates[chosen]).fit(x[~mask], y[~mask, j]).predict(x[mask]) for j in range(3)])
        predicted = objective_values(raw, x[mask], fuel)
        rows = []
        for i, case in enumerate(cases[mask]):
            for j, target in enumerate(OBJECTIVES):
                rows.append({"case": case, "目标": target, "CFD计算值": actual[mask][i, j],
                             "冻结模型预测值": predicted[i, j], "相对误差(%)": 100 * abs(predicted[i, j] / actual[mask][i, j] - 1),
                             "训练组数": int(np.sum(~mask)), "留出组数": int(np.sum(mask)),
                             "检验类型": "响应留出；模型类型采用全样本CV选型"})
        holdout = pd.DataFrame(rows)
    return CoupledBundle(variable_names, OBJECTIVES.copy(), models, metrics, pd.DataFrame(candidates),
                         pd.concat(validation, ignore_index=True), lower, upper, integer_indices,
                         domain, x, fuel, evaluations[chosen][0], holdout)
