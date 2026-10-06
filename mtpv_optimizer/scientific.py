"""Auditable numerical evidence for decision matrices and research plots."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from .core import decision_scores, normalized_weights


def decision_matrix(frame, targets, directions, weights, anchors, method="IDEAL"):
    if frame.empty:
        return pd.DataFrame()
    y = frame[targets].to_numpy(float)
    low, high = map(np.asarray, anchors)
    scaled = (y - low) / (high - low)
    loss = np.where(np.asarray(directions) == "min", scaled, 1 - scaled)
    effective = np.maximum(loss, 0)
    w = normalized_weights(weights, len(targets))
    coefficient = method.upper() == "IDEAL-COEFFICIENT"
    contribution = (w**2 if coefficient else w) * effective**2
    result = frame.copy()
    for j, target in enumerate(targets):
        result[f"归一化::{target}"] = scaled[:, j]
        result[f"方向损失::{target}"] = loss[:, j]
        result[f"有效损失::{target}"] = effective[:, j]
        result[f"加权平方::{target}"] = contribution[:, j]
    result["D"] = np.sqrt(contribution.sum(axis=1))
    result["S_c"] = 1 / (1 + result["D"])
    result["复算综合效用"] = decision_scores(y, weights, method, anchors, directions)
    result["效用复算差"] = result["复算综合效用"] - frame["综合效用"].to_numpy(float)
    return result


def correlations(frame, columns):
    data = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pairs = data.notna().astype(int).T @ data.notna().astype(int)
    return dict(columns=list(columns), pearson=data.corr(method="pearson", min_periods=3),
        spearman=data.corr(method="spearman", min_periods=3), pairs=pairs,
        training_rows=len(data), constant_columns=[c for c in columns if data[c].nunique() < 2],
        scope="Training rows only; correlation is association, not causal effect.")


def spatial_field(frame, x, y, value, resolution=50):
    if len({x, y, value}) != 3 or any(c not in frame for c in (x, y, value)):
        raise ValueError("空间场需要三个不同数据列：位置 X、位置 Y 与温度/场变量。")
    if not 16 <= int(resolution) <= 100:
        raise ValueError("插值网格每个方向应为 16–100 点。")
    work = frame[[x, y, value]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    dropped = len(frame) - len(work)
    if work.groupby([x, y])[value].nunique().gt(1).any():
        raise ValueError("同一位置出现不同场值。请先选定一个工况/时刻/截面，再导入空间数据。")
    work = work.drop_duplicates([x, y])
    points = work[[x, y]].to_numpy(float)
    if len(work) < 3 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("二维云图至少需要三个不共线的位置点；一维壁面剖面应使用散点图。")
    gx = np.linspace(points[:, 0].min(), points[:, 0].max(), int(resolution))
    gy = np.linspace(points[:, 1].min(), points[:, 1].max(), int(resolution))
    xx, yy = np.meshgrid(gx, gy)
    zz = griddata(points, work[value].to_numpy(float), (xx, yy), method="linear", fill_value=np.nan)
    grid = pd.DataFrame({x: xx.ravel(), y: yy.ravel(), value: zz.ravel()})
    return dict(points=work, grid=grid, x=gx.tolist(), y=gy.tolist(), resolution=int(resolution),
        valid_cells=int(np.isfinite(zz).sum()), masked_cells=int(np.isnan(zz).sum()),
        input_rows=len(frame), unique_points=len(work), dropped_rows=dropped,
        data_sha256=hashlib.sha256(work.to_json(orient="split", double_precision=15).encode()).hexdigest(),
        method="SciPy griddata linear (Delaunay triangulation); outside convex hull masked; no extrapolation.",
        scope="Interpolation of supplied coordinates and field values, not a newly solved CFD temperature field.")
