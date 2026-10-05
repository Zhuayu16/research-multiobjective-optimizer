from __future__ import annotations

import math

import numpy as np

from .core import objective_signs
from .figures import apply_publication_style, PALETTE


def label(item):
    return item.column + (f" ({item.unit})" if item.unit else "")


def draw_general_pareto(fig, output, problem, axes=None):
    apply_publication_style("Microsoft YaHei")
    fig.clear()
    fig.set_size_inches(9, 6, forward=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1), wspace=.5, hspace=.6)
    hero, balance, rank = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    result, observed = output.predicted, output.observed
    objectives = problem.objectives
    m = len(objectives)
    if m == 1:
        x_name, x_label = problem.variables[0].column, label(problem.variables[0])
        y_index, color_index = 0, None
    else:
        x_index, y_index, color_index = axes or (0, 1, 2 if m > 2 else None)
        x_name, x_label = objectives[x_index].column, label(objectives[x_index])
    y_name = objectives[y_index].column
    colors = result[objectives[color_index].column] if color_index is not None else PALETTE["blue"]
    scatter = hero.scatter(result[x_name], result[y_name], c=colors, cmap="viridis" if color_index is not None else None, s=24, alpha=.85)
    hero.scatter(result[x_name].iloc[0], result[y_name].iloc[0], marker="*", color=PALETTE["red"], s=120, label="预测首选", zorder=4)
    if not observed.empty:
        hero.scatter(observed[x_name], observed[y_name], facecolors="none", edgecolors="#333333", s=35, label="输入表可行 Pareto")
    hero.set_xlabel(x_label); hero.set_ylabel(label(objectives[y_index]))
    hero.set_title("目标权衡与预测候选", loc="left"); hero.legend(loc="best")
    if color_index is not None:
        fig.colorbar(scatter, ax=hero, fraction=.04, pad=.03).set_label(label(objectives[color_index]))
    values = result[problem.target_names].to_numpy(float)
    lo, hi = np.array(output.config["anchor_minimum"]), np.array(output.config["anchor_maximum"])
    desirable = (values - lo) / (hi - lo)
    costs = objective_signs(problem.directions, m) > 0
    desirable[:, costs] = 1 - desirable[:, costs]
    for i in np.linspace(0, len(result) - 1, min(70, len(result)), dtype=int):
        balance.plot(np.arange(m), desirable[i], color=PALETTE["blue_pale"], alpha=.4, lw=.7)
    balance.plot(np.arange(m), desirable[0], color=PALETTE["red"], lw=2, marker="o", ms=3)
    balance.set_xticks(np.arange(m)); balance.set_xticklabels([("↓ " if o.direction == "min" else "↑ ") + label(o) for o in objectives], rotation=25, ha="right")
    balance.set_ylabel("固定训练尺度下的期望程度")
    balance.set_title("全部目标的首选方案表现", loc="left")
    utility = result["综合效用"].to_numpy(float)
    rank.plot(np.arange(1, len(result) + 1), utility, color=PALETTE["blue"])
    rank.set_xlabel("推荐排序"); rank.set_ylabel(problem.decision + " 效用")
    rank.set_title("权重决策", loc="left")
    fig.subplots_adjust(left=.1, right=.94, bottom=.15, top=.9)


def draw_general_validation(fig, output, problem):
    apply_publication_style("Microsoft YaHei")
    fig.clear()
    items = problem.objectives + problem.responses
    m = len(items)
    columns = min(3, m)
    rows = math.ceil(m / columns)
    fig.set_size_inches(9, 3.2 * rows + 2.8, forward=True)
    grid = fig.add_gridspec(rows + 1, columns, hspace=.7, wspace=.45)
    frame = output.bundle.validation_frame
    palette = [PALETTE["blue"], PALETTE["teal"], PALETTE["red"], "#815cab"]
    for j, target in enumerate(items):
        ax = fig.add_subplot(grid[j // columns, j % columns])
        subset = frame.loc[frame["目标"].eq(target.column) & frame["用于验证"]]
        truth, pred = subset["实测值"].to_numpy(float), subset["交叉验证预测值"].to_numpy(float)
        lo, hi = min(truth.min(), pred.min()), max(truth.max(), pred.max())
        pad = max((hi - lo) * .05, 1e-12)
        ax.scatter(truth, pred, s=20, alpha=.8, color=palette[j % len(palette)])
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888888", ls="--", lw=.8)
        ax.set_xlabel("输入响应"); ax.set_ylabel("验证预测")
        metric = output.bundle.metrics[j]
        ax.set_title(f"{label(target)}\n{metric.model_name}，R²={metric.r2_cv:.3f}", loc="left", fontsize=8)
    comparison = fig.add_subplot(grid[-1, :])
    candidates = output.bundle.candidate_metrics
    usable = candidates.loc[candidates["状态"].eq("可用")]
    models = list(dict.fromkeys(usable["模型"]))
    for j, target in enumerate(items):
        subset = usable.loc[usable["目标"].eq(target.column)].set_index("模型")
        values = [subset.loc[n, "归一化RMSE"] if n in subset.index else np.nan for n in models]
        comparison.scatter(np.arange(len(models)) + (j - (m - 1) / 2) * .4 / max(m, 1), values, label=target.column, s=22, color=palette[j % len(palette)])
    comparison.set_xticks(np.arange(len(models))); comparison.set_xticklabels(models, rotation=20, ha="right")
    comparison.set_ylabel("验证 NRMSE"); comparison.set_title(output.config["cv"], loc="left")
    comparison.legend(ncol=min(m, 4), fontsize=7)
    fig.subplots_adjust(left=.09, right=.97, bottom=.12, top=.9)
