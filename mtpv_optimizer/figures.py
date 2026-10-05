from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


PALETTE = {
    "blue": "#0F4D92",
    "blue_soft": "#7884B4",
    "blue_pale": "#B4C0E4",
    "red": "#B64342",
    "teal": "#42949E",
    "neutral": "#767676",
    "light": "#D8D8D8",
    "black": "#272727",
}


def apply_publication_style(font_name: str = "Arial") -> None:
    nature_font = font_name.strip() or "Arial"
    is_serif = "times" in nature_font.lower()
    mpl.rcParams.update(
        {
            "font.family": "serif" if is_serif else "sans-serif",
            "font.serif": [nature_font, "DejaVu Serif"],
            "font.sans-serif": [nature_font, "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
        }
    )


def _panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.03, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def _publication_label(name: str) -> str:
    text = str(name)
    canonical = {"dp_Pa": "Pressure drop (Pa)", "P_total_W": "Gross electric power (W)",
                 "eta_gross_pct": "Gross fuel-based efficiency (%)", "P_MTPV_W": "MTPV power (W)",
                 "P_MTEG_W": "MTEG power (W)", "L_mm": "Fin length (mm)",
                 "phi": "Equivalence ratio", "qm_1e5_kg_s": "Mixture mass flow (10^-5 kg/s)"}
    if text in canonical:
        return canonical[text]
    unit = ""
    if "(" in text and ")" in text:
        unit = " " + text[text.find("(") : text.rfind(")") + 1]
    mappings = (
        ("压降", r"Pressure drop, $\Delta p$"),
        ("能量输出", "Energy output"),
        ("输出功率", "Output power"),
        ("净输出", "Net output power"),
        ("系统效率", "System efficiency"),
        ("净能量效率", "Net energy efficiency"),
        ("当量比", r"Equivalence ratio, $\phi$"),
        ("质量流量", "Mass flow rate"),
        ("翅片长度", "Fin length"),
        ("翅片数", "Number of fins"),
    )
    for source, target in mappings:
        if source in text:
            return target + unit
    if text.strip().lower() == "pnet":
        return r"Net output power, $P_{net}$"
    return text


def _desirability(objectives: np.ndarray, anchors=None) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    minima, maxima = (values.min(axis=0), values.max(axis=0)) if anchors is None else anchors
    spans = np.maximum(maxima - minima, 1e-12)
    normalized = (values - minima) / spans
    normalized[:, 0] = 1.0 - normalized[:, 0]
    return normalized


def draw_pareto_figure(
    fig: Figure,
    result: pd.DataFrame,
    variable_names: list[str],
    target_names: list[str],
    font_name: str = "Arial",
    decision_method: str = "TOPSIS",
    observed: pd.DataFrame | None = None,
    anchors=None,
) -> None:
    apply_publication_style(font_name)
    fig.clear()
    fig.set_size_inches(7.2, 4.8, forward=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0), height_ratios=(1.15, 0.85), wspace=0.70, hspace=0.48)
    hero = fig.add_subplot(grid[:, 0])
    tradeoff = fig.add_subplot(grid[0, 1])
    ranking = fig.add_subplot(grid[1, 1])

    x = result[target_names[0]].to_numpy(float)
    y = result[target_names[1]].to_numpy(float)
    efficiency = result[target_names[2]].to_numpy(float)
    scatter = hero.scatter(
        x,
        y,
        c=efficiency,
        cmap="viridis",
        s=26,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.92,
        zorder=2,
        label="Model candidates",
    )
    hero.scatter(x[0], y[0], s=105, marker="*", color=PALETTE["red"], edgecolors="white", linewidths=0.7, zorder=4)
    hero.annotate(
        "Model choice",
        (x[0], y[0]),
        xytext=(12, -20),
        textcoords="offset points",
        color=PALETTE["red"],
        fontsize=7,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": PALETTE["red"], "linewidth": 0.7},
    )
    hero.set_xlabel(_publication_label(target_names[0]))
    hero.set_ylabel(_publication_label(target_names[1]))
    hero.set_title("Pareto trade-off landscape", loc="left", fontweight="bold")
    if observed is not None and not observed.empty:
        ox, oy = observed[target_names[0]].to_numpy(float), observed[target_names[1]].to_numpy(float)
        hero.scatter(ox, oy, marker="o", facecolors="none", edgecolors=PALETTE["black"], s=36, linewidths=0.8,
                     label="Calculated Pareto", zorder=3)
        hero.scatter(ox[0], oy[0], marker="D", color=PALETTE["black"], s=40, edgecolors="white", linewidths=0.5,
                     label="Calculated choice", zorder=5)
    hero.legend(loc="upper left", fontsize=6)
    colorbar = fig.colorbar(scatter, ax=hero, pad=0.02, fraction=0.045)
    colorbar.set_label("Gross efficiency (%)" if target_names[2] == "eta_gross_pct" else _publication_label(target_names[2]))
    colorbar.outline.set_linewidth(0.6)
    _panel_label(hero, "a")

    objective_values = result[target_names].to_numpy(float)
    desirable = _desirability(objective_values, anchors)
    max_lines = min(80, len(result))
    selected = np.linspace(0, len(result) - 1, max_lines, dtype=int)
    axis_x = np.arange(3)
    for index in selected:
        tradeoff.plot(axis_x, desirable[index], color=PALETTE["blue_pale"], lw=0.65, alpha=0.34)
    tradeoff.plot(axis_x, desirable[0], color=PALETTE["red"], lw=2.0, marker="o", ms=3.5)
    tradeoff.set_xticks(axis_x)
    tradeoff.set_xticklabels(["Low Δp", "Power", "Efficiency"])
    tradeoff.set_ylim(min(-0.04, desirable.min() - 0.04), max(1.04, desirable.max() + 0.04))
    tradeoff.set_ylabel("Normalized desirability")
    tradeoff.set_title("Objective balance", loc="left", fontweight="bold")
    _panel_label(tradeoff, "b")

    utility_column = "综合效用" if "综合效用" in result.columns else "ARAS综合效用"
    utility = result[utility_column].to_numpy(float)
    ranks = np.arange(1, len(utility) + 1)
    ranking.plot(ranks, utility, color=PALETTE["blue"], lw=1.4)
    ranking.fill_between(ranks, utility, color=PALETTE["blue_pale"], alpha=0.35)
    ranking.scatter([1], [utility[0]], color=PALETTE["red"], s=30, zorder=3)
    ranking.set_xlabel("Decision rank")
    ranking.set_ylabel(f"{decision_method} utility")
    ranking.set_title("Decision layer", loc="left", fontweight="bold")
    _panel_label(ranking, "c")
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.12, top=0.92)


def draw_validation_figure(
    fig: Figure,
    validation: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    target_names: list[str],
    font_name: str = "Arial",
) -> None:
    apply_publication_style(font_name)
    fig.clear()
    fig.set_size_inches(7.2, 5.4, forward=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.85), hspace=0.58, wspace=0.38)
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["red"]]
    for index, (target, color) in enumerate(zip(target_names, colors)):
        ax = fig.add_subplot(grid[0, index])
        subset = validation[validation["目标"] == target]
        observed = subset["CFD计算值" if "CFD计算值" in subset else "实测值"].to_numpy(float)
        predicted = subset["交叉验证预测值"].to_numpy(float)
        lo = min(observed.min(), predicted.min())
        hi = max(observed.max(), predicted.max())
        margin = max((hi - lo) * 0.05, 1e-12)
        ax.scatter(observed, predicted, s=22, color=color, alpha=0.82, edgecolors="white", linewidths=0.35)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], ls="--", lw=0.9, color=PALETTE["neutral"])
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel("CFD observed")
        ax.set_ylabel("Cross-validated prediction")
        model = str(subset["模型"].iloc[0]) if not subset.empty else ""
        r2 = 1.0 - np.sum((observed - predicted) ** 2) / max(np.sum((observed - observed.mean()) ** 2), 1e-12)
        label = {"dp_Pa": "Pressure drop (Pa)", "P_total_W": "Gross power (W)", "eta_gross_pct": "Fuel efficiency (%)"}.get(target, _publication_label(target))
        ax.set_title(f"{label}\n{model}, $R^2$={r2:.3f}", loc="left", fontweight="bold", fontsize=8)
        _panel_label(ax, chr(ord("a") + index))

    comparison = fig.add_subplot(grid[1, :])
    usable = candidate_metrics[candidate_metrics["状态"] == "可用"].copy()
    models = list(dict.fromkeys(usable["模型"].tolist()))
    x = np.arange(len(models))
    offsets = np.linspace(-0.22, 0.22, len(target_names))
    for offset, target, color in zip(offsets, target_names, colors):
        subset = usable[usable["目标"] == target].set_index("模型")
        values = np.array([subset.loc[model, "归一化RMSE"] if model in subset.index else np.nan for model in models], dtype=float)
        comparison.scatter(x + offset, values, color=color, s=25, label=_publication_label(target), zorder=3)
    comparison.set_xticks(x)
    comparison.set_xticklabels(models, rotation=23, ha="right")
    comparison.set_ylabel("Cross-validated normalized RMSE")
    comparison.set_title("Candidate model comparison (lower is better)", loc="left", fontweight="bold")
    current_top = comparison.get_ylim()[1]
    comparison.set_ylim(0.0, current_top * 1.28)
    comparison.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.63, 0.99))
    _panel_label(comparison, "d")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.84)


def draw_dat_profiles(fig: Figure, data: pd.DataFrame, font_name: str = "Arial") -> None:
    apply_publication_style(font_name)
    fig.clear()
    fig.set_size_inches(7.2, 4.3, forward=True)
    ax = fig.add_subplot(111)
    sources = list(data["源文件"].dropna().unique())[:16]
    colors = mpl.colormaps["viridis"](np.linspace(0.10, 0.88, max(len(sources), 2)))
    plotted = 0
    for color, source in zip(colors, sources):
        subset = data[data["源文件"] == source]
        numeric = [column for column in subset.columns if pd.api.types.is_numeric_dtype(subset[column]) and column not in {"文件名Q", "文件名N"}]
        if len(numeric) < 2:
            continue
        valid = subset[[numeric[0], numeric[1]]].dropna().sort_values(numeric[0])
        ax.plot(valid[numeric[0]], valid[numeric[1]], lw=1.25, color=color, alpha=0.90, label=str(source))
        plotted += 1
    ax.set_title("Imported DAT profiles", loc="left", fontweight="bold")
    ax.set_xlabel("Coordinate / first numeric column")
    ax.set_ylabel("Response / second numeric column")
    if plotted:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6)
    _panel_label(ax, "a")
    fig.subplots_adjust(left=0.10, right=0.76 if plotted else 0.96, bottom=0.13, top=0.92)


def save_figure_bundle(fig: Figure, base_path: str | Path) -> list[Path]:
    base = Path(base_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        (base.with_suffix(".svg"), None),
        (base.with_suffix(".pdf"), None),
        (base.with_suffix(".tiff"), 600),
        (base.with_suffix(".png"), 300),
    ]
    saved: list[Path] = []
    for path, dpi in outputs:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        if path.suffix == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"图表导出失败：{path.name}")
        saved.append(path)
    svg = saved[0].read_text(encoding="utf-8")
    if "<text" not in svg:
        raise RuntimeError("SVG 中没有可编辑文本节点，请检查字体导出设置。")
    return saved
