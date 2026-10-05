"""Illustrative domain presets. Synthetic examples are not physical validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .problem import Constraint, Objective, Problem, Variable


PRESETS = {"battery": "电池液冷", "exchanger": "换热器", "structure": "结构设计", "mtpv": "MTPV/MTEG（直接响应）"}


def preset(key):
    if key == "battery":
        return Problem("电池液冷", [Variable("flow_L_min", 0.5, 3, "L/min"), Variable("channel_mm", 2, 8, "mm"), Variable("channels", 2, 6, "", "integer")],
            [Objective("Tmax_K", "min", "K"), Objective("deltaT_K", "min", "K"), Objective("pump_W", "min", "W")],
            [Constraint("Tmax_K", "<=", 315, scale=10)], model="RSM-Quadratic")
    if key == "exchanger":
        return Problem("换热器", [Variable("velocity_m_s", 1, 5, "m/s"), Variable("pitch_mm", 2, 6, "mm", "discrete", [2, 4, 6])],
            [Objective("heat_W", "max", "W"), Objective("dp_Pa", "min", "Pa")], [Constraint("dp_Pa", "<=", 350, scale=100)], model="RSM-Quadratic")
    if key == "structure":
        return Problem("结构设计", [Variable("thickness_mm", 2, 8, "mm"), Variable("width_mm", 10, 40, "mm")],
            [Objective("mass_kg", "min", "kg"), Objective("deflection_mm", "min", "mm"), Objective("safety_margin", "max"), Objective("cost", "min")],
            [Constraint("safety_margin", ">=", 0, scale=1)], model="GaussianProcess")
    if key == "mtpv":
        return Problem("MTPV/MTEG 直接响应", [Variable("phi", 0.7, 1.1), Variable("qm_1e5_kg_s", 3.54, 5.43, "10^-5 kg/s"), Variable("L_mm", 15, 35, "mm")],
            [Objective("dp_Pa", "min", "Pa"), Objective("P_total_W", "max", "W"), Objective("eta_gross_pct", "max", "%")], domain="hull")
    raise ValueError("未知领域模板。")


def example(key, seed=73):
    if key == "mtpv":
        raise ValueError("MTPV 模板请导入原计算表；物理耦合模式在原 MTPV 界面。")
    problem = preset(key)
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(v.lower, v.upper, 60) for v in problem.variables])
    # Include range corners to keep the configured bounds inside the sample ranges.
    x[:2] = [[v.lower for v in problem.variables], [v.upper for v in problem.variables]]
    for j, variable in enumerate(problem.variables):
        if variable.kind == "integer":
            x[:, j] = np.rint(x[:, j])
        elif variable.kind == "discrete":
            choices = np.asarray(variable.allowed)
            x[:, j] = choices[np.argmin(abs(x[:, j, None] - choices), axis=1)]
    if key == "battery":
        flow, channel, count = x.T
        y = np.column_stack([320 - 5 * flow - .45 * channel - .5 * count + .2 * flow**2,
                             8 - .7 * flow - .3 * channel - .25 * count + .04 * channel**2,
                             .5 + 1.2 * flow**2 + .15 * count + .05 * channel])
    elif key == "exchanger":
        velocity, pitch = x.T
        y = np.column_stack([200 + 80 * velocity - 15 * pitch + 2 * velocity * pitch,
                             20 + 15 * velocity**2 + 4 * pitch])
    else:
        thickness, width = x.T
        y = np.column_stack([.002 * thickness * width, 30 / (thickness**2 * width),
                             (thickness * width - 100) / 100, 10 + .08 * thickness * width + .1 * width**2])
    table = pd.DataFrame(np.column_stack([x, y]), columns=problem.variable_names + problem.target_names)
    table.insert(0, "case", [f"SYNTHETIC_{i:03d}" for i in range(1, len(table) + 1)])
    table["batch"] = np.repeat(["A", "B", "C", "D", "E", "F"], 10)
    table["time"] = np.arange(len(table))
    problem.description = "合成函数演示数据，只用于软件功能测试，不代表真实试验或仿真结果。"
    return problem, table
