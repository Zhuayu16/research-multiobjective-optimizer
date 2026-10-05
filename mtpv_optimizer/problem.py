"""Serializable research problems, safe arithmetic constraints and project files."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_VERSION = 1


@dataclass
class Variable:
    column: str
    lower: float
    upper: float
    unit: str = ""
    kind: str = "continuous"
    allowed: list[float] = field(default_factory=list)


@dataclass
class Objective:
    column: str
    direction: str = "min"
    unit: str = ""
    weight: float = 1.0


@dataclass
class Response:
    column: str
    unit: str = ""


@dataclass
class Constraint:
    expression: str
    relation: str = "<="
    limit: float = 0.0
    tolerance: float = 0.0
    scale: float = 1.0


@dataclass
class Problem:
    name: str
    variables: list[Variable]
    objectives: list[Objective]
    constraints: list[Constraint] = field(default_factory=list)
    model: str = "Auto"
    domain: str = "box"
    max_distance: float | None = None
    validation: str = "shuffled"
    folds: int = 5
    group_column: str = ""
    time_column: str = ""
    holdout_column: str = ""
    holdout_value: str = ""
    duplicates: str = "reject"
    seed: int = 42
    population: int = 80
    generations: int = 40
    runs: int = 2
    decision: str = "IDEAL"
    sources: list[str] = field(default_factory=list)
    description: str = ""
    responses: list[Response] = field(default_factory=list)

    @property
    def variable_names(self):
        return [v.column for v in self.variables]

    @property
    def target_names(self):
        return [o.column for o in self.objectives]

    @property
    def response_names(self):
        return self.target_names + [r.column for r in self.responses]

    @property
    def directions(self):
        return [o.direction for o in self.objectives]

    @property
    def weights(self):
        return [o.weight for o in self.objectives]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        for key, constructor in (("variables", Variable), ("objectives", Objective), ("constraints", Constraint), ("responses", Response)):
            raw[key] = [constructor(**item) for item in raw.get(key, [])]
        return cls(**raw)

    def validate(self, frame):
        names = self.variable_names + self.response_names
        if not self.variables or not self.objectives or len(names) != len(set(names)):
            raise ValueError("至少设置一个变量和一个目标，且列名不能重复。")
        reserved = {"_source_row", "_data_role", "_replicate_count", "_validation_group", "推荐排序", "综合效用", "理想点距离", "结果类型", "最近样本距离", "约束违反量", "可行", "搜索域内"}
        if set(frame.columns) & reserved or any(str(c).startswith("约束") and (str(c).endswith("值") or str(c).endswith("违反量")) for c in frame.columns):
            raise ValueError("数据列与软件诊断保留列名重复，请重命名相关列。")
        if not frame.columns.is_unique or any(not isinstance(c, str) for c in frame.columns):
            raise ValueError("数据列名必须是互不重复的文本。")
        missing = set(names) - set(frame.columns)
        if missing:
            raise ValueError("数据缺少列：" + ", ".join(sorted(missing)))
        for v in self.variables:
            if v.kind not in {"continuous", "integer", "discrete"} or not np.isfinite([v.lower, v.upper]).all() or v.upper <= v.lower:
                raise ValueError(f"变量 {v.column} 的类型或边界不正确。")
            if v.kind == "integer" and np.ceil(v.lower) > np.floor(v.upper):
                raise ValueError(f"变量 {v.column} 的范围不包含整数。")
            if v.kind == "discrete":
                if len(set(v.allowed)) < 2 or not np.isfinite(v.allowed).all() or any(a < v.lower or a > v.upper for a in v.allowed):
                    raise ValueError(f"离散变量 {v.column} 至少需要两个范围内的有限允许值。")
        if any(o.direction not in {"min", "max"} or not np.isfinite(o.weight) or o.weight < 0 for o in self.objectives) or sum(self.weights) <= 0:
            raise ValueError("目标方向必须为 min/max，权重非负且总和大于零。")
        if self.validation not in {"shuffled", "group", "time"} or self.duplicates not in {"reject", "keep", "mean"} or self.domain not in {"box", "hull"}:
            raise ValueError("未知的数据处理或搜索域设置。")
        if not 2 <= self.folds <= 20 or self.population < 20 or self.generations < 10 or not 1 <= self.runs <= 20:
            raise ValueError("折数需为 2–20，种群至少 20，迭代至少 10，独立运行 1–20 次。")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1 - 7919 * (self.runs - 1):
            raise ValueError("随机种子超出允许范围。")
        for column in (self.group_column if self.validation == "group" else "", self.time_column if self.validation == "time" else "", self.holdout_column):
            if column and column not in frame:
                raise ValueError(f"缺少验证分组/时间/留出列：{column}")
        if self.validation == "group" and not self.group_column or self.validation == "time" and not self.time_column:
            raise ValueError("请选择分组列或时间列。")
        if self.max_distance is not None and (not np.isfinite(self.max_distance) or self.max_distance <= 0):
            raise ValueError("归一化最近样本距离必须为正值。")
        for c in self.constraints:
            if c.relation not in {"<=", ">=", "=="} or not np.isfinite([c.limit, c.tolerance, c.scale]).all() or c.tolerance < 0 or c.scale <= 0:
                raise ValueError("约束需要 <=、>= 或 ==，容差非负、归一化尺度为正。")
            Arithmetic(c.expression, names)


class Arithmetic:
    """Interpret a small AST; never execute Python supplied in a project file."""
    operators = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}

    def __init__(self, expression, columns):
        self.columns = set(columns)
        # A raw column name is accepted even when it contains spaces or units.
        self.node = ast.Constant(expression) if expression in self.columns else ast.parse(expression, mode="eval").body
        self._visit(self.node, {c: np.ones(1) for c in columns})

    def _visit(self, node, data):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and node.value in data:
                return data[node.value]
            if type(node.value) in {int, float} and np.isfinite(node.value):
                return node.value
        if isinstance(node, ast.Name) and node.id in data:
            return data[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in self.operators:
            return self.operators[type(node.op)](self._visit(node.left, data), self._visit(node.right, data))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._visit(node.operand, data)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords and len(node.args) == 1:
            if node.func.id == "col" and isinstance(node.args[0], ast.Constant) and node.args[0].value in data:
                return data[node.args[0].value]
            if node.func.id == "abs":
                return np.abs(self._visit(node.args[0], data))
        raise ValueError("约束表达式只支持列名、col(\"列名\")、数字、加减乘除、括号和 abs()。")

    def evaluate(self, data):
        with np.errstate(all="ignore"):
            return self._visit(self.node, data)


class ConstraintEvaluator:
    def __init__(self, problem):
        self.problem = problem
        self.expressions = [Arithmetic(c.expression, problem.variable_names + problem.response_names) for c in problem.constraints]

    def details(self, x, y):
        data = dict(zip(self.problem.variable_names + self.problem.response_names, np.column_stack([x, y]).T))
        columns = {}
        for i, (constraint, expression) in enumerate(zip(self.problem.constraints, self.expressions), 1):
            value = np.broadcast_to(expression.evaluate(data), (len(x),))
            if constraint.relation == "<=":
                residual = value - constraint.limit
            elif constraint.relation == ">=":
                residual = constraint.limit - value
            else:
                residual = np.abs(value - constraint.limit)
            violation = np.maximum(0, residual - constraint.tolerance) / constraint.scale
            columns[f"约束{i}值"] = value
            columns[f"约束{i}违反量"] = np.where(np.isfinite(violation), violation, np.inf)
        return pd.DataFrame(columns, index=np.arange(len(x)))

    def __call__(self, x, y):
        detail = self.details(x, y)
        # Search augments penalties in place; pandas CoW may expose a read-only view.
        return detail.filter(like="违反量").sum(axis=1).to_numpy(float, copy=True) if self.expressions else np.zeros(len(x))


def save_project(path, problem, frame):
    problem.validate(frame)
    # Preserve floating-point values, including very small unit-scaled quantities.
    def scalar(value):
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(value).isoformat()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    table = {"columns": list(frame.columns), "data": [[scalar(v) for v in row] for row in frame.itertuples(index=False, name=None)]}
    payload = {"format": "research-optimizer-project", "version": PROJECT_VERSION, "problem": problem.to_dict(), "table": table}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_project(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("format") != "research-optimizer-project" or payload.get("version") != PROJECT_VERSION:
        raise ValueError("不是受支持的研究优化项目文件。")
    table = payload["table"]
    frame = pd.DataFrame(table["data"], columns=table["columns"])
    problem = Problem.from_dict(payload["problem"])
    problem.validate(frame)
    return problem, frame
