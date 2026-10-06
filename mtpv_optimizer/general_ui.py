"""Qt editor for reusable numeric research problems."""
from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QLineEdit, QDoubleSpinBox,
    QSpinBox, QPushButton, QLabel, QTextEdit, QProgressBar, QFileDialog, QMessageBox, QCheckBox, QScrollArea)

from .core import load_table, metrics_frame, _candidate_models
from .figures import save_figure_bundle
from .general import run_general, design_template
from .general_figures import draw_general_pareto, draw_general_validation
from .presets import PRESETS, preset, example
from .problem import Problem, Variable, Objective, Response, Constraint, load_project, save_project
from .workflow import export_workbook


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT if (ROOT / "run_gui.bat").is_file() else Path.home() / "Documents"


class CompactNumber(QDoubleSpinBox):
    def textFromValue(self, value):
        return format(value, ".12g")


class GeneralWorker(QThread):
    advanced = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, frame, problem, parent=None):
        super().__init__(parent)
        self.frame = frame.copy(deep=True)
        self.problem = Problem.from_dict(problem.to_dict())

    def run(self):
        try:
            def progress(value, message):
                if self.isInterruptionRequested():
                    raise RuntimeError("计算已取消。")
                self.advanced.emit(value, message)
            self.completed.emit(run_general(self.frame, self.problem, progress))
        except Exception as exc:
            self.failed.emit(exc)


class GeneralWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工程多目标优化 · 通用研究项目")
        self.resize(1500, 920)
        self.frame, self.output, self.worker = None, None, None
        self.sources, self.description = [], ""
        self._loading, self.dirty = True, False
        self._build()
        self.set_problem(preset("battery"))
        self.statusBar().showMessage("导入研究结果表，配置变量、目标和约束；示例为合成数据。")

    @staticmethod
    def number(value=0, integer=False, minimum=-1e12, maximum=1e12):
        widget = QSpinBox() if integer else CompactNumber()
        widget.setRange(int(minimum) if integer else minimum, int(maximum) if integer else maximum)
        if not integer:
            widget.setDecimals(15)
        widget.setValue(value)
        return widget

    @staticmethod
    def choice(items, value=None, editable=False):
        box = QComboBox(); box.setEditable(editable)
        for text, data in items:
            box.addItem(text, data)
        if value is not None:
            index = box.findData(value)
            if index >= 0:
                box.setCurrentIndex(index)
            elif editable:
                box.setEditText(str(value))
        return box

    def _build(self):
        central = QWidget(); outer = QVBoxLayout(central); self.setCentralWidget(central)
        toolbar = QHBoxLayout()
        self.action_buttons = []
        for text, handler in (("导入数据表", self.import_data), ("打开项目", self.open_project_dialog),
                              ("保存项目", self.save_project_dialog), ("MTPV/MTEG 物理耦合", self.open_mtpv)):
            button = QPushButton(text); button.clicked.connect(handler); toolbar.addWidget(button); self.action_buttons.append(button)
        toolbar.addStretch(1)
        outer.addLayout(toolbar)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("领域模板"))
        self.preset_box = self.choice([(name, key) for key, name in PRESETS.items()])
        preset_row.addWidget(self.preset_box)
        for text, handler in (("应用模板", self.apply_preset), ("载入合成示例", self.load_example)):
            button = QPushButton(text); button.clicked.connect(handler); preset_row.addWidget(button); self.action_buttons.append(button)
        preset_row.addStretch(1)
        outer.addLayout(preset_row)
        self.data_label = QLabel("尚未导入数据"); self.data_label.setWordWrap(True); outer.addWidget(self.data_label)
        split = QSplitter(Qt.Horizontal); outer.addWidget(split, 1)
        self.editor = QTabWidget(); self.editor.setMinimumWidth(420); split.addWidget(self.editor)
        self.variable_table = self._editor_table(["数据列", "单位", "类型", "下限", "上限", "允许值（逗号分隔）"], "变量", self.add_variable)
        self.objective_table = self._editor_table(["数据列", "单位", "目标方向或约束响应", "权重"], "目标/响应", self.add_objective)
        self.constraint_table = self._editor_table(["列名或算术表达式", "关系", "限值", "容差", "归一化尺度"], "约束", self.add_constraint)
        constraint_help = QLabel('表达式可用：Tmax_K、abs(x-y)、col("带单位的列名")。可在目标/响应页把指标设为“仅用于约束”。容差与表达式同单位；尺度用于比较违反量。')
        constraint_help.setWordWrap(True); self.editor.widget(2).layout().addWidget(constraint_help)
        validation = QWidget(); form = QFormLayout(validation)
        self.domain_box = self.choice([("训练数据边界框", "box"), ("训练样本凸包（≤6维）", "hull")])
        self.distance_check = QCheckBox("限制归一化最近样本距离")
        self.distance = self.number(.35, minimum=1e-8, maximum=100)
        distance_row = QHBoxLayout(); distance_row.addWidget(self.distance_check); distance_row.addWidget(self.distance)
        self.validation_box = self.choice([("随机 K 折", "shuffled"), ("按分组 K 折", "group"), ("按时间块前向验证", "time")])
        self.folds = self.number(5, True, 2, 20)
        self.group_box, self.time_box, self.holdout_box = [self.column_box("") for _ in range(3)]
        self.holdout_value = QLineEdit()
        self.duplicates_box = self.choice([("冲突响应报错，相同记录去重", "reject"), ("保留重复测量", "keep"), ("同一设计取响应均值", "mean")])
        for text, control in (("搜索域", self.domain_box), ("验证方式", self.validation_box), ("验证折数", self.folds),
             ("验证分组列", self.group_box), ("时间列", self.time_box), ("独立留出列（可选）", self.holdout_box),
             ("留出值（精确匹配文本）", self.holdout_value), ("重复设计处理", self.duplicates_box)):
            form.addRow(text, control)
        form.addRow(distance_row)
        help_label = QLabel("独立留出数据不参与模型选型、拟合和搜索域建立。随机验证遇到重复测量时自动按设计分组。距离表示样本覆盖程度。")
        help_label.setWordWrap(True); form.addRow(help_label)
        self.editor.addTab(validation, "数据与验证")
        settings = QWidget(); form = QFormLayout(settings)
        self.name_edit = QLineEdit()
        self.model_box = self.choice([(n, n) for n in ["Auto"] + list(_candidate_models(42))])
        self.population = self.number(80, True, 20, 2000)
        self.generations = self.number(40, True, 10, 2000)
        self.runs = self.number(2, True, 1, 20)
        self.seed = self.number(42, minimum=0, maximum=2**32 - 1)
        self.seed.setDecimals(0)
        self.decision_box = self.choice([(n, n) for n in ("IDEAL", "IDEAL-COEFFICIENT", "TOPSIS", "ARAS")])
        for name, control in (("项目名称", self.name_edit), ("代理模型", self.model_box), ("种群规模", self.population),
             ("迭代次数", self.generations), ("独立运行次数", self.runs), ("随机种子", self.seed), ("推荐排序方法", self.decision_box)):
            form.addRow(name, control)
        self.description_edit = QTextEdit(); self.description_edit.setPlaceholderText("研究对象、数据来源或演示说明")
        form.addRow("说明", self.description_edit)
        doe_button = QPushButton("导出 LHS 试验设计（保留3个中心重复）"); doe_button.clicked.connect(self.export_design); form.addRow(doe_button)
        self.editor.addTab(settings, "运行与设计")
        self.result_tabs = QTabWidget(); split.addWidget(self.result_tabs); split.setSizes([720, 780])
        self.preview = self._result_table("输入数据")
        self.log = QTextEdit(); self.log.setReadOnly(True); self.result_tabs.addTab(self.log, "验证摘要")
        self.predicted_table = self._result_table("预测 Pareto")
        self.observed_table = self._result_table("输入表 Pareto")
        self.diagnostic_table = self._result_table("约束与覆盖")
        self.holdout_table = self._result_table("独立留出")
        self.sensitivity_table = self._result_table("权重对比")
        self.pareto_figure, self.validation_figure = Figure(), Figure()
        self.pareto_canvas, self.validation_canvas = FigureCanvasQTAgg(self.pareto_figure), FigureCanvasQTAgg(self.validation_figure)
        figure_panel = QWidget(); figure_layout = QVBoxLayout(figure_panel)
        axes_row = QHBoxLayout(); self.axes_boxes = [QComboBox() for _ in range(3)]
        for text, box in zip(("横轴", "纵轴", "颜色"), self.axes_boxes):
            axes_row.addWidget(QLabel(text)); axes_row.addWidget(box); box.currentIndexChanged.connect(self.redraw_pareto)
        figure_layout.addLayout(axes_row); figure_layout.addWidget(self.pareto_canvas)
        self.result_tabs.addTab(figure_panel, "目标图")
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.validation_canvas); self.result_tabs.addTab(scroll, "验证图")
        row = QHBoxLayout()
        self.run_button = QPushButton("开始优化"); self.run_button.clicked.connect(self.run_optimization)
        self.cancel_button = QPushButton("取消"); self.cancel_button.clicked.connect(self.cancel); self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("导出本次结果与图表"); self.export_button.clicked.connect(self.export_dialog); self.export_button.setEnabled(False)
        self.progress = QProgressBar()
        for w in (self.run_button, self.cancel_button, self.export_button, self.progress): row.addWidget(w)
        outer.addLayout(row)
        for box in self.findChildren(QComboBox):
            if box not in self.axes_boxes and box is not self.preset_box:
                box.currentTextChanged.connect(self.mark_dirty)
        for spin in self.findChildren(QDoubleSpinBox) + self.findChildren(QSpinBox): spin.valueChanged.connect(self.mark_dirty)
        for edit in self.findChildren(QLineEdit): edit.textChanged.connect(self.mark_dirty)
        self.description_edit.textChanged.connect(self.mark_dirty)
        self.distance_check.toggled.connect(self.mark_dirty)

    def _editor_table(self, headers, name, add):
        page = QWidget(); layout = QVBoxLayout(page)
        table = QTableWidget(0, len(headers)); table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        widths = {6: [145, 65, 90, 90, 90, 150], 4: [155, 65, 190, 90], 5: [220, 75, 100, 90, 110]}[len(headers)]
        for i, width in enumerate(widths): table.setColumnWidth(i, width)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(table)
        buttons = QHBoxLayout()
        plus, minus = QPushButton("添加" + name), QPushButton("删除所选行")
        plus.clicked.connect(lambda: add()); minus.clicked.connect(lambda: self.remove_row(table))
        buttons.addWidget(plus); buttons.addWidget(minus); layout.addLayout(buttons)
        self.editor.addTab(page, name)
        return table

    def _result_table(self, name):
        table = QTableWidget(); table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_tabs.addTab(table, name)
        return table

    def column_box(self, value):
        columns = [] if self.frame is None else list(self.frame.columns)
        return self.choice([(c, c) for c in [""] + columns], value, editable=True)

    def _cells(self, table, widgets):
        row = table.rowCount(); table.insertRow(row)
        for i, widget in enumerate(widgets):
            table.setCellWidget(row, i, widget)
            if isinstance(widget, QComboBox): widget.currentTextChanged.connect(self.mark_dirty)
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.valueChanged.connect(lambda _, w=widget: w.setProperty("source_value", None))
                widget.valueChanged.connect(self.mark_dirty)
            elif isinstance(widget, QLineEdit): widget.textChanged.connect(self.mark_dirty)
        self.mark_dirty()

    def add_variable(self, variable=None):
        v = variable or Variable("", 0, 1)
        column, lo, hi = self.column_box(v.column), self.number(v.lower), self.number(v.upper)
        self._cells(self.variable_table, [column, QLineEdit(v.unit), self.choice([("连续", "continuous"), ("整数", "integer"), ("数值集合", "discrete")], v.kind), lo, hi, QLineEdit(",".join(map(str, v.allowed)))])
        lo.setProperty("source_value", v.lower); hi.setProperty("source_value", v.upper)
        def update_bounds(text):
            if not self._loading and self.frame is not None and text in self.frame:
                values = pd.to_numeric(self.frame[text], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not values.empty:
                    lo.setValue(float(values.min())); hi.setValue(float(values.max()))
                    lo.setProperty("source_value", float(values.min())); hi.setProperty("source_value", float(values.max()))
        column.currentTextChanged.connect(update_bounds)

    def add_objective(self, objective=None):
        o = objective or Objective("")
        direction = self.choice([("最小化 ↓", "min"), ("最大化 ↑", "max"), ("仅用于约束", "response")], o.direction)
        weight = self.number(o.weight, minimum=0, maximum=1e9)
        weight.setEnabled(o.direction != "response")
        direction.currentIndexChanged.connect(lambda: weight.setEnabled(direction.currentData() != "response"))
        self._cells(self.objective_table, [self.column_box(o.column), QLineEdit(o.unit), direction, weight])

    def add_constraint(self, constraint=None):
        c = constraint or Constraint("")
        self._cells(self.constraint_table, [QLineEdit(c.expression), self.choice([(n, n) for n in ("<=", ">=", "==")], c.relation), self.number(c.limit), self.number(c.tolerance, minimum=0), self.number(c.scale, minimum=1e-10)])

    def remove_row(self, table):
        for row in sorted({item.row() for item in table.selectedIndexes()}, reverse=True): table.removeRow(row)
        self.mark_dirty()

    def get_problem(self):
        variables, objectives, constraints, responses = [], [], [], []
        for r in range(self.variable_table.rowCount()):
            w = [self.variable_table.cellWidget(r, c) for c in range(6)]
            allowed = [float(v.strip()) for v in w[5].text().replace("，", ",").split(",") if v.strip()]
            bounds = [spin.value() if spin.property("source_value") is None else spin.property("source_value") for spin in w[3:5]]
            variables.append(Variable(w[0].currentText().strip(), *bounds, w[1].text().strip(), w[2].currentData(), allowed))
        for r in range(self.objective_table.rowCount()):
            w = [self.objective_table.cellWidget(r, c) for c in range(4)]
            if w[2].currentData() == "response":
                responses.append(Response(w[0].currentText().strip(), w[1].text().strip()))
            else:
                objectives.append(Objective(w[0].currentText().strip(), w[2].currentData(), w[1].text().strip(), w[3].value()))
        for r in range(self.constraint_table.rowCount()):
            w = [self.constraint_table.cellWidget(r, c) for c in range(5)]
            constraints.append(Constraint(w[0].text().strip(), w[1].currentData(), w[2].value(), w[3].value(), w[4].value()))
        return Problem(self.name_edit.text().strip() or "研究项目", variables, objectives, constraints,
            model=self.model_box.currentData(), domain=self.domain_box.currentData(), max_distance=self.distance.value() if self.distance_check.isChecked() else None,
            validation=self.validation_box.currentData(), folds=self.folds.value(), group_column=self.group_box.currentText().strip(),
            time_column=self.time_box.currentText().strip(), holdout_column=self.holdout_box.currentText().strip(), holdout_value=self.holdout_value.text(),
            duplicates=self.duplicates_box.currentData(), seed=int(self.seed.value()), population=self.population.value(), generations=self.generations.value(),
            runs=self.runs.value(), decision=self.decision_box.currentData(), sources=list(self.sources), description=self.description_edit.toPlainText(), responses=responses)

    def set_problem(self, problem):
        self._loading = True
        for table in (self.variable_table, self.objective_table, self.constraint_table): table.setRowCount(0)
        for v in problem.variables: self.add_variable(v)
        for o in problem.objectives: self.add_objective(o)
        for r in problem.responses: self.add_objective(Objective(r.column, "response", r.unit, 0))
        for c in problem.constraints: self.add_constraint(c)
        for box, value in ((self.model_box, problem.model), (self.domain_box, problem.domain), (self.validation_box, problem.validation),
             (self.duplicates_box, problem.duplicates), (self.decision_box, problem.decision)):
            box.setCurrentIndex(box.findData(value))
        for box, value in ((self.group_box, problem.group_column), (self.time_box, problem.time_column), (self.holdout_box, problem.holdout_column)):
            box.clear(); box.addItems([""] + ([] if self.frame is None else list(self.frame.columns))); box.setCurrentText(value)
        for widget, value in ((self.folds, problem.folds), (self.population, problem.population), (self.generations, problem.generations), (self.runs, problem.runs), (self.seed, problem.seed)):
            widget.setValue(value)
        self.name_edit.setText(problem.name); self.holdout_value.setText(problem.holdout_value)
        self.distance_check.setChecked(problem.max_distance is not None); self.distance.setValue(problem.max_distance or .35)
        self.description_edit.setPlainText(problem.description); self.sources = list(problem.sources)
        self._loading = False; self.dirty = False

    def mark_dirty(self, *_):
        if not self._loading:
            self.dirty = True
            if self.output is not None:
                self.statusBar().showMessage("设置已更改；当前图表与导出仍对应上次运行，重新优化后更新。")

    def accept_frame(self, frame, problem=None, source=None):
        if frame.empty:
            raise ValueError("数据表为空。")
        frame = frame.copy(); frame.columns = [str(c) for c in frame.columns]
        if not frame.columns.is_unique:
            raise ValueError("数据列名重复。")
        if problem is None:
            try:
                current = self.get_problem()
            except (TypeError, ValueError):
                current = None
            if current is not None and set(current.variable_names + current.response_names).issubset(frame.columns):
                problem = current
                for v in problem.variables:
                    values = pd.to_numeric(frame[v.column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if not values.empty: v.lower, v.upper = float(values.min()), float(values.max())
            else:
                numeric = [c for c in frame if pd.to_numeric(frame[c], errors="coerce").notna().sum() >= 10 and c.lower() not in {"case", "time", "batch", "id"}]
                if len(numeric) < 2:
                    raise ValueError("至少需要两列数值数据（变量及响应），请检查表头和数据类型。")
                n = 1 if len(numeric) == 2 else min(2, len(numeric) - 1)
                problem = Problem("自定义研究项目", [Variable(c, float(pd.to_numeric(frame[c], errors="coerce").min()), float(pd.to_numeric(frame[c], errors="coerce").max())) for c in numeric[:n]],
                                  [Objective(c) for c in numeric[n:]], description="自动初选数值列；请确认哪些是变量、哪些是响应以及目标方向。")
        if source:
            problem.sources = [str(source)]
            if "合成" in problem.description:
                problem.description = f"导入数据：{Path(source).name}。请确认列定义、单位和目标方向。"
        self.frame = frame; self.clear_results(); self.show_table(self.preview, frame.head(200))
        self.set_problem(problem)
        self.data_label.setText(f"{len(frame)} 行 / {len(frame.columns)} 列（预览前200行）。{problem.description}")
        self.result_tabs.setCurrentIndex(0)

    def apply_preset(self):
        problem = preset(self.preset_box.currentData()); problem.sources = list(self.sources)
        self.set_problem(problem); self.mark_dirty()
        self.statusBar().showMessage("模板已应用；请确认数据列、单位、边界和约束适合当前研究对象。")

    def load_example(self):
        try:
            p, frame = example(self.preset_box.currentData()); self.accept_frame(frame, p)
        except Exception as exc: self.show_error(exc)

    def import_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入研究结果表", str(DEFAULT_DIR), "数据表 (*.xlsx *.xlsm *.xls *.csv *.tsv *.json *.dat);;所有文件 (*)")
        if not path: return
        try:
            if Path(path).suffix.lower() in {".xlsx", ".xlsm"}:
                from PyQt5.QtWidgets import QInputDialog
                with pd.ExcelFile(path) as book: sheets = book.sheet_names
                sheet, ok = QInputDialog.getItem(self, "工作表", "选择包含变量和响应的数据表", sheets, 0, False) if len(sheets) > 1 else (sheets[0], True)
                if not ok: return
                frame = pd.read_excel(path, sheet_name=sheet)
            else: frame = load_table(path)
            self.accept_frame(frame, source=path)
        except Exception as exc: self.show_error(exc)

    def save_project_dialog(self):
        try:
            if self.frame is None: raise ValueError("请先导入数据。")
            problem = self.get_problem(); problem.validate(self.frame)
            path, _ = QFileDialog.getSaveFileName(self, "保存完整项目（数据及配置）", str(DEFAULT_DIR / "research_project.optproj"), "研究项目 (*.optproj)")
            if path:
                if not path.lower().endswith(".optproj"): path += ".optproj"
                save_project(path, problem, self.frame); self.statusBar().showMessage(f"项目已保存：{path}")
        except Exception as exc: self.show_error(exc)

    def open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开研究项目", str(DEFAULT_DIR), "研究项目 (*.optproj)")
        if not path: return
        try:
            p, frame = load_project(path); self.accept_frame(frame, p); self.statusBar().showMessage("已恢复项目数据及全部配置；可重新运行。")
        except Exception as exc: self.show_error(exc)

    def export_design(self):
        try:
            p = self.get_problem(); frame = design_template(p)
            path, _ = QFileDialog.getSaveFileName(self, "导出试验设计", str(DEFAULT_DIR / "research_DOE.xlsx"), "Excel (*.xlsx)")
            if path:
                if not path.lower().endswith(".xlsx"): path += ".xlsx"
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    frame.to_excel(writer, sheet_name="试验设计", index=False)
                    pd.DataFrame({"说明": ["LHS 30 行 + 3 个保留的中心重复测量；填写响应后导入。", "仅生成变量设计，不保证未知响应约束可行；不做有限集合的去重。"]}).to_excel(writer, sheet_name="说明", index=False)
                self.statusBar().showMessage(f"试验设计已导出：{path}")
        except Exception as exc: self.show_error(exc)

    def open_mtpv(self):
        from app import OptimizationWindow
        if not hasattr(self, "mtpv_window"):
            self.mtpv_window = OptimizationWindow()
        self.mtpv_window.show(); self.mtpv_window.raise_()

    def clear_results(self):
        self.output = None; self.export_button.setEnabled(False); self.progress.setValue(0); self.log.clear()
        for table in (self.predicted_table, self.observed_table, self.diagnostic_table, self.holdout_table, self.sensitivity_table):
            table.setRowCount(0); table.setColumnCount(0)
        for figure, canvas in ((self.pareto_figure, self.pareto_canvas), (self.validation_figure, self.validation_canvas)):
            figure.clear(); canvas.draw_idle()

    def run_optimization(self):
        try:
            if self.worker is not None and self.worker.isRunning(): return
            if self.frame is None: raise ValueError("请先导入数据或载入合成示例。")
            problem = self.get_problem(); problem.validate(self.frame)
            self.clear_results(); self.set_running(True); self.dirty = False
            self.worker = GeneralWorker(self.frame, problem, self)
            self.worker.advanced.connect(self.on_progress); self.worker.completed.connect(self.receive_output)
            self.worker.failed.connect(self.on_failed); self.worker.finished.connect(lambda: self.set_running(False))
            self.worker.start()
        except Exception as exc: self.show_error(exc)

    def set_running(self, running):
        self.editor.setEnabled(not running); self.preset_box.setEnabled(not running)
        for button in self.action_buttons: button.setEnabled(not running)
        self.run_button.setEnabled(not running); self.cancel_button.setEnabled(running)
        self.export_button.setEnabled(not running and self.output is not None)

    def on_progress(self, value, message):
        self.progress.setValue(value); self.statusBar().showMessage(message)

    def cancel(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption(); self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("正在结束当前计算步骤……")

    def on_failed(self, exc):
        self.clear_results()
        if str(exc) == "计算已取消。": self.statusBar().showMessage(str(exc))
        else: self.show_error(exc)

    def receive_output(self, output):
        try:
            self.output = output
            self.result_problem = Problem.from_dict(output.config["problem"])
            p = self.result_problem
            for table, frame in ((self.predicted_table, output.predicted), (self.observed_table, output.observed),
                (self.diagnostic_table, output.constraint_audit), (self.holdout_table, output.bundle.holdout_frame), (self.sensitivity_table, output.sensitivity)):
                self.show_table(table, frame)
            lines = [f"训练 {output.config['sample_count']} 行；独立留出 {output.config['holdout_count']} 行。", f"{output.config['cv']}，实际 {output.config['cv_folds']} 折。", ""]
            for m in output.bundle.metrics:
                lines.append(f"{m.target} → {m.model_name}：R²={m.r2_cv:.4f}，NRMSE={m.nrmse_cv:.4g}，RMSE={m.rmse_cv:.6g}，MAE={m.mae_cv:.6g}")
            lines.extend(["", *output.warnings])
            self.log.setPlainText("\n".join(lines))
            for box in self.axes_boxes: box.blockSignals(True); box.clear()
            for box in self.axes_boxes[:2]:
                for j, objective in enumerate(p.objectives): box.addItem(objective.column, j)
            self.axes_boxes[1].setCurrentIndex(min(1, len(p.objectives) - 1))
            self.axes_boxes[2].addItem("无", None)
            for j, objective in enumerate(p.objectives): self.axes_boxes[2].addItem(objective.column, j)
            if len(p.objectives) > 2: self.axes_boxes[2].setCurrentIndex(3)
            for box in self.axes_boxes: box.blockSignals(False)
            self.redraw_pareto()
            draw_general_validation(self.validation_figure, output, p)
            self.validation_canvas.setMinimumHeight(round(self.validation_figure.get_size_inches()[1] * 100))
            self._fit_canvas(self.validation_canvas)
            self.result_tabs.setCurrentIndex(2)
            self.statusBar().showMessage(f"完成：{len(output.predicted)} 个预测候选，{len(output.observed)} 个输入表可行 Pareto 方案。")
        except Exception as exc: self.on_failed(exc)

    def redraw_pareto(self, *_):
        if self.output is not None:
            axes = tuple(box.currentData() for box in self.axes_boxes)
            draw_general_pareto(self.pareto_figure, self.output, self.result_problem, axes)
            self._fit_canvas(self.pareto_canvas)

    @staticmethod
    def _fit_canvas(canvas):
        # Sync the Matplotlib drawing size after choosing a new multi-panel layout.
        from matplotlib.backend_bases import ResizeEvent
        dpi = canvas.figure.dpi
        canvas.figure.set_size_inches(canvas.width() / dpi, canvas.height() / dpi, forward=False)
        ResizeEvent("resize_event", canvas)._process()
        canvas.draw_idle()

    def export_to(self, path):
        if self.output is None: raise ValueError("请先完成优化。")
        path = Path(path)
        export_workbook(self.output, path)
        save_figure_bundle(self.pareto_figure, path.with_suffix("").with_name(path.stem + "_Pareto"))
        save_figure_bundle(self.validation_figure, path.with_suffix("").with_name(path.stem + "_验证"))
        save_project(path.with_suffix(".optproj"), self.result_problem, self.frame)

    def export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出本次结果与图表", str(DEFAULT_DIR / "research_result.xlsx"), "Excel (*.xlsx)")
        if not path: return
        try:
            if not path.lower().endswith(".xlsx"): path += ".xlsx"
            self.export_to(path); self.statusBar().showMessage("已导出工作簿、运行设置、可恢复项目和 SVG/PDF/TIFF/PNG 图表。")
        except Exception as exc: self.show_error(exc)

    @staticmethod
    def show_table(table, frame):
        table.clear(); table.setRowCount(len(frame)); table.setColumnCount(len(frame.columns))
        table.setHorizontalHeaderLabels([str(c) for c in frame.columns])
        for row in range(len(frame)):
            for column in range(len(frame.columns)):
                value = frame.iloc[row, column]
                text = f"{value:.8g}" if isinstance(value, (float, np.floating)) else str(value)
                table.setItem(row, column, QTableWidgetItem(text))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def show_error(self, exc):
        self.statusBar().showMessage(str(exc))
        box = QMessageBox(QMessageBox.Critical, "无法完成操作", str(exc), parent=self)
        box.setDetailedText("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))); box.exec_()

    def closeEvent(self, event):
        running = self.worker is not None and self.worker.isRunning()
        if hasattr(self, "mtpv_window"):
            running |= self.mtpv_window.worker is not None and self.mtpv_window.worker.isRunning()
        if running:
            self.cancel()
            if hasattr(self, "mtpv_window"): self.mtpv_window.cancel_optimization()
            event.ignore(); return
        if hasattr(self, "mtpv_window"): self.mtpv_window.close()
        super().closeEvent(event)
