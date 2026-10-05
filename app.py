from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mtpv_optimizer.core import (
    fit_surrogates,
    load_table,
    metrics_frame,
    validate_dataset,
)
from mtpv_optimizer.dat_parser import import_dat_files
from mtpv_optimizer.doe import central_composite_design
from mtpv_optimizer.figures import (
    draw_dat_profiles,
    draw_pareto_figure,
    draw_validation_figure,
    save_figure_bundle,
)
from mtpv_optimizer.nsga2 import optimize_nsga2
from mtpv_optimizer.workflow import RunConfig, export_workbook, run_workflow


APP_DIR = Path(__file__).resolve().parent

FIELD_LABELS = {
    "phi": "当量比 φ",
    "flow": "入口质量流量",
    "length": "翅片长度 L",
    "number": "翅片数 N（可选）",
    "pressure": "压降 Δp（最小化）",
    "power": "能量输出（最大化）",
    "efficiency": "系统效率（最大化）",
    "pv": "MTPV 功率 (W)",
    "te": "MTEG 功率 (W)",
    "fraction": "入口 H2 质量分数（可选）",
}

ALIASES = {
    "phi": ["当量比", "phi", "φ", "equivalence ratio", "er"],
    "flow": ["qm_1e5_kg_s", "入口质量流量", "质量流量", "流量", "mass flow", "mass_flow", "m_dot", "q_m", "qm"],
    "length": ["L_mm", "翅片长度", "长度", "l", "length", "fin length"],
    "number": ["翅片数", "n", "fin number", "number of fins"],
    "pressure": ["dp_Pa", "压降", "压力损失", "pressure drop", "delta p", "dp", "Δp"],
    "power": ["P_total_W", "能量输出", "输出功率", "净输出", "pnet", "p_net", "power", "output power", "q_tot", "qtot"],
    "efficiency": ["eta_gross_pct", "系统效率", "净能量效率", "净效率", "mtpv效率", "efficiency", "system efficiency", "eta"],
    "pv": ["P_MTPV_W", "MTPV功率", "MTPV power"],
    "te": ["P_MTEG_W", "MTEG功率", "MTEG power"],
    "fraction": ["Y_H2_in", "H2质量分数", "Y_H2"],
}


class OptimizationWorker(QThread):
    advanced = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, frame, config, parent=None):
        super().__init__(parent)
        self.frame, self.config = frame.copy(deep=True), config

    def run(self):
        try:
            def progress(value, message):
                if self.isInterruptionRequested():
                    raise RuntimeError("计算已取消。")
                self.advanced.emit(value, message)
            self.completed.emit(run_workflow(self.frame, self.config, progress))
        except Exception as exc:
            self.failed.emit(exc)


class OptimizationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MTPV 多目标智能优化：AutoML + NSGA-II")
        self.resize(1450, 900)
        self.frame: pd.DataFrame | None = None
        self.result_frame: pd.DataFrame | None = None
        self.metrics: pd.DataFrame | None = None
        self.current_path: Path | None = None
        self.current_paths: list[Path] = []
        self.dat_summary: pd.DataFrame | None = None
        self.bundle = None
        self.output = None
        self.worker = None
        self.fixed_parameters = {}
        self.mapping_boxes: dict[str, QComboBox] = {}
        self.bound_spins: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        self._build_ui()
        self.statusBar().showMessage("请载入 CFD 结果表，或先生成试验设计模板。")

    def _build_ui(self) -> None:
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#f7f8fa;color:#20242a;}"
            "QGroupBox{font-weight:600;border:1px solid #d9dde3;border-radius:7px;margin-top:9px;padding-top:9px;background:white;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}"
            "QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:white;border:1px solid #cfd5dd;border-radius:4px;padding:4px;}"
            "QPushButton{background:white;border:1px solid #c6ccd4;border-radius:5px;padding:6px 10px;}"
            "QPushButton:hover{border-color:#0F4D92;color:#0F4D92;}"
            "QTabWidget::pane{border:1px solid #d9dde3;background:white;}"
            "QTabBar::tab{background:#e9edf2;padding:7px 13px;margin-right:2px;}"
            "QTabBar::tab:selected{background:white;color:#0F4D92;font-weight:600;}"
        )
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        title = QLabel("MTPV 微燃烧器多目标智能优化")
        title.setFont(QFont("Microsoft YaHei UI", 16, QFont.Bold))
        outer.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setWidget(self._build_controls())
        controls_scroll.setMinimumWidth(controls_scroll.widget().minimumSizeHint().width() + 60)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(self._build_results())
        splitter.setSizes([510, 900])
        self.setStatusBar(QStatusBar())

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.setting_groups = []

        data_group = QGroupBox("1. CFD 数据")
        data_layout = QVBoxLayout(data_group)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Excel / CSV / JSON / DAT")
        browse = QPushButton("导入表格 / DAT")
        browse.clicked.connect(self.load_data)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        data_layout.addLayout(path_row)

        design_row = QHBoxLayout()
        dat_button = QPushButton("批量导入 DAT 曲线")
        dat_button.clicked.connect(self.load_dat_batch)
        design_button = QPushButton("生成 CFD 试验设计模板")
        design_button.clicked.connect(self.create_design)
        design_row.addWidget(dat_button)
        design_row.addWidget(design_button)
        design_row.addStretch(1)
        data_layout.addLayout(design_row)
        paper_button = QPushButton("载入论文 46 组数据")
        paper_button.clicked.connect(self.load_research_data)
        data_layout.addWidget(paper_button)
        self.data_info = QLabel()
        data_layout.addWidget(self.data_info)
        layout.addWidget(data_group)
        self.setting_groups.append(data_group)

        mapping_group = QGroupBox("2. 列映射")
        mapping_layout = QFormLayout(mapping_group)
        self.variable_checks: dict[str, QCheckBox] = {}
        for key, label in FIELD_LABELS.items():
            box = QComboBox()
            box.addItem("— 请选择 —")
            self.mapping_boxes[key] = box
            if key in {"phi", "flow", "length", "number"}:
                row = QHBoxLayout()
                check = QCheckBox("纳入")
                check.setChecked(key != "number")
                check.stateChanged.connect(self._toggle_n)
                self.variable_checks[key] = check
                if key == "number":
                    self.use_n = check
                row.addWidget(box, 1)
                row.addWidget(check)
                mapping_layout.addRow(label, row)
            else:
                mapping_layout.addRow(label, box)
        layout.addWidget(mapping_group)
        self.setting_groups.append(mapping_group)

        physics_group = QGroupBox("3. 计算定义")
        physics = QFormLayout(physics_group)
        self.workflow_mode = QComboBox()
        self.workflow_mode.addItem("直接响应：压降/功率/效率分别拟合", "generic")
        self.workflow_mode.addItem("H2/air：MTPV + MTEG 总功率与效率", "coupled")
        self.flow_unit = QComboBox()
        self.flow_unit.addItem("质量流量数值 × 10^-5 kg/s", 1e-5)
        self.flow_unit.addItem("质量流量数值 kg/s", 1.0)
        self.lhv = self._double_spin(119.96, 0.1)
        self.lhv.setRange(1, 1000)
        self.air_fuel = self._double_spin(34.32, 0.01)
        self.air_fuel.setRange(1, 1000)
        self.hull_check = QCheckBox("样本凸包")
        self.hull_check.setChecked(True)
        self.endpoint_check = QCheckBox("差分进化端点核对")
        self.endpoint_check.setChecked(True)
        self.holdout_check = QCheckBox("3.3_confirmation")
        physics.addRow("模式", self.workflow_mode)
        physics.addRow("混合物质量流量单位", self.flow_unit)
        physics.addRow("H2 低位热值 (MJ/kg)", self.lhv)
        physics.addRow("化学计量空燃质量比", self.air_fuel)
        physics.addRow("搜索域", self.hull_check)
        physics.addRow("端点核对", self.endpoint_check)
        physics.addRow("响应留出", self.holdout_check)
        self.workflow_mode.currentIndexChanged.connect(self._toggle_mode)
        layout.addWidget(physics_group)
        self.setting_groups.append(physics_group)

        bounds_group = QGroupBox("4. 优化边界")
        bounds_layout = QGridLayout(bounds_group)
        bounds_layout.addWidget(QLabel("变量"), 0, 0)
        bounds_layout.addWidget(QLabel("下限"), 0, 1)
        bounds_layout.addWidget(QLabel("上限"), 0, 2)
        default_bounds = {
            "phi": (0.7, 1.1, 0.01),
            "flow": (3.5, 5.5, 0.01),
            "length": (15.0, 35.0, 0.1),
            "number": (2.0, 8.0, 1.0),
        }
        for row, key in enumerate(("phi", "flow", "length", "number"), start=1):
            lower, upper, step = default_bounds[key]
            low_spin = self._double_spin(lower, step)
            high_spin = self._double_spin(upper, step)
            self.bound_spins[key] = (low_spin, high_spin)
            bounds_layout.addWidget(QLabel(FIELD_LABELS[key]), row, 0)
            bounds_layout.addWidget(low_spin, row, 1)
            bounds_layout.addWidget(high_spin, row, 2)
        layout.addWidget(bounds_group)
        self.setting_groups.append(bounds_group)

        settings_group = QGroupBox("5. 代理模型、NSGA-II 与决策权重")
        settings = QGridLayout(settings_group)
        self.population = QSpinBox()
        self.population.setRange(20, 1000)
        self.population.setValue(150)
        self.generations = QSpinBox()
        self.generations.setRange(10, 2000)
        self.generations.setValue(100)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(20260929)
        settings.addWidget(QLabel("种群规模"), 0, 0)
        settings.addWidget(self.population, 0, 1)
        settings.addWidget(QLabel("迭代次数"), 1, 0)
        settings.addWidget(self.generations, 1, 1)
        settings.addWidget(QLabel("随机种子"), 2, 0)
        settings.addWidget(self.seed, 2, 1)

        self.weight_pressure = self._weight_spin(1.0)
        self.weight_power = self._weight_spin(1.0)
        self.weight_efficiency = self._weight_spin(1.0)
        settings.addWidget(QLabel("压降权重"), 0, 2)
        settings.addWidget(self.weight_pressure, 0, 3)
        settings.addWidget(QLabel("能量输出权重"), 1, 2)
        settings.addWidget(self.weight_power, 1, 3)
        settings.addWidget(QLabel("系统效率权重"), 2, 2)
        settings.addWidget(self.weight_efficiency, 2, 3)

        self.model_mode = QComboBox()
        self.model_mode.addItems(
            [
                "Auto",
                "RSM-Quadratic",
                "MLP-NeuralNet",
                "GaussianProcess",
                "SVR-RBF",
                "RandomForest",
                "ExtraTrees",
                "GradientBoosting",
                "Linear",
            ]
        )
        self.independent_runs = QSpinBox()
        self.independent_runs.setRange(1, 10)
        self.independent_runs.setValue(3)
        self.figure_font = QComboBox()
        self.figure_font.addItems(["Arial", "Times New Roman"])
        self.decision_method = QComboBox()
        self.decision_method.addItems(["IDEAL", "TOPSIS", "ARAS"])
        settings.addWidget(QLabel("代理模型"), 3, 0)
        settings.addWidget(self.model_mode, 3, 1)
        settings.addWidget(QLabel("独立运行次数"), 3, 2)
        settings.addWidget(self.independent_runs, 3, 3)
        settings.addWidget(QLabel("图表字体"), 4, 0)
        settings.addWidget(self.figure_font, 4, 1)
        settings.addWidget(QLabel("折中决策"), 4, 2)
        settings.addWidget(self.decision_method, 4, 3)
        layout.addWidget(settings_group)
        self.setting_groups.append(settings_group)

        anchor_group = QGroupBox("6. IDEAL 固定归一化尺度")
        anchor_layout = QGridLayout(anchor_group)
        self.fixed_anchors = QCheckBox("使用指定尺度")
        self.anchor_spins = []
        anchor_layout.addWidget(self.fixed_anchors, 0, 0, 1, 3)
        anchor_layout.addWidget(QLabel("最小值"), 1, 1)
        anchor_layout.addWidget(QLabel("最大值"), 1, 2)
        for row, (name, low, high) in enumerate((("压降", 227.020954, 663.392046), ("总功率", 6.3122919, 12.8958395), ("效率", 5.92596128, 8.10363089)), 2):
            spins = (self._double_spin(low, 0.1), self._double_spin(high, 0.1))
            for spin in spins:
                spin.setDecimals(8)
            self.anchor_spins.append(spins)
            anchor_layout.addWidget(QLabel(name), row, 0)
            for col, spin in enumerate(spins, 1):
                anchor_layout.addWidget(spin, row, col)
                spin.setEnabled(False)
        self.fixed_anchors.toggled.connect(lambda enabled: [spin.setEnabled(enabled) for pair in self.anchor_spins for spin in pair])
        layout.addWidget(anchor_group)
        self.setting_groups.append(anchor_group)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("建立模型并优化")
        self.run_button.setStyleSheet("QPushButton{background:#0F4D92;color:white;font-weight:bold;padding:9px;border-radius:4px}")
        self.run_button.clicked.connect(self.run_optimization)
        self.export_button = QPushButton("导出结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_optimization)
        button_row.addWidget(self.run_button, 1)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.export_button)
        layout.addLayout(button_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        layout.addStretch(1)
        self._toggle_n()
        self._toggle_mode()
        for box in panel.findChildren(QComboBox):
            box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            box.setMinimumWidth(80)
        for spin in panel.findChildren(QSpinBox) + panel.findChildren(QDoubleSpinBox):
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            spin.setMinimumWidth(86)
        settings.setColumnStretch(1, 1)
        settings.setColumnStretch(3, 1)
        self.seed.setMinimumWidth(125)
        for pair in self.anchor_spins:
            for spin in pair:
                spin.setMinimumWidth(125)
        return panel

    def _build_results(self) -> QWidget:
        tabs = QTabWidget()
        self.result_tabs = tabs
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs.addTab(self.preview_table, "数据预览")

        dat_page = QWidget()
        dat_layout = QVBoxLayout(dat_page)
        self.dat_figure = Figure(figsize=(7.2, 4.3))
        self.dat_canvas = FigureCanvas(self.dat_figure)
        self.dat_table = QTableWidget()
        self.dat_table.setEditTriggers(QTableWidget.NoEditTriggers)
        dat_layout.addWidget(self.dat_canvas, 3)
        dat_layout.addWidget(self.dat_table, 2)
        tabs.addTab(dat_page, "DAT 曲线与摘要")

        model_page = QWidget()
        model_layout = QVBoxLayout(model_page)
        self.model_text = QTextEdit()
        self.model_text.setReadOnly(True)
        self.model_text.setMaximumHeight(150)
        self.model_text.setPlaceholderText("运行后显示交叉验证精度和数据警告。")
        model_layout.addWidget(self.model_text)
        self.validation_figure = Figure(figsize=(7.2, 5.4))
        self.validation_canvas = FigureCanvas(self.validation_figure)
        model_layout.addWidget(self.validation_canvas, 3)
        self.candidate_table = QTableWidget()
        self.candidate_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.candidate_table.setMaximumHeight(210)
        model_layout.addWidget(self.candidate_table, 1)
        tabs.addTab(model_page, "模型检验")

        pareto_page = QWidget()
        pareto_layout = QVBoxLayout(pareto_page)
        self.figure = Figure(figsize=(7.2, 4.8))
        self.canvas = FigureCanvas(self.figure)
        pareto_layout.addWidget(self.canvas, 3)
        self.result_table = QTableWidget()
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        pareto_layout.addWidget(self.result_table, 2)
        tabs.addTab(pareto_page, "Pareto 解")
        self.observed_table = QTableWidget()
        self.observed_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs.addTab(self.observed_table, "已计算方案")
        self.sensitivity_table = QTableWidget()
        self.sensitivity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs.addTab(self.sensitivity_table, "偏好对比")
        self.stability_table = QTableWidget()
        self.stability_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs.addTab(self.stability_table, "独立运行")
        self.holdout_table = QTableWidget()
        self.holdout_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tabs.addTab(self.holdout_table, "响应留出")
        return tabs

    @staticmethod
    def _double_spin(value: float, step: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setDecimals(6)
        widget.setRange(-1e12, 1e12)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    @staticmethod
    def _weight_spin(value: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setDecimals(2)
        widget.setRange(0.0, 100.0)
        widget.setSingleStep(0.1)
        widget.setValue(value)
        return widget

    def _toggle_n(self) -> None:
        for key, check in self.variable_checks.items():
            enabled = check.isChecked()
            self.mapping_boxes[key].setEnabled(enabled)
            for spin in self.bound_spins[key]:
                spin.setEnabled(enabled)

    def _toggle_mode(self):
        coupled = self.workflow_mode.currentData() == "coupled"
        for key in ("pv", "te", "fraction"):
            self.mapping_boxes[key].setEnabled(coupled)
        for key in ("power", "efficiency"):
            self.mapping_boxes[key].setEnabled(not coupled)
        for widget in (self.flow_unit, self.lhv, self.air_fuel, self.holdout_check):
            widget.setEnabled(coupled)

    def _clear_results(self):
        self.output = self.bundle = self.result_frame = self.metrics = None
        self.export_button.setEnabled(False)
        for table in (self.result_table, self.observed_table, self.sensitivity_table, self.stability_table, self.holdout_table, self.candidate_table):
            table.setRowCount(0)
        self.figure.clear()
        self.validation_figure.clear()
        self.canvas.draw_idle()
        self.validation_canvas.draw_idle()
        self.model_text.clear()

    def load_research_data(self):
        try:
            path = APP_DIR / "data" / "research_46_20261002.json"
            if not path.is_file():
                candidate = APP_DIR / "share" / "research-optimizer" / path.name
                path = candidate if candidate.is_file() else Path(sys.prefix) / "share" / "research-optimizer" / path.name
            if not path.is_file():
                raise FileNotFoundError("公开仓库不包含私有 46 组研究数据。请导入自己的数据表，或在本机 data 目录放入已获授权的数据快照。")
            self._accept_frame(load_table(path), path)
            self.workflow_mode.setCurrentIndex(1)
            self.population.setValue(90)
            self.generations.setValue(65)
            self.independent_runs.setValue(2)
            self.seed.setValue(20260929)
            self.model_mode.setCurrentText("Auto")
            self.decision_method.setCurrentText("IDEAL")
            self.flow_unit.setCurrentIndex(0)
            self.lhv.setValue(119.96)
            self.air_fuel.setValue(34.32)
            self.fixed_anchors.setChecked(True)
            for pair, values in zip(self.anchor_spins, ((227.020954, 663.392046), (6.3122919, 12.8958395), (5.92596128, 8.10363089))):
                pair[0].setValue(values[0])
                pair[1].setValue(values[1])
            self.hull_check.setChecked(True)
            self.endpoint_check.setChecked(True)
            self.holdout_check.setChecked(True)
            for key in ("phi", "flow", "length"):
                self.variable_checks[key].setChecked(True)
            self.use_n.setChecked(False)
            self.fixed_parameters = {"N": 6}
            self.data_info.setText("46 组 CFD，N = 6")
        except Exception as exc:
            self._show_error("论文数据载入失败", exc)

    def _accept_frame(self, frame, path):
        self._clear_results()
        self.frame = frame
        self.current_path = Path(path)
        self.current_paths = [self.current_path]
        self.dat_summary = None
        self.fixed_parameters = {}
        self.data_info.setText(f"{len(frame)} 行，{len(frame.columns)} 列")
        self.dat_figure.clear()
        self.dat_canvas.draw_idle()
        self.dat_table.setRowCount(0)
        self.path_edit.setText(str(path))
        self._populate_mapping()
        self._show_frame(self.preview_table, frame.head(100))
        self.result_tabs.setCurrentIndex(0)
        self.statusBar().showMessage(f"已载入 {len(frame)} 组数据。")

    def _selected_column(self, key: str) -> str:
        value = self.mapping_boxes[key].currentText()
        if value.startswith("—"):
            raise ValueError(f"请选择“{FIELD_LABELS[key]}”对应的数据列。")
        return value

    def load_data(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 CFD 结果表",
            str(self.current_path.parent if self.current_path else APP_DIR),
            "数据文件 (*.xlsx *.xls *.xlsm *.csv *.tsv *.json *.dat *.dat.h5)",
        )
        if not filename:
            return
        try:
            if filename.lower().endswith(".dat"):
                imported = import_dat_files([filename])
                self._accept_frame(imported.data, filename)
                self.dat_summary = imported.summary
                self._show_dat_result()
            elif Path(filename).suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
                excel = pd.ExcelFile(filename)
                sheet_names = excel.sheet_names.copy()
                excel.close()
                default_index = next(
                    (index for index, name in enumerate(sheet_names) if "总体性能" in name or "优化" in name),
                    0,
                )
                sheet, accepted = QInputDialog.getItem(
                    self,
                    "选择工作表",
                    "请选择包含决策变量和目标值的工作表：",
                    sheet_names,
                    default_index,
                    False,
                )
                if not accepted:
                    return
                self._accept_frame(pd.read_excel(filename, sheet_name=sheet), filename)
            else:
                self._accept_frame(load_table(filename), filename)
        except Exception as exc:
            self._show_error("载入失败", exc)

    def load_dat_batch(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "选择一个或多个 DAT 文本文件",
            str(self.current_path.parent if self.current_path else APP_DIR),
            "DAT 文本 (*.dat);;Fluent 数据 (*.dat.h5)",
        )
        if not filenames:
            return
        try:
            imported = import_dat_files(filenames)
            self._accept_frame(imported.data, imported.files[0])
            self.dat_summary = imported.summary
            self.current_paths = imported.files
            self.current_path = imported.files[0]
            self.path_edit.setText(f"已导入 {len(imported.files)} 个 DAT 文件：{self.current_path.parent}")
            self._populate_mapping()
            self._show_frame(self.preview_table, self.frame.head(200))
            self._show_dat_result()
            self.result_tabs.setCurrentIndex(1)
            self.statusBar().showMessage(f"已解析 {len(imported.files)} 个 DAT 文件，共 {len(self.frame)} 个数据点。")
        except Exception as exc:
            self._show_error("DAT 导入失败", exc)

    def _show_dat_result(self) -> None:
        if self.frame is None or self.dat_summary is None:
            return
        self._show_frame(self.dat_table, self.dat_summary)
        draw_dat_profiles(self.dat_figure, self.frame, self.figure_font.currentText())
        self.dat_canvas.draw_idle()

    def _populate_mapping(self) -> None:
        assert self.frame is not None
        columns = [str(column) for column in self.frame.columns]
        normalize = lambda value: value.strip().lower().replace("_", " ")
        normalized = {column: normalize(column) for column in columns}
        for key, box in self.mapping_boxes.items():
            box.blockSignals(True)
            box.clear()
            box.addItem("— 请选择 —")
            box.addItems(columns)
            matches = [column for column, cleaned in normalized.items() if any(normalize(alias) == cleaned for alias in ALIASES[key])]
            for column, cleaned in normalized.items():
                if any(
                    len(alias.strip()) > 2 and normalize(alias) in cleaned
                    for alias in ALIASES[key]
                ):
                    if column not in matches:
                        matches.append(column)
            if matches:
                box.setCurrentText(matches[0])
            box.blockSignals(False)
        self._toggle_n()
        self._update_bounds_from_data()

    def _update_bounds_from_data(self) -> None:
        if self.frame is None:
            return
        for key in ("phi", "flow", "length", "number"):
            box = self.mapping_boxes[key]
            if box.currentIndex() <= 0:
                continue
            numeric = pd.to_numeric(self.frame[box.currentText()], errors="coerce").dropna()
            if not numeric.empty:
                low, high = self.bound_spins[key]
                low.setValue(float(numeric.min()))
                high.setValue(float(numeric.max()))

    def _variable_configuration(self) -> tuple[list[str], np.ndarray, np.ndarray, tuple[int, ...], list[str]]:
        keys = [key for key in ("phi", "flow", "length", "number") if self.variable_checks[key].isChecked()]
        if len(keys) < 2:
            raise ValueError("至少选择两个决策变量。")
        names = [self._selected_column(key) for key in keys]
        lower = np.array([self.bound_spins[key][0].value() for key in keys], dtype=float)
        upper = np.array([self.bound_spins[key][1].value() for key in keys], dtype=float)
        if self.frame is not None:
            # Recover exact sample endpoints when the displayed decimals round them.
            for i, key in enumerate(keys):
                data = pd.to_numeric(self.frame[names[i]], errors="coerce")
                data = data[np.isfinite(data)]
                if data.empty:
                    continue
                for values, spin, bound in ((lower, self.bound_spins[key][0], data.min()),
                                            (upper, self.bound_spins[key][1], data.max())):
                    if abs(values[i] - bound) <= 0.51 * 10 ** (-spin.decimals()):
                        values[i] = bound
        if np.any(upper <= lower):
            raise ValueError("所有变量的上限必须大于下限。")
        integer_indices = (keys.index("number"),) if "number" in keys else ()
        return names, lower, upper, integer_indices, keys

    def create_design(self) -> None:
        try:
            keys = [key for key in ("phi", "flow", "length", "number") if self.variable_checks[key].isChecked()]
            if len(keys) < 2:
                raise ValueError("至少选择两个决策变量后再生成试验设计。")
            default_names = {"phi": "当量比", "flow": "入口质量流量", "length": "翅片长度L", "number": "翅片数N"}
            names = [default_names[key] for key in keys]
            lower = np.array([self.bound_spins[key][0].value() for key in keys])
            upper = np.array([self.bound_spins[key][1].value() for key in keys])
            integer_indices = (keys.index("number"),) if "number" in keys else ()
            design = central_composite_design(names, lower, upper, integer_indices)
            coupled = self.workflow_mode.currentData() == "coupled"
            for column in (("dp_Pa", "P_MTPV_W", "P_MTEG_W") if coupled else ("压降", "能量输出", "系统效率")):
                design[column] = np.nan
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "保存试验设计模板",
                str(APP_DIR / "CFD试验设计模板.xlsx"),
                "Excel 工作簿 (*.xlsx)",
            )
            if filename:
                if not filename.lower().endswith(".xlsx"):
                    filename += ".xlsx"
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    design.to_excel(writer, index=False, sheet_name="试验设计")
                    pd.DataFrame(
                        {
                            "说明": [
                                "上下限仅为 GUI 当前设置，请按实际研究范围确认后再运行 CFD。",
                                "H2/air 模式填写 dp_Pa、P_MTPV_W、P_MTEG_W；总功率和燃料效率由程序统一计算。" if coupled else "每一行完成 CFD 后填写压降、能量输出和系统效率。",
                                f"混合物质量流量单位倍率：{self.flow_unit.currentData()} kg/s；H2低位热值：{self.lhv.value()} MJ/kg。" if coupled else "入口质量流量和效率必须带一致单位。",
                                "N 为离散整数变量；若不研究 N，请保持关闭。",
                            ]
                        }
                    ).to_excel(writer, index=False, sheet_name="说明")
                self.statusBar().showMessage(f"已生成 {len(design)} 组试验设计：{filename}")
        except Exception as exc:
            self._show_error("生成失败", exc)

    def run_optimization(self) -> None:
        if self.frame is None:
            QMessageBox.information(self, "尚未载入数据", "请先载入已填写 CFD 结果的 Excel 或 CSV 文件。")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            variables, lower, upper, integer_indices, _ = self._variable_configuration()
            coupled = self.workflow_mode.currentData() == "coupled"
            targets = [self._selected_column("pressure")]
            if not coupled:
                targets += [self._selected_column("power"), self._selected_column("efficiency")]
            config = RunConfig(
                variables=variables, targets=targets, lower=lower.tolist(), upper=upper.tolist(),
                integer_indices=integer_indices, mode=self.workflow_mode.currentData(),
                phi_column=self.mapping_boxes["phi"].currentText(), flow_column=self.mapping_boxes["flow"].currentText(),
                pv_column=self._selected_column("pv") if coupled else "",
                te_column=self._selected_column("te") if coupled else "",
                fraction_column=self.mapping_boxes["fraction"].currentText() if coupled and self.mapping_boxes["fraction"].currentIndex() > 0 else None,
                flow_scale=self.flow_unit.currentData(), lhv_J_kg=self.lhv.value() * 1e6,
                stoichiometric_air_fuel_ratio=self.air_fuel.value(), use_hull=self.hull_check.isChecked(),
                model_mode=self.model_mode.currentText(), seed=self.seed.value(),
                population_size=self.population.value(),
                generations=self.generations.value(),
                weights=(self.weight_pressure.value(), self.weight_power.value(), self.weight_efficiency.value()),
                runs=self.independent_runs.value(),
                decision_method=self.decision_method.currentText(),
                anchor_minimum=[pair[0].value() for pair in self.anchor_spins] if self.fixed_anchors.isChecked() else None,
                anchor_maximum=[pair[1].value() for pair in self.anchor_spins] if self.fixed_anchors.isChecked() else None,
                check_endpoints=self.endpoint_check.isChecked(),
                holdout_batch="3.3_confirmation" if coupled and self.holdout_check.isChecked() else None,
                sources=[str(path) for path in self.current_paths],
                fixed_parameters=self.fixed_parameters.copy(),
            )
            self._clear_results()
            self.run_font = self.figure_font.currentText()
            self._set_running(True)
            self.progress.setValue(1)
            self.worker = OptimizationWorker(self.frame, config, self)
            self.worker.advanced.connect(self._worker_progress)
            self.worker.completed.connect(self._receive_output)
            self.worker.failed.connect(self._worker_failed)
            self.worker.finished.connect(lambda: self._set_running(False))
            self.worker.start()
        except Exception as exc:
            self.progress.setValue(0)
            self._show_error("优化失败", exc)

    def _set_running(self, running):
        for group in self.setting_groups:
            group.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.export_button.setEnabled(not running and self.output is not None)

    def _worker_progress(self, value, message):
        self.progress.setValue(value)
        self.statusBar().showMessage(message)

    def cancel_optimization(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("正在结束当前计算步骤……")

    def _worker_failed(self, exc):
        self._clear_results()
        self.progress.setValue(0)
        if str(exc) == "计算已取消。":
            self.statusBar().showMessage(str(exc))
        else:
            self._show_error("优化失败", exc)

    def _receive_output(self, output):
        try:
            self.output = output
            self.bundle = output.bundle
            self.clean_frame = output.clean
            self.result_frame = output.predicted
            self.metrics = metrics_frame(output.bundle)
            output.config["figure_font"] = self.run_font
            self._show_metrics(output.warnings, len(output.clean))
            for table, frame in ((self.candidate_table, output.bundle.candidate_metrics), (self.result_table, output.predicted),
                                 (self.observed_table, output.observed), (self.sensitivity_table, output.sensitivity),
                                 (self.stability_table, output.run_summary), (self.holdout_table, getattr(output.bundle, "holdout_frame", pd.DataFrame()))):
                self._show_frame(table, frame)
            self._plot_pareto(output.config["variables"], output.config["target_names"])
            draw_validation_figure(self.validation_figure, output.bundle.validation_frame, output.bundle.candidate_metrics,
                                   output.config["target_names"], self.run_font)
            self.validation_canvas.draw_idle()
            self.result_tabs.setCurrentIndex(4)
            first = output.observed.iloc[0]
            self.statusBar().showMessage(f"完成：{len(output.predicted)} 个模型候选，{len(output.observed)} 个已计算非支配方案；计算方案首选 {first.get('case', '第1行')}。")
        except Exception as exc:
            self._worker_failed(exc)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.cancel_optimization()
            event.ignore()
            return
        super().closeEvent(event)

    def _show_metrics(self, warnings: list[str], sample_count: int) -> None:
        assert self.metrics is not None
        folds = self.output.config.get("cv_folds", 5 if self.output.config["mode"] == "coupled" else min(5, max(2, sample_count // 4)))
        lines = [f"有效样本：{sample_count} 组", "", f"{folds} 折交叉验证："]
        for _, row in self.metrics.iterrows():
            lines.append(
                f"• {row['目标']} → {row['选用模型']}：CV R²={row['交叉验证R2']:.4f}，"
                f"NRMSE={row['归一化RMSE']:.4f}，RMSE={row['交叉验证RMSE']:.6g}，"
                f"MAE={row['交叉验证MAE']:.6g}，"
                f"MAPE={row['MAPE(%)']:.3f}%，最大相对误差={row['最大相对误差(%)']:.3f}%"
            )
        weak = self.metrics[self.metrics["交叉验证R2"] < 0.8]
        if not weak.empty:
            warnings.append("存在交叉验证 R² < 0.8 的目标；Pareto 结果仅供补充算例选点，不宜直接作为最终结论。")
        if warnings:
            lines.extend(["", "数据提示："] + [f"• {item}" for item in warnings])
        self.model_text.setPlainText("\n".join(lines))

    def _plot_pareto(self, variable_names: list[str], target_names: list[str]) -> None:
        assert self.result_frame is not None
        draw_pareto_figure(
            self.figure,
            self.result_frame,
            variable_names,
            target_names,
            self.run_font,
            self.output.config["decision_method"],
            observed=self.output.observed,
            anchors=(np.array(self.output.config["anchor_minimum"]), np.array(self.output.config["anchor_maximum"])),
        )
        self.canvas.draw_idle()

    def export_results(self) -> None:
        if self.result_frame is None or self.metrics is None:
            QMessageBox.information(self, "没有结果", "请先完成一次优化。")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出 Pareto 结果",
            str(APP_DIR / "MTPV_Pareto优化结果.xlsx"),
            "Excel 工作簿 (*.xlsx)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".xlsx"):
            filename += ".xlsx"
        try:
            export_workbook(self.output, filename)
            output_base = Path(filename).with_suffix("")
            pareto_files = save_figure_bundle(self.figure, output_base.parent / f"{output_base.name}_Pareto")
            validation_files = save_figure_bundle(self.validation_figure, output_base.parent / f"{output_base.name}_模型验证")
            self.statusBar().showMessage(f"已导出工作簿及 {len(pareto_files) + len(validation_files)} 个 Nature 图表文件。")
            QMessageBox.information(
                self,
                "导出完成",
                f"结果工作簿：{filename}\n\n图表已同时导出为 SVG、PDF、600 dpi TIFF 和 PNG。",
            )
        except Exception as exc:
            self._show_error("导出失败", exc)

    @staticmethod
    def _show_frame(table: QTableWidget, frame: pd.DataFrame) -> None:
        table.clear()
        table.setRowCount(len(frame))
        table.setColumnCount(len(frame.columns))
        table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        for row in range(len(frame)):
            for column, name in enumerate(frame.columns):
                value = frame.iloc[row, column]
                if isinstance(value, (float, np.floating)):
                    text = f"{value:.8g}"
                else:
                    text = str(value)
                table.setItem(row, column, QTableWidgetItem(text))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

    def _show_error(self, title: str, exc: Exception) -> None:
        self.statusBar().showMessage(str(exc))
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        box = QMessageBox(QMessageBox.Critical, title, str(exc), parent=self)
        box.setDetailedText(details)
        box.exec_()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    from mtpv_optimizer.general_ui import GeneralWindow
    window = GeneralWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
