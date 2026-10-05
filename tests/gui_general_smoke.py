"""Exercise the actual Qt controls, workers, restore, charts and exported artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton

from mtpv_optimizer.general_ui import GeneralWindow
from mtpv_optimizer.problem import Problem, Variable, Objective, Constraint, load_project


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "outputs" / "generalization_validation_20261005"
    folder.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion"); app.setFont(QFont("Microsoft YaHei UI", 9))
    window = GeneralWindow(); window.show()
    errors = []
    window.show_error = lambda exc: errors.append(str(exc))
    heartbeats = 0
    def click(text):
        button = next(b for b in window.findChildren(QPushButton) if b.text() == text)
        QTest.mouseClick(button, Qt.LeftButton); QTest.qWait(50)
    def run():
        nonlocal heartbeats
        window.population.setValue(32); window.generations.setValue(15); window.runs.setValue(1)
        QTest.mouseClick(window.run_button, Qt.LeftButton)
        assert window.worker is not None, errors
        started = time.monotonic()
        while window.worker.isRunning():
            QTest.qWait(30); heartbeats += 1
            assert time.monotonic() - started < 120
        QTest.qWait(200)
        assert not errors, errors
        assert window.output is not None and window.export_button.isEnabled()
        assert (window.output.predicted["约束违反量"] <= 1e-10).all()
    QTest.qWait(150)
    click("载入合成示例")
    assert len(window.frame) == 60
    assert window.get_problem().directions == ["min"] * 3
    window.grab().save(str(folder / "battery_loaded.png"))
    run()
    assert (window.output.predicted.Tmax_K <= 315 + 1e-10).all()
    np.testing.assert_allclose(window.output.predicted.channels, np.rint(window.output.predicted.channels))
    project_path = folder / "battery_saved.optproj"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(project_path), "")):
        click("保存项目")
    saved_problem, saved_table = load_project(project_path)
    assert saved_problem.to_dict() == window.get_problem().to_dict()
    pd.testing.assert_frame_equal(saved_table, window.frame)
    window.objective_table.cellWidget(0, 3).setValue(9)
    assert window.dirty
    exported = folder / "battery.xlsx"
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(exported), "")):
        QTest.mouseClick(window.export_button, Qt.LeftButton)
    assert not errors, errors
    p, _ = load_project(exported.with_suffix(".optproj"))
    assert p.weights == [1, 1, 1], "Export must retain the completed run settings"
    assert json.loads(exported.with_suffix(".json").read_text(encoding="utf-8"))["software_version"] == "0.2.0"
    assert all((folder / f"battery_{name}.{suffix}").stat().st_size > 0 for name in ("Pareto", "验证") for suffix in ("svg", "pdf", "tiff", "png"))
    window.result_tabs.setCurrentIndex(7); QTest.qWait(150); window.grab().save(str(folder / "gui_battery_pareto.png"))
    window.result_tabs.setCurrentIndex(8); QTest.qWait(150); window.grab().save(str(folder / "battery_validation.png"))
    # A measured response can constrain the search without becoming an objective.
    direction = window.objective_table.cellWidget(1, 2)
    direction.setCurrentIndex(direction.findData("response"))
    window.add_constraint(Constraint("deltaT_K", "<=", 6.5))
    run()
    assert len(window.output.bundle.target_names) == 2 and len(window.output.bundle.metrics) == 3
    assert (window.output.predicted.deltaT_K <= 6.5 + 1e-9).all()
    window.preset_box.setCurrentIndex(window.preset_box.findData("exchanger")); click("载入合成示例")
    assert window.output is None and not window.export_button.isEnabled()
    # Verify retained centre replicates through the actual design export control.
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(folder / "exchanger_DOE.xlsx"), "")):
        click("导出 LHS 试验设计（保留3个中心重复）")
    design = pd.read_excel(folder / "exchanger_DOE.xlsx", sheet_name="试验设计")
    assert len(design) == 33 and design.design_role.eq("center_replicate").sum() == 3
    assert design.pitch_mm.isin([2, 4, 6]).all()
    run()
    assert len(window.output.bundle.target_names) == 2
    assert window.output.predicted.pitch_mm.isin([2, 4, 6]).all()
    window.result_tabs.setCurrentIndex(7); QTest.qWait(100); window.grab().save(str(folder / "gui_exchanger_pareto.png"))
    window.export_to(folder / "exchanger.xlsx")
    window.preset_box.setCurrentIndex(window.preset_box.findData("structure")); click("载入合成示例")
    run()
    assert len(window.output.bundle.target_names) == 4
    assert (window.output.predicted.safety_margin >= -1e-10).all()
    window.axes_boxes[0].setCurrentIndex(2); window.axes_boxes[1].setCurrentIndex(3); window.axes_boxes[2].setCurrentIndex(0)
    window.result_tabs.setCurrentIndex(7); QTest.qWait(100); window.grab().save(str(folder / "gui_structure_pareto.png"))
    window.result_tabs.setCurrentIndex(8); QTest.qWait(100); window.grab().save(str(folder / "structure_validation.png"))
    window.export_to(folder / "structure.xlsx")
    # Restore the full project using the real Open button.
    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(project_path), "")):
        click("打开项目")
    assert window.output is None and window.get_problem().to_dict() == saved_problem.to_dict()
    assert not errors, errors
    window.validation_box.setCurrentIndex(window.validation_box.findData("group"))
    window.group_box.setCurrentText("batch")
    window.holdout_box.setCurrentText("batch"); window.holdout_value.setText("B")
    run()
    assert window.output.config["holdout_count"] == 10
    assert set(window.output.bundle.holdout_frame["训练组数"]) == {50}
    assert window.output.config["cv"] == "按指定分组验证"
    window.validation_box.setCurrentIndex(window.validation_box.findData("time")); window.time_box.setCurrentText("time")
    run()
    assert window.output.config["cv"].startswith("按时间块前向")
    assert window.output.bundle.validation_frame.loc[window.output.bundle.validation_frame.fold.eq(0), "交叉验证预测值"].isna().all()
    # Failed import preserves the current project.
    prior = window.frame.copy()
    try:
        window.accept_frame(pd.DataFrame({"bad": ["text"]}))
        raise AssertionError("Invalid import was accepted")
    except ValueError:
        pass
    pd.testing.assert_frame_equal(window.frame, prior)
    # Single-objective charts and a signed response through the same UI.
    x = np.linspace(-1, 1, 25)
    p = Problem("single signed", [Variable("x", -1, 1)], [Objective("response", "max")], model="Linear", population=24, generations=12, runs=1)
    window.accept_frame(pd.DataFrame({"x": x, "response": x}), p)
    run()
    assert abs(window.output.predicted.iloc[0].x - 1) < 1e-10
    window.result_tabs.setCurrentIndex(7); QTest.qWait(100); window.grab().save(str(folder / "single_pareto.png"))
    # Backward compatibility entry works even without the private dataset.
    click("MTPV/MTEG 物理耦合")
    assert window.mtpv_window.isVisible()
    if (root / "data" / "research_46_20261002.json").is_file():
        window.mtpv_window.load_research_data()
        assert len(window.mtpv_window.frame) == 46
    else:
        assert window.mtpv_window.frame is None
    window.mtpv_window.close()
    # Cancellation clears results and keeps the thread alive until it finishes.
    window.model_box.setCurrentIndex(window.model_box.findData("GaussianProcess"))
    QTest.mouseClick(window.run_button, Qt.LeftButton)
    window.cancel(); window.close()
    assert window.isVisible(), "Closing while worker is active must be deferred"
    started = time.monotonic()
    while window.worker.isRunning():
        QTest.qWait(30); assert time.monotonic() - started < 120
    QTest.qWait(100)
    assert window.output is None and window.run_button.isEnabled() and not errors, errors
    window.close(); app.processEvents()
    report = {"status": "passed", "event_loop_heartbeats": heartbeats, "checks": ["editable variable/objective/constraint tables", "three synthetic domains", "integer and finite-set variables", "1/2/3/4 objectives", "constraint-only response", "safe engineering constraints", "group/time validation and independent holdout controls", "DOE centre replicates", "project save/restore", "immutable run export", "eight figure formats", "axis selectors", "legacy MTPV entry", "cancellation and close protection"]}
    (folder / "gui_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
