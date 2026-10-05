"""Exercise the real Qt window, background worker, dialogs and exports."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton, QScrollArea

from app import OptimizationWindow


def main():
    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "research_46_20261002.json").is_file():
        print(json.dumps({"status": "skipped", "reason": "private 46-case dataset is not included in the public repository"}))
        return
    output_dir = root / "outputs" / "generalization_validation_20261005" / "legacy"
    output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = OptimizationWindow()
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    window.show()
    QTest.qWait(150)
    controls = window.findChild(QScrollArea)
    assert controls.widget().width() <= controls.viewport().width(), "Controls clipped at default width"
    button = next(b for b in window.findChildren(QPushButton) if b.text() == "载入论文 46 组数据")
    QTest.mouseClick(button, Qt.LeftButton)
    assert not errors, errors
    assert len(window.frame) == 46
    assert window.mapping_boxes["phi"].currentText() == "phi"
    assert window.mapping_boxes["flow"].currentText() == "qm_1e5_kg_s"
    assert window.mapping_boxes["length"].currentText() == "L_mm"
    assert window.mapping_boxes["pv"].currentText() == "P_MTPV_W"
    assert window.mapping_boxes["te"].currentText() == "P_MTEG_W"
    assert window.seed.value() == 20260929
    window.grab().save(str(output_dir / "gui_loaded.png"))
    QTest.mouseClick(window.run_button, Qt.LeftButton)
    heartbeat = 0
    started = time.monotonic()
    while window.worker is not None and window.worker.isRunning():
        QTest.qWait(30)
        heartbeat += 1
        assert time.monotonic() - started < 180, "Worker timed out"
    QTest.qWait(200)
    assert not errors, errors
    assert window.output is not None
    assert window.export_button.isEnabled()
    assert window.output.observed.iloc[0]["case"] == "OPT01"
    assert window.output.bundle.domain.contains(window.output.predicted[window.output.config["variables"]].to_numpy()).all()
    assert len(window.output.bundle.holdout_frame) == 12
    assert np.allclose(window.output.predicted["P_total_W"], window.output.predicted["P_MTPV_W"] + window.output.predicted["P_MTEG_W"])
    for tab in (2, 3, 4, 5, 6, 7):
        window.result_tabs.setCurrentIndex(tab)
        QTest.qWait(80)
        window.grab().save(str(output_dir / f"gui_tab_{tab}.png"))
    window.resize(1100, 800)
    window.result_tabs.setCurrentIndex(3)
    QTest.qWait(150)
    assert controls.widget().width() <= controls.viewport().width(), "Controls clipped at compact width"
    window.grab().save(str(output_dir / "gui_compact.png"))
    filename = output_dir / "research_validation.xlsx"
    saved_weights = window.output.config["weights"].copy() if isinstance(window.output.config["weights"], list) else window.output.config["weights"]
    window.weight_power.setValue(9)
    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(filename), "")), patch.object(QMessageBox, "information", return_value=QMessageBox.Ok):
        QTest.mouseClick(window.export_button, Qt.LeftButton)
    assert not errors, errors
    saved = json.loads(filename.with_suffix(".json").read_text(encoding="utf-8"))
    assert tuple(saved["weights"]) == tuple(saved_weights), "Export used edited UI settings"
    assert all((output_dir / f"research_validation_{name}.{extension}").stat().st_size > 0
               for name in ("Pareto", "模型验证") for extension in ("svg", "pdf", "tiff", "png"))
    window.load_research_data()
    assert window.output is None and not window.export_button.isEnabled()
    # Exercise cooperative cancellation and protect the window from closing a live QThread.
    window.model_mode.setCurrentText("GaussianProcess")
    QTest.mouseClick(window.run_button, Qt.LeftButton)
    window.cancel_optimization()
    while window.worker.isRunning():
        QTest.qWait(30)
        assert time.monotonic() - started < 180
    QTest.qWait(100)
    assert window.output is None and window.run_button.isEnabled()
    assert not errors, errors
    window.close()
    app.processEvents()
    print(json.dumps({"status": "passed", "event_loop_heartbeats": heartbeat,
                      "seconds": round(time.monotonic() - started, 2), "workbook": str(filename),
                      "checks": ["46-row import", "column mapping", "full background optimization", "responsive event loop",
                                 "calculated recommendation", "hull feasibility", "stage-power identity", "response holdout",
                                 "eight figure exports", "immutable run settings", "reload invalidation", "cancellation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
