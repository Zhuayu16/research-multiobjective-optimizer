"""Capture real application windows for the README; no rendered mockups."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from PyQt5.QtGui import QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from app import OptimizationWindow
from mtpv_optimizer.general_ui import GeneralWindow
from mtpv_optimizer.presets import example
from mtpv_optimizer.problem import save_project


def main():
    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion"); app.setFont(QFont("Microsoft YaHei UI", 9))
    window = GeneralWindow(); window.resize(1560, 980); window.show()
    errors, report = [], []
    window.show_error = lambda exc: errors.append(str(exc))

    def capture(widget, name):
        QTest.qWait(200)
        path = assets / name
        if not widget.grab().save(str(path)):
            raise RuntimeError(f"Screenshot failed: {name}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"Empty screenshot: {name}")

    def wait_worker(widget):
        started = time.monotonic()
        while widget.worker is not None and widget.worker.isRunning():
            QTest.qWait(30)
            if time.monotonic() - started > 180:
                widget.worker.requestInterruption()
                raise RuntimeError("Screenshot optimization timed out")
        QTest.qWait(200)
        if errors:
            raise RuntimeError("; ".join(errors))
        if widget.output is None:
            raise RuntimeError("No optimization result")

    for key in ("battery", "exchanger", "structure"):
        problem, frame = example(key)
        destination = ROOT / "examples" / key
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / "data.csv", index=False, encoding="utf-8-sig")
        save_project(destination / "project.optproj", problem, frame)
        window.accept_frame(frame, problem)
        window.preset_box.setCurrentIndex(window.preset_box.findData(key))
        window.run_optimization(); wait_worker(window)
        window.editor.setCurrentIndex(1)
        window.result_tabs.setCurrentIndex(7)
        capture(window, "overview.png" if key == "battery" else f"{key}.png")
        report.append({"domain": key, "source": "synthetic_function_example", "population": problem.population,
                       "generations": problem.generations, "runs": problem.runs, "seed": problem.seed,
                       "predicted_count": len(window.output.predicted)})
        if key == "battery":
            window.editor.setCurrentIndex(2); window.result_tabs.setCurrentIndex(0)
            capture(window, "constraints.png")
            window.editor.setCurrentIndex(3); window.result_tabs.setCurrentIndex(8)
            capture(window, "validation.png")
    window.close()

    legacy = OptimizationWindow(); legacy.resize(1560, 980)
    legacy._show_error = lambda title, exc: errors.append(str(exc))
    legacy.show()
    capture(legacy, "mtpv.png")
    report.append({"domain": "mtpv_mteg", "source": "empty dedicated interface; user data required"})
    legacy.close(); app.processEvents()
    folder = ROOT / "outputs" / "readme_publication_20261005"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "screenshot_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "screenshots": sorted(p.name for p in assets.glob("*.png")), "runs": report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
