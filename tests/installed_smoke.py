"""Check the wheel in a separate install directory using existing dependencies."""
from pathlib import Path
from dataclasses import replace
import json
import sys
from unittest.mock import patch

install_dir = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(install_dir))

import app
import mtpv_optimizer
from mtpv_optimizer.general import run_general
from mtpv_optimizer.presets import example
from mtpv_optimizer.launcher import main
from mtpv_optimizer.general_ui import GeneralWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

assert Path(app.__file__).resolve().parent == install_dir
assert Path(mtpv_optimizer.__file__).resolve().parent.parent == install_dir
p, frame = example("exchanger")
output = run_general(frame, replace(p, population=24, generations=12, runs=1))
assert output.predicted.pitch_mm.isin([2, 4, 6]).all()
original_exec = QApplication.exec_
checks = []
failures = []

def tested_exec(application):
    def inspect_and_quit():
        try:
            window = next(w for w in application.topLevelWidgets() if isinstance(w, GeneralWindow))
            assert window.isVisible()
            window.grab().save(str(output_dir / "installed_startup.png"))
            window.open_mtpv()
            window.mtpv_window._show_error = lambda title, exc: failures.append(str(exc))
            assert window.mtpv_window.isVisible()
            assert window.mtpv_window.frame is None
            checks.extend(["installed package imports", "two-objective constrained search", "actual launcher", "coupled interface opens without private data"])
            window.close()
        except Exception as exc:
            failures.append(str(exc))
        finally:
            application.quit()
    QTimer.singleShot(350, inspect_and_quit)
    return original_exec()

with patch.object(QApplication, "exec_", tested_exec):
    assert main() == 0
assert not failures, failures
assert len(checks) == 4
report = {"status": "passed", "version": mtpv_optimizer.__version__, "checks": checks,
          "environment": "separate package directory; existing Anaconda dependencies"}
(output_dir / "installed_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
