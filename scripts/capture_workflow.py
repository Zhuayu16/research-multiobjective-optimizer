"""Capture actual Qt workflow views using synthetic/analytical data only."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PyQt5.QtGui import QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from mtpv_optimizer.general_ui import GeneralWindow
from mtpv_optimizer.presets import example
from mtpv_optimizer.problem import load_project, save_project, Response, Constraint
from validate_feasibility import problem_data


def main():
    assets = ROOT/'docs'/'assets'/'workflow'; assets.mkdir(parents=True, exist_ok=True)
    folder = ROOT/'validation'/'results'
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion'); app.setFont(QFont('Microsoft YaHei UI',9))
    window=GeneralWindow(); window.resize(1680,1000); window.show()
    errors=[]; window.show_error=lambda error:errors.append(''.join(traceback.format_exception(type(error),error,error.__traceback__)))
    QTest.qWait(250)
    names=[]
    def capture(name,left,right):
        window.editor.setCurrentIndex(left); window.result_tabs.setCurrentIndex(right)
        QTest.qWait(200)
        assert window.grab().save(str(assets/f'{name}.png'))
        names.append(name)
    def run():
        window.run_optimization(); start=time.monotonic()
        while window.worker is not None and window.worker.isRunning():
            QTest.qWait(30)
            if time.monotonic()-start>180:
                window.worker.requestInterruption()
                raise RuntimeError('GUI capture timed out')
        QTest.qWait(200)
        assert not errors, errors
        assert window.output is not None
    p,frame=example('battery')
    p=replace(p,validation='group',group_column='batch',holdout_column='batch',holdout_value='B')
    window.accept_frame(frame,p); run()
    assert window.output.config['sample_count']==50 and window.output.config['holdout_count']==10
    for name,left,right in [('variables',0,0),('run_settings',4,1),('summary',3,1),
                            ('predicted',1,2),('input_pareto',1,3),('coverage',2,4),
                            ('holdout',3,5),('preferences',1,6)]:
        capture(name,left,right)
    restored_path=folder/'workflow_battery.optproj'
    save_project(restored_path,window.get_problem(),window.frame)
    restored, restored_frame=load_project(restored_path)
    window.accept_frame(restored_frame,restored)
    assert window.output is None
    run(); capture('project_restored',4,7)
    auxiliary=replace(p,objectives=[p.objectives[0],p.objectives[2]],responses=[Response('deltaT_K','K')],
                      constraints=p.constraints+[Constraint('deltaT_K','<=',6.5)])
    window.accept_frame(frame,auxiliary); run()
    assert len(window.output.bundle.metrics)==3 and len(window.output.bundle.target_names)==2
    capture('auxiliary_response',1,4)
    p,frame=example('exchanger')
    p=replace(p,model='Auto',runs=1)
    window.preset_box.setCurrentIndex(window.preset_box.findData('exchanger'))
    window.accept_frame(frame,p); run()
    assert len(window.output.bundle.candidate_metrics)>2
    capture('model_comparison',4,8)
    p,frame=problem_data('quadratic')
    window.preset_box.setCurrentIndex(-1)
    window.accept_frame(frame,p); run()
    assert window.output.predicted.iloc[0].loss < 1e-4
    capture('known_optimum',0,7)
    window.close(); app.processEvents()
    report={'status':'passed','screenshots':names,'data':'synthetic and analytical only',
            'battery_train':50,'battery_holdout':10,'project_restoration':'verified',
            'auxiliary_response':'3 modelled responses, 2 objectives',
            'auto_model_families':'actual cross-validation comparison'}
    (folder/'workflow_capture.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
