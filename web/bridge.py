"""Browser adapter. Computation is performed by the existing Python engine."""
import base64
import io
import json
from pathlib import Path
import time
import sys
import sklearn
import scipy
import openpyxl

import numpy as np
import pandas as pd
from mtpv_optimizer import core
from mtpv_optimizer.general import run_general, design_template
from mtpv_optimizer.problem import Problem, save_project
from mtpv_optimizer.workflow import export_workbook
from mtpv_optimizer.scientific import spatial_field

_models = core._candidate_models
def _serial_models(*args, **kwargs):
    candidates = _models(*args, **kwargs)
    for model in candidates.values():
        changes = {key: 1 for key in model.get_params(deep=True) if key.endswith('n_jobs')}
        if changes:
            model.set_params(**changes)
    return candidates
core._candidate_models = _serial_models

_last_output = None
_last_frame = None
_last_problem = None

def _records(table):
    return json.loads(table.to_json(orient='records', force_ascii=False, double_precision=15))

def _file_result(content, name, mime):
    return {'base64':base64.b64encode(content).decode(), 'name':name, 'mime':mime}

def _problem(raw):
    raw = dict(raw)
    raw['sources'] = []
    p = Problem.from_dict(raw)
    if p.population > 240 or p.generations > 300 or p.runs > 5:
        raise ValueError('浏览器单次计算上限：种群 240、迭代 300、独立运行 5 次。')
    return p

def dispatch(request, progress):
    global _last_output, _last_frame, _last_problem
    op, payload = request['op'], request.get('payload', {})
    if op == 'import':
        content = base64.b64decode(payload['base64'])
        suffix = Path(payload['name']).suffix.lower()
        if suffix in ('.xlsx', '.xls'):
            book = pd.ExcelFile(io.BytesIO(content), engine='openpyxl' if suffix == '.xlsx' else 'xlrd')
            sheet = payload.get('sheet') or book.sheet_names[0]
            frame = pd.read_excel(book, sheet_name=sheet)
            names = book.sheet_names
        elif suffix in ('.csv', '.tsv', '.txt'):
            frame = pd.read_csv(io.BytesIO(content), sep='\t' if suffix == '.tsv' else None, engine='python', encoding='utf-8-sig')
            names = []
        elif suffix == '.json':
            records = json.loads(content.decode('utf-8-sig'))
            if not isinstance(records, list):
                raise ValueError('JSON 数据应为记录列表；项目请使用 .optproj 文件。')
            frame = pd.DataFrame(records)
            names = []
        else:
            raise ValueError('请选择 CSV、TSV、Excel、记录列表 JSON 或 .optproj。')
        if frame.empty or len(frame) > 50000:
            raise ValueError('请选择 1–50,000 行的非空数值表格。')
        frame.columns = [str(c) for c in frame.columns]
        return dict(columns=list(frame.columns), rows=_records(frame), sheets=names,
                    selected_sheet=sheet if names else '', source_rows=len(frame))
    if op == 'run':
        _last_output = None
        p = _problem(payload['problem'])
        frame = pd.DataFrame(payload['rows'], columns=payload['columns'])
        started = time.perf_counter()
        output = run_general(frame, p, progress=progress)
        output.config['calculation_audit']['runtime_initialization_seconds'] = payload.get('runtime_initialization_seconds',0)
        output.config['calculation_audit']['runtime_warm'] = payload.get('runtime_warm',False)
        output.config['browser_runtime'] = dict(platform=sys.platform, python=sys.version.split()[0],
            numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
            sklearn=sklearn.__version__, openpyxl=openpyxl.__version__,
            **json.loads(globals().get('_browser_manifest_json', '{}')))
        _last_output, _last_frame, _last_problem = output, frame.copy(), p
        return dict(predicted=_records(output.predicted), observed=_records(output.observed),
                    metrics=_records(core.metrics_frame(output.bundle)),
                    model_comparison=_records(output.bundle.candidate_metrics),
                    validation=_records(output.bundle.validation_frame),
                    holdout=_records(getattr(output.bundle, 'holdout_frame', pd.DataFrame())),
                    holdout_metrics=_records(output.holdout_metrics),
                    sensitivity=_records(output.sensitivity), runs=_records(output.run_summary),
                    constraints=_records(output.constraint_audit), cleaning=_records(output.cleaning_audit),
                    decision_matrix=_records(output.decision_matrix), observed_decision_matrix=_records(output.observed_decision_matrix),
                    correlation={**{k:v for k,v in output.correlation.items() if k not in ('pearson','spearman','pairs')},
                        **{k:json.loads(output.correlation[k].to_json(orient='values')) for k in ('pearson','spearman','pairs')}},
                    search_history=_records(output.search_history),
                    config=output.config, notes=output.warnings, elapsed_seconds=time.perf_counter()-started)
    if op == 'field':
        frame = pd.DataFrame(payload['rows'], columns=payload['columns'])
        if len(frame) > 50000:
            raise ValueError('空间数据最多支持 50,000 行。')
        result = spatial_field(frame, payload['x'], payload['y'], payload['value'], payload.get('resolution',50))
        return {**{k:v for k,v in result.items() if k not in ('points','grid')},
                'points':_records(result['points']), 'grid':_records(result['grid']),
                'library_versions':{'numpy':np.__version__, 'scipy':scipy.__version__},
                'engine_source_sha256':json.loads(globals().get('_browser_manifest_json','{}')).get('source_sha256',{})}
    if op == 'doe':
        table = design_template(_problem(payload['problem']), payload.get('samples',30), payload.get('centers',3))
        return _file_result(table.to_csv(index=False).encode('utf-8-sig'), 'next-experiment.csv', 'text/csv')
    if op == 'export':
        if _last_output is None:
            raise ValueError('请先完成一次优化。')
        kind = payload['kind']
        if kind == 'csv':
            return _file_result(_last_output.predicted.to_csv(index=False).encode('utf-8-sig'), 'pareto-candidates.csv', 'text/csv')
        if kind == 'xlsx':
            path = Path('/tmp/research-results.xlsx')
            export_workbook(_last_output, path)
            return _file_result(path.read_bytes(), path.name, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if kind == 'project':
            path = Path('/tmp/research-project.optproj')
            save_project(path, _last_problem, _last_frame)
            return _file_result(path.read_bytes(), path.name, 'application/json')
        if kind == 'config':
            return _file_result(json.dumps(_last_output.config, ensure_ascii=False, indent=2).encode('utf-8'), 'run-configuration.json', 'application/json')
    raise ValueError('未知操作。')

def browser_call(raw_request):
    from js import postMessage
    from pyodide.ffi import to_js
    request = json.loads(raw_request)
    def progress(value, message):
        postMessage(to_js({'type':'progress', 'id':request['id'], 'percent':value, 'message':message}))
    return json.dumps(dispatch(request, progress), ensure_ascii=False, allow_nan=False)
