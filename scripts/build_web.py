"""Build the serverless site from a software-only, explicit source allowlist."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
MODULES = ['__init__', 'core', 'coupled', 'dat_parser', 'doe', 'domain', 'general',
           'nsga2', 'presets', 'problem', 'workflow']


def main():
    import argparse
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Copy only the public website assets to this directory.')
    args = parser.parse_args()
    import pandas as pd
    from mtpv_optimizer.presets import example
    folder = WEB / 'assets'
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(folder / 'engine.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for module in MODULES:
            relative = f'mtpv_optimizer/{module}.py'
            content = (ROOT / relative).read_text(encoding='utf-8').encode('utf-8')
            hashes[relative] = hashlib.sha256(content).hexdigest()
            entry = zipfile.ZipInfo(relative, date_time=(2026, 10, 5, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
    examples = {}
    for key in ['battery', 'exchanger', 'structure']:
        problem, frame = example(key)
        candidates = pd.read_csv(ROOT / f'validation/results/{key}_candidates.csv')
        examples[key] = dict(problem=problem.to_dict(), table=json.loads(frame.to_json(orient='split', index=False)),
                             preview=json.loads(candidates.to_json(orient='records')), synthetic=True)
    (folder / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
    manifest = dict(pyodide='0.29.3', source_sha256=hashes,
                    scope='General numerical-table workflow; no Qt, no private research records.',
                    browser_adjustment='Forest estimators run with n_jobs=1 in the browser.')
    (folder / 'engine-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    if args.output:
        static = ['index.html', 'styles.css', 'professional.css', 'app.js', 'analysis.js', 'worker.js', 'bridge.py',
                  'assets/icon.svg', 'assets/engine.zip', 'assets/engine-manifest.json', 'assets/examples.json',
                  'assets/benchmark-zdt1.png', 'assets/benchmark-optima.png']
        for relative in static:
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WEB / relative, destination)
    print(json.dumps({'status':'built', 'modules':len(MODULES), 'examples':3, 'private_data':'excluded'}))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    main()
