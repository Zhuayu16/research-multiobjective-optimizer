/* The worker receives files in memory; it never uploads their contents. */
const INDEX = 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/';
let ready;
async function initialize(id) {
  const stage = (percent, message) => self.postMessage({type:'progress', id, percent, message});
  stage(4, '正在加载浏览器计算环境');
  importScripts(INDEX + 'pyodide.js');
  const py = await loadPyodide({indexURL:INDEX});
  stage(12, '正在准备数值分析与代理模型');
  const packages=['numpy', 'scipy', 'joblib', 'threadpoolctl', 'pandas', 'scikit-learn', 'xlrd', 'micropip'];
  for(let attempt=1;attempt<=3;attempt++) {
    const failures=[];
    await py.loadPackage(packages, {errorCallback:message=>failures.push(message)});
    if(!failures.length) break;
    if(attempt===3) throw new Error('计算依赖下载未完成，请检查网络后重试。'+failures.join(' '));
    stage(12,`网络中断，正在重新下载计算依赖（${attempt}/2）`);
  }
  await py.runPythonAsync('import numpy, pandas, scipy, sklearn');
  stage(18, '正在准备 Excel 文件支持');
  await py.runPythonAsync("import micropip\nawait micropip.install(['openpyxl==3.1.5', 'et-xmlfile==2.0.0'])");
  const response = await fetch('./assets/engine.zip');
  if (!response.ok) throw new Error('计算内核下载失败，请刷新页面重试。');
  py.unpackArchive(await response.arrayBuffer(), 'zip', {extractDir:'/home/pyodide'});
  const manifest = await fetch('./assets/engine-manifest.json');
  if (!manifest.ok) throw new Error('内核版本记录下载失败。');
  py.globals.set('_browser_manifest_json', await manifest.text());
  const bridge = await fetch('./bridge.py');
  if (!bridge.ok) throw new Error('计算接口下载失败。');
  await py.runPythonAsync(await bridge.text());
  return py;
}
let queue = Promise.resolve();
self.onmessage = event => {
  const request = event.data;
  queue = queue.catch(()=>{}).then(async()=> {
    try {
      ready ??= initialize(request.id).catch(error => {ready=undefined; throw error;});
      const py = await ready;
      py.globals.set('_browser_request', JSON.stringify(request));
      const result = JSON.parse(await py.runPythonAsync('browser_call(_browser_request)'));
      self.postMessage({type:'result', id:request.id, result});
    } catch (error) {
      const detail=String(error.message || error);
      console.error(detail);
      const lines=detail.split('\n').filter(line=>line.trim());
      const meaningful=lines.findLast(line=>/^(ValueError|RuntimeError|TypeError|ImportError|ModuleNotFoundError|AttributeError):/.test(line.trim()));
      self.postMessage({type:'error', id:request.id, message:meaningful || detail});
    }
  });
};
