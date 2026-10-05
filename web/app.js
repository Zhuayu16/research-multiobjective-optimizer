const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clone = value => structuredClone(value);
const state = {examples:{}, columns:[], rows:[], p:null, targets:[], result:null, preview:[], busy:false, showPredicted:true, showObserved:true, importedFile:null};
let worker, sequence=0, pending=new Map(), toastTimer;

function toast(message) { $('toast').textContent=message; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,7000); }
function format(value, digits=5) {
  if(value===null || value===undefined || value==='') return '—';
  if(typeof value==='boolean') return value?'是':'否';
  if(typeof value!=='number') return String(value);
  if(!Number.isFinite(value)) return '—';
  if(value===0) return '0';
  if(Math.abs(value)<0.0001 || Math.abs(value)>=1e7) return value.toExponential(2);
  return Number(value.toPrecision(digits)).toLocaleString('en-US',{maximumFractionDigits:8});
}
function download(content,name,mime='application/octet-stream') {
  const url=URL.createObjectURL(new Blob([content],{type:mime}));
  const a=document.createElement('a'); a.href=url; a.download=name; document.body.append(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),10000);
}
function downloadResponse(result) { const bytes=Uint8Array.from(atob(result.base64),c=>c.charCodeAt(0)); download(bytes,result.name,result.mime); }
function toBase64(buffer) { const bytes=new Uint8Array(buffer); let text=''; for(let i=0;i<bytes.length;i+=32768) text+=String.fromCharCode(...bytes.subarray(i,i+32768)); return btoa(text); }
function csv(rows,columns) { const cell=x=>'"'+String(x??'').replaceAll('"','""')+'"'; return '\ufeff'+[columns.map(cell).join(','),...rows.map(row=>columns.map(c=>cell(row[c])).join(','))].join('\r\n'); }

function createWorker() {
  worker=new Worker(new URL('./worker.js',import.meta.url));
  worker.onmessage=event=>{
    const data=event.data;
    if(!pending.has(data.id)) return;
    if(data.type==='progress') { $('run-status').textContent=data.message; $('progress-bar').style.width=`${Math.max(3,Math.min(100,data.percent))}%`; return; }
    const {resolve,reject}=pending.get(data.id); pending.delete(data.id);
    if(data.type==='result') resolve(data.result); else reject(new Error(data.message));
  };
  worker.onerror=event=>{ const error=new Error(event.message || '浏览器计算环境未能启动，请检查网络后重试。'); pending.forEach(p=>p.reject(error)); pending.clear(); worker.terminate(); worker=undefined; };
}
function request(op,payload) {
  if(!worker) createWorker();
  const id=++sequence;
  return new Promise((resolve,reject)=>{ pending.set(id,{resolve,reject}); worker.postMessage({id,op,payload}); });
}
function setBusy(busy, message='正在准备计算') {
  state.busy=busy; $('config-fieldset').disabled=busy; $('run-button').disabled=busy || !state.p;
  $('run-button').textContent=busy?'正在计算…':'运行优化 ↗';
  ['example-select','import-button','sheet-select','download-input','doe-button'].forEach(id=>$(id).disabled=busy);
  $('cancel-button').hidden=!busy; $('progress-track').hidden=!busy;
  $('progress-bar').style.width='3%'; $('export-toggle').disabled=busy || !state.result;
  if(message) $('run-status').textContent=message;
}
function clearResult(message='配置已变更，运行优化以生成当前结果。') {
  state.result=null; state.preview=[]; $('dirty-badge').textContent='待计算'; $('export-options').hidden=true; $('export-toggle').setAttribute('aria-expanded','false');
  $('export-toggle').disabled=true; $('run-status').textContent=message; renderResults();
}
function options(selected, blank=false) { return (blank?'<option value="">不设置</option>':'')+state.columns.map(c=>`<option value="${escapeHTML(c)}" ${c===selected?'selected':''}>${escapeHTML(c)}</option>`).join(''); }
function selectField(label, content, attrs='') { return `<label class="field">${label}<select ${attrs}>${content}</select></label>`; }
function numberField(label,value,attrs='') { return `<label class="field">${label}<input type="number" step="any" value="${escapeHTML(value)}" ${attrs}></label>`; }
function unitField(value,attrs='') { return `<label class="field">单位<input value="${escapeHTML(value)}" placeholder="可选" ${attrs}></label>`; }
function propertyAttrs(type,index,key) {return `data-array="${type}" data-index="${index}" data-property="${key}"`;}

function renderVariables() {
  $('variable-fields').innerHTML=state.p.variables.map((v,i)=>{
    const attr=key=>propertyAttrs('variables',i,key);
    return `<div class="definition-card"><div class="definition-card-head"><select aria-label="变量 ${i+1} 数据列" ${attr('column')}>${options(v.column)}</select><button class="remove-button" data-remove="variables" data-index="${i}" aria-label="删除变量 ${i+1}">×</button></div><div class="field-row">${selectField('类型',`<option value="continuous" ${v.kind==='continuous'?'selected':''}>连续</option><option value="integer" ${v.kind==='integer'?'selected':''}>整数</option><option value="discrete" ${v.kind==='discrete'?'selected':''}>数值集合</option>`,attr('kind'))}${unitField(v.unit,attr('unit'))}</div><div class="field-row">${numberField('下界',v.lower,attr('lower'))}${numberField('上界',v.upper,attr('upper'))}</div>${v.kind==='discrete'?`<label class="field">允许值（逗号分隔）<input value="${escapeHTML(v.allowed.join(', '))}" placeholder="2, 4, 6" ${attr('allowed')}></label>`:''}</div>`;
  }).join('');
}
function renderTargets() {
  $('target-fields').innerHTML=state.targets.map((t,i)=>{
    const attr=key=>propertyAttrs('targets',i,key);
    return `<div class="definition-card"><div class="definition-card-head"><select aria-label="响应 ${i+1} 数据列" ${attr('column')}>${options(t.column)}</select><button class="remove-button" data-remove="targets" data-index="${i}" aria-label="删除响应 ${i+1}">×</button></div><div class="field-row">${selectField('方向 / 用途',`<option value="min" ${t.role==='min'?'selected':''}>最小化 ↓</option><option value="max" ${t.role==='max'?'selected':''}>最大化 ↑</option><option value="aux" ${t.role==='aux'?'selected':''}>仅用于约束</option>`,attr('role'))}${unitField(t.unit,attr('unit'))}</div>${t.role!=='aux'?numberField('相对权重',t.weight,attr('weight')+' min="0"'):''}</div>`;
  }).join('');
}
function renderConstraints() {
  $('constraint-fields').innerHTML=state.p.constraints.map((c,i)=>{
    const attr=key=>propertyAttrs('constraints',i,key);
    return `<div class="constraint-card"><div><span>约束 ${i+1}</span><button class="remove-button" data-remove="constraints" data-index="${i}" aria-label="删除约束 ${i+1}">×</button></div><label class="field">变量或响应表达式<input value="${escapeHTML(c.expression)}" placeholder="例如 Tmax_K" ${attr('expression')}></label><div class="constraint-values">${selectField('关系',['<=','>=','=='].map(op=>`<option value="${escapeHTML(op)}" ${op===c.relation?'selected':''}>${escapeHTML(op)}</option>`).join(''),attr('relation'))}${numberField('阈值',c.limit,attr('limit'))}</div><div class="field-row">${numberField('允许容差',c.tolerance,attr('tolerance')+' min="0"')}${numberField('归一化尺度',c.scale,attr('scale')+' min="0.000001"')}</div></div>`;
  }).join('') || '<div class="field-note">尚未设置约束。可添加响应上限、下限或设计表达式关系。</div>';
}
const settings={ 'problem-name':'name','domain':'domain','max-distance':'max_distance','model':'model','validation':'validation','folds':'folds','group-column':'group_column','time-column':'time_column','holdout-column':'holdout_column','holdout-value':'holdout_value','duplicates':'duplicates','population':'population','generations':'generations','runs':'runs','seed':'seed','decision':'decision'};
const numericSettings=new Set(['folds','population','generations','runs','seed','max_distance']);
function renderConfig() {
  state.targets=[...state.p.objectives.map(t=>({...t,role:t.direction})),...(state.p.responses||[]).map(t=>({...t,role:'aux',weight:0}))];
  ['group-column','time-column','holdout-column'].forEach(id=>$(id).innerHTML=options(state.p[settings[id]],true));
  Object.entries(settings).forEach(([id,key])=>$(id).value=state.p[key]??'');
  renderVariables(); renderTargets(); renderConstraints();
}
function readProblem() {
  const p=clone(state.p);
  p.objectives=state.targets.filter(t=>t.role!=='aux').map(t=>({column:t.column,direction:t.role,unit:t.unit||'',weight:t.weight}));
  p.responses=state.targets.filter(t=>t.role==='aux').map(t=>({column:t.column,unit:t.unit||''}));
  p.sources=[];
  return p;
}
function loadDataset(problem, columns, rows, {preview=[],name,synthetic=false}={}) {
  if(state.busy) {toast('请先取消当前计算，再更换数据。'); return;}
  state.p=clone(problem); state.p.sources=[]; state.columns=columns; state.rows=rows; state.result=null; state.preview=preview;
  state.showPredicted=state.showObserved=true;
  $('toggle-predicted').setAttribute('aria-pressed','true'); $('toggle-observed').setAttribute('aria-pressed','true');
  $('dataset-name').textContent=name || problem.name;
  $('dataset-info').textContent=`${rows.length} 个样本 · ${columns.length} 个数据列${synthetic?' · 合成函数示例':' · 本地导入'}`;
  $('dirty-badge').textContent=preview.length?'示例预览':'待计算'; $('export-toggle').disabled=true; $('export-options').hidden=true;
  $('run-status').textContent=synthetic?'点击运行优化，生成你自己的计算结果。首次运行需要下载计算环境。':'已建立初始配置，请核对变量、目标和验证设置后运行。';
  renderConfig(); setupAxes(); renderResults(); $('run-button').disabled=false;
}
function selectExample(key,scroll=false) {
  const example=state.examples[key]; if(!example) return;
  const rows=example.table.data.map(values=>Object.fromEntries(example.table.columns.map((c,i)=>[c,values[i]])));
  loadDataset(example.problem,example.table.columns,rows,{preview:example.preview,name:example.problem.name,synthetic:true});
  $('example-select').value=key; state.importedFile=null; $('sheet-picker').hidden=true;
  if(scroll) $('workspace').scrollIntoView({behavior:'smooth'});
}
function defaultProblem(columns,rows,name) {
  const numeric=columns.filter(c=>!['case','batch','time'].includes(c) && rows.some(r=>r[c]!==null&&r[c]!=='') && rows.every(r=>r[c]===null||r[c]===''||Number.isFinite(Number(r[c]))));
  if(numeric.length<2) throw new Error('至少需要两列数值：一个设计变量和一个响应。');
  const count=numeric.length>=3?2:1;
  const variables=numeric.slice(0,count).map(column=>{const values=rows.map(r=>Number(r[column])).filter(Number.isFinite);return {column,lower:Math.min(...values),upper:Math.max(...values),unit:'',kind:'continuous',allowed:[]};});
  return {...clone(state.examples.battery.problem),name:name.replace(/\.[^.]+$/,''),variables,
    objectives:[{column:numeric[count],direction:'min',weight:1,unit:''}],responses:[],constraints:[],model:'RSM-Quadratic',domain:'box',validation:'shuffled',holdout_column:'',holdout_value:'',sources:[],description:'用户本地导入的数据。'};
}
function setupAxes() {
  const p=readProblem(), targets=p.objectives.map(t=>t.column);
  const names=[...targets,...p.variables.map(v=>v.column)];
  ['axis-x','axis-y'].forEach(id=>$(id).innerHTML=names.map(c=>`<option value="${escapeHTML(c)}">${escapeHTML(c)}</option>`).join(''));
  $('axis-x').value=targets.length>1?targets[0]:p.variables[0]?.column;
  $('axis-y').value=targets.length>1?targets[1]:targets[0];
  $('axis-color').innerHTML='<option value="">无</option>'+targets.map(c=>`<option value="${escapeHTML(c)}">${escapeHTML(c)}</option>`).join('');
  if(targets.length>2) $('axis-color').value=targets[2];
}

function table(id,rows,columns=null,limit=100) {
  if(!rows?.length) {$(id).innerHTML='<table><tbody><tr><td class="empty-row">暂无记录。请运行优化，或检查是否配置了独立留出与模型比较。</td></tr></tbody></table>';return;}
  columns=columns||Object.keys(rows[0]);
  $(id).innerHTML=`<table><thead><tr>${columns.map(c=>`<th scope="col">${escapeHTML(c)}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,limit).map(row=>`<tr>${columns.map(c=>`<td>${escapeHTML(format(row[c]))}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function emptyChart(message='运行优化，探索你的设计空间。') {return `<div class="empty-state"><span class="empty-glyph">↗</span><strong>${escapeHTML(message)}</strong><p>设置设计变量、目标方向与工程约束，生成当前配置下的候选和验证结果。</p></div>`;}
function scatter(host, {predicted=[],observed=[],x,y,color='',recommended=true,interactive=true,label='',height=320}) {
  const valid=row=>row[x]!==null&&row[y]!==null&&Number.isFinite(Number(row[x]))&&Number.isFinite(Number(row[y]));
  const a=predicted.filter(valid),b=observed.filter(valid),all=[...a,...b];
  if(!all.length || !x || !y) {host.innerHTML=emptyChart();return;}
  const W=680,H=height,L=66,R=24,T=20,B=49;
  let xmin=Math.min(...all.map(r=>Number(r[x]))),xmax=Math.max(...all.map(r=>Number(r[x]))),ymin=Math.min(...all.map(r=>Number(r[y]))),ymax=Math.max(...all.map(r=>Number(r[y])));
  const padX=(xmax-xmin||Math.max(Math.abs(xmin)*.1,1))*.075,padY=(ymax-ymin||Math.max(Math.abs(ymin)*.1,1))*.09;
  xmin-=padX;xmax+=padX;ymin-=padY;ymax+=padY;
  const X=value=>L+(Number(value)-xmin)/(xmax-xmin)*(W-L-R),Y=value=>H-B-(Number(value)-ymin)/(ymax-ymin)*(H-T-B);
  const palette=t=>`rgb(${Math.round(15+100*t)},${Math.round(105+75*t)},${Math.round(220-10*t)})`;
  const cs=all.map(r=>Number(r[color])).filter(Number.isFinite),cmin=Math.min(...cs),cmax=Math.max(...cs);
  const c= row=>color&&Number.isFinite(Number(row[color]))?palette((Number(row[color])-cmin)/(cmax-cmin||1)):'#1680d7';
  let lines='',ticks='';
  for(let i=0;i<=4;i++) {const xx=L+(W-L-R)*i/4,yy=T+(H-T-B)*i/4;
    lines+=`<path d="M${L} ${yy}H${W-R}" stroke="#edf0f4" stroke-width="1"/>`;
    ticks+=`<text x="${xx}" y="${H-B+20}" text-anchor="middle">${escapeHTML(format(xmin+(xmax-xmin)*i/4,3))}</text><text x="${L-12}" y="${yy+3}" text-anchor="end">${escapeHTML(format(ymax-(ymax-ymin)*i/4,3))}</text>`;
  }
  const points=(rows,kind)=>rows.map((r,i)=>`<circle class="point" cx="${X(r[x])}" cy="${Y(r[y])}" r="${kind==='observed'?3.5:3.1}" fill="${kind==='observed'?'white':c(r)}" fill-opacity="${kind==='observed'?1:.7}" stroke="${kind==='observed'?'#acb7c6':'white'}" stroke-width="${kind==='observed'?1.3:.6}" data-kind="${kind}" data-index="${i}"><title>${escapeHTML(`${kind==='observed'?'输入样本':'预测候选'} · ${x}: ${format(r[x])} · ${y}: ${format(r[y])}`)}</title></circle>`).join('');
  const rec=a[0];
  const star=rec&&recommended?`<path d="M0-8 2.2-2.6 8-2.6 3.6 1 5.2 7 0 3.6-5.2 7-3.6 1-8-2.6-2.2-2.6Z" transform="translate(${X(rec[x])} ${Y(rec[y])})" fill="#d68c61" stroke="white" stroke-width="1.2"><title>当前偏好推荐</title></path>`:'';
  host.innerHTML=`<svg class="scatter-chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHTML(label||`${x} 与 ${y} 的设计权衡`)}" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif"><rect width="${W}" height="${H}" fill="white"/>${lines}<path d="M${L} ${T}V${H-B}H${W-R}" fill="none" stroke="#dbe1e8" stroke-width="1"/><g fill="#98a2b0" font-size="10">${ticks}</g>${points(b,'observed')}${points(a,'predicted')}${star}<text x="${(W+L-R)/2}" y="${H-9}" fill="#818e9f" text-anchor="middle" font-size="11">${escapeHTML(x)}</text><text transform="translate(15 ${(H+T-B)/2}) rotate(-90)" fill="#818e9f" text-anchor="middle" font-size="11">${escapeHTML(y)}</text></svg>`;
  if(interactive) host.querySelectorAll('.point').forEach(point=>{
    const row=(point.dataset.kind==='observed'?b:a)[Number(point.dataset.index)];
    point.addEventListener('pointermove',event=>{
      const tip=$('chart-tooltip');tip.innerHTML=`<strong>${point.dataset.kind==='observed'?'输入表方案':'预测候选'} ${escapeHTML(row['推荐排序']??'')}</strong>${escapeHTML(x)}: ${escapeHTML(format(row[x]))}<br>${escapeHTML(y)}: ${escapeHTML(format(row[y]))}${color?`<br>${escapeHTML(color)}: ${escapeHTML(format(row[color]))}`:''}`;
      tip.hidden=false;tip.style.left=Math.max(8,Math.min(window.innerWidth-260,event.clientX+15))+'px';tip.style.top=Math.min(window.innerHeight-125,event.clientY+12)+'px';
    });
    point.addEventListener('pointerleave',()=>$('chart-tooltip').hidden=true);
    point.addEventListener('click',()=>renderRecommendation(row,point.dataset.kind==='observed'?'当前查看 · 输入表方案':'当前查看 · 预测候选'));
  });
}
function renderRecommendation(row,label='当前偏好推荐') {
  if(!row) {$('recommendation').innerHTML='<p>运行优化后，将在这里显示按当前权重与排序方法推荐的方案。</p>';return;}
  const p=readProblem(), fields=[...p.variables,...p.objectives,...p.responses];
  $('recommendation').innerHTML=`<div class="rec-head"><strong>${escapeHTML(label)}</strong><span>${escapeHTML(p.decision)} · ${state.result?'本次计算':'示例预览'}</span></div><div class="rec-values">${fields.map(f=>`<div><span>${escapeHTML(f.column)}</span><b>${escapeHTML(format(row[f.column],5))}</b><small>${escapeHTML(f.unit||'')}</small></div>`).join('')}</div><p>按当前偏好选出的权衡方案，需要通过原函数、试验或仿真重新确认。</p>`;
}
function renderOverview() {
  const result=state.result,p=readProblem(),pred=result?.predicted||state.preview,obs=result?.observed||[];
  const stats=result?[['训练样本',result.config.sample_count,`${result.config.holdout_count} 个独立留出`],['预测候选',pred.length,`${obs.length} 个输入表 Pareto`],['独立运行',result.config.runs,`${format(result.elapsed_seconds,3)} 秒 · 本地计算`]]:[['输入样本',state.rows.length,'完整数据用于计算'],['设计变量',p.variables.length,'连续 / 整数 / 集合'],['优化目标',p.objectives.length,state.preview.length?'合成示例结果预览':'等待本次优化']];
  $('result-stats').innerHTML=stats.map(([label,value,note])=>`<div class="stat"><span>${escapeHTML(label)}</span><strong>${escapeHTML(format(value))}</strong><small>${escapeHTML(note)}</small></div>`).join('');
  scatter($('main-chart'),{predicted:state.showPredicted?pred:[],observed:state.showObserved?obs:[],x:$('axis-x').value,y:$('axis-y').value,color:$('axis-color').value});
  renderRecommendation(pred[0]);
}
function renderResults() {
  if(!state.p) return;
  const result=state.result,p=readProblem();
  $('result-state').textContent=result?'本次优化已完成':state.preview.length?'合成示例预览':'等待计算';
  $('result-title').textContent=result?'每一份权衡，都有依据。':state.preview.length?'先看见，再进一步。':'准备探索你的设计空间。';
  renderOverview(); renderCandidateTable();
  table('data-table',state.rows,state.columns); $('data-description').textContent=`${state.rows.length} 行完整输入数据 · ${state.columns.length} 列`;
  table('metric-table',result?.metrics);table('model-table',result?.model_comparison);table('holdout-table',result?.holdout_metrics);
  if(result?.validation?.length) {
    const target=p.objectives[0]?.column, records=result.validation.filter(r=>r['目标']===target&&r['用于验证']!==false);
    scatter($('validation-chart'),{predicted:records.map(r=>({truth:r['实测值'],prediction:r['冻结模型预测值']??r['交叉验证预测值']??r['预测值']})),x:'truth',y:'prediction',recommended:false,interactive:false,label:'交叉验证预测与实测数值',height:260});
  } else $('validation-chart').innerHTML=emptyChart('完成计算后，检查模型预测。');
  if(result) {
    const rows=[['计算位置','当前浏览器 · WebAssembly'],['代理模型',Object.entries(result.config.selected_models).map(([k,v])=>`${k}: ${v}`).join(' / ')],['验证方式',result.config.cv],['随机种子',result.config.seed],['搜索设置',`种群 ${result.config.population} / 迭代 ${result.config.generations} / ${result.config.runs} 次`],['软件版本',result.config.software_version],['训练数据 SHA-256',result.config.clean_data_sha256]];
    $('trace-summary').innerHTML=rows.map(([k,v])=>`<div class="trace-row"><span>${escapeHTML(k)}</span><b>${escapeHTML(v)}</b></div>`).join('')+`<div class="trace-notes">${result.notes.map(n=>`<p>${escapeHTML(n)}</p>`).join('')}</div>`;
    $('config-json').textContent=JSON.stringify(result.config,null,2);
  } else {$('trace-summary').innerHTML=emptyChart('可追溯的结果，从一次计算开始。');$('config-json').textContent='尚未运行。';}
}
function renderCandidateTable() {const source=$('candidate-source').value;const rows=state.result?.[source]||(source==='predicted'?state.preview:[]); const p=readProblem();table('candidate-table',rows,['推荐排序',...p.variables.map(v=>v.column),...p.objectives.map(t=>t.column),...p.responses.map(t=>t.column),'综合效用','最近样本距离','约束违反量']);}

async function run() {
  if(state.busy||!state.p) return;
  const snapshot={problem:readProblem(),columns:[...state.columns],rows:clone(state.rows)};
  const invalid=$('config-fieldset').querySelector('input:invalid');
  if(invalid) {invalid.reportValidity();invalid.focus();return;}
  clearResult('正在开始本次研究');setBusy(true,'正在准备浏览器计算环境。首次运行需要网络连接。');
  try {
    const result=await request('run',snapshot);state.result=result;$('dirty-badge').textContent='本次计算';renderResults();
    $('run-status').textContent=`已完成 · ${result.predicted.length} 个预测候选 · ${format(result.elapsed_seconds,3)} 秒。`;
    toast('优化完成。查看模型验证，并下载项目以留存这次研究。');
  } catch(error) { $('run-status').textContent=error.message==='已取消'?'已取消计算，可修改配置后重新开始。':'计算未完成：'+error.message; if(error.message!=='已取消') toast('计算未完成。'+error.message); }
  finally {setBusy(false,null);}
}
async function importFile(file,sheet) {
  if(!file || state.busy) return;
  if(file.size>20*1024*1024) {toast('请选择不超过 20 MB 的文件，以保持浏览器流畅。');return;}
  if(file.name.toLowerCase().endsWith('.optproj')) {
    try {const raw=JSON.parse(await file.text());if(raw.format!=='research-optimizer-project'||raw.version!==1||!Array.isArray(raw.table?.columns)||!Array.isArray(raw.table?.data)||!raw.problem?.variables?.length)throw new Error('不是受支持的研究项目文件。'); if(raw.table.data.length>50000)throw new Error('项目最多支持 50,000 个样本。');const rows=raw.table.data.map(values=>Object.fromEntries(raw.table.columns.map((c,i)=>[c,values[i]])));loadDataset(raw.problem,raw.table.columns,rows,{name:file.name});$('sheet-picker').hidden=true;state.importedFile=null;toast('项目已恢复。数据与配置已加载，可重新计算。');} catch(error) {toast('项目无法打开：'+error.message);}return;
  }
  setBusy(true,'正在本地读取文件。首次读取需要下载计算环境。');
  try {
    const result=await request('import',{name:file.name,base64:toBase64(await file.arrayBuffer()),sheet});
    setBusy(false,null);
    loadDataset(defaultProblem(result.columns,result.rows,file.name),result.columns,result.rows,{name:file.name});
    state.importedFile=file;$('sheet-picker').hidden=result.sheets.length<2;
    $('sheet-select').innerHTML=result.sheets.map(s=>`<option value="${escapeHTML(s)}" ${s===result.selected_sheet?'selected':''}>${escapeHTML(s)}</option>`).join('');
    toast('数据已导入。请核对变量、目标与单位，再运行优化。');
  }catch(error){$('run-status').textContent='文件读取未完成：'+error.message;toast(error.message);}finally{setBusy(false,null);}
}

document.querySelectorAll('[data-config]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('[data-config]').forEach(b=>{b.setAttribute('aria-selected',String(b===button));$('config-'+b.dataset.config).hidden=b!==button;});
}));
document.querySelectorAll('[data-result]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('[data-result]').forEach(b=>{b.setAttribute('aria-selected',String(b===button));$('view-'+b.dataset.result).hidden=b!==button;});
}));
$('config-fieldset').addEventListener('change',event=>{
  if(!state.p||state.busy) return;
  const input=event.target;
  if(input.dataset.array) {
    const {array,index,property}=input.dataset, list=array==='targets'?state.targets:state.p[array];
    let value=input.value;
    if(property==='allowed') value=value.split(/[,，]/).filter(v=>v.trim()!=='').map(v=>Number(v.trim())).filter(Number.isFinite);
    else if(input.type==='number') value=value===''?null:Number(value);
    list[Number(index)][property]=value;
    if(property==='kind')renderVariables(); if(property==='role')renderTargets();
  }else if(settings[input.id]) {
    const key=settings[input.id];state.p[key]=numericSettings.has(key)?(input.value===''?null:Number(input.value)):input.value;
  }else return;
  clearResult();setupAxes();renderOverview();
});
$('config-fieldset').addEventListener('click',event=>{
  const button=event.target.closest('[data-remove]');if(!button||state.busy)return;
  const array=button.dataset.remove,list=array==='targets'?state.targets:state.p[array];list.splice(Number(button.dataset.index),1);
  if(array==='variables')renderVariables();else if(array==='targets')renderTargets();else renderConstraints();clearResult();setupAxes();renderOverview();
});
$('add-variable').addEventListener('click',()=>{
  const used=new Set([...state.p.variables.map(v=>v.column),...state.targets.map(t=>t.column)]),column=state.columns.find(c=>!used.has(c))||state.columns[0];
  const values=state.rows.map(r=>Number(r[column])).filter(Number.isFinite);
  state.p.variables.push({column,lower:Math.min(...values),upper:Math.max(...values),unit:'',kind:'continuous',allowed:[]});renderVariables();clearResult();setupAxes();renderOverview();
});
$('add-target').addEventListener('click',()=>{
  const used=new Set([...state.p.variables.map(v=>v.column),...state.targets.map(t=>t.column)]),column=state.columns.find(c=>!used.has(c))||state.columns[0];
  state.targets.push({column,role:'min',unit:'',weight:1});renderTargets();clearResult();setupAxes();renderOverview();
});
$('add-constraint').addEventListener('click',()=>{state.p.constraints.push({expression:state.targets[0]?.column||state.p.variables[0]?.column||'',relation:'<=',limit:0,tolerance:0,scale:1});renderConstraints();clearResult();});
$('example-select').addEventListener('change',event=>selectExample(event.target.value));
document.querySelectorAll('[data-example]').forEach(button=>button.addEventListener('click',()=>selectExample(button.dataset.example,true)));
$('try-example').addEventListener('click',()=>selectExample('exchanger',true));
$('run-button').addEventListener('click',run);
$('cancel-button').addEventListener('click',()=>{worker?.terminate();worker=undefined;pending.forEach(p=>p.reject(new Error('已取消')));pending.clear();state.result=null;renderResults();});
$('import-button').addEventListener('click',()=>$('file-input').click());
$('file-input').addEventListener('change',event=>{const file=event.target.files[0];event.target.value='';importFile(file);});
$('sheet-select').addEventListener('change',event=>importFile(state.importedFile,event.target.value));
['axis-x','axis-y','axis-color'].forEach(id=>$(id).addEventListener('change',renderOverview));
['predicted','observed'].forEach(kind=>$('toggle-'+kind).addEventListener('click',()=>{const key=kind==='predicted'?'showPredicted':'showObserved';state[key]=!state[key];$('toggle-'+kind).setAttribute('aria-pressed',String(state[key]));renderOverview();}));
$('candidate-source').addEventListener('change',renderCandidateTable);
$('download-input').addEventListener('click',()=>download(csv(state.rows,state.columns),'input-data.csv','text/csv;charset=utf-8'));
$('export-toggle').addEventListener('click',()=>{$('export-options').hidden=!$('export-options').hidden;$('export-toggle').setAttribute('aria-expanded',String(!$('export-options').hidden));});
document.addEventListener('click',event=>{if(!event.target.closest('.export-menu')){$('export-options').hidden=true;$('export-toggle').setAttribute('aria-expanded','false');}});
document.querySelectorAll('[data-export]').forEach(button=>button.addEventListener('click',async()=>{
  if(!state.result||state.busy)return;$('export-options').hidden=true;$('export-toggle').setAttribute('aria-expanded','false');
  if(button.dataset.export==='svg'){const chart=$('main-chart').querySelector('svg');if(chart)download(new XMLSerializer().serializeToString(chart),'design-tradeoff.svg','image/svg+xml');return;}
  setBusy(true,'正在生成本次结果文件');
  try{downloadResponse(await request('export',{kind:button.dataset.export}));$('run-status').textContent='结果已导出，包含本次计算的配置与记录。';}catch(error){toast(error.message);}finally{setBusy(false,null);}
}));
$('doe-button').addEventListener('click',async()=>{
  if(state.busy||!state.p)return;setBusy(true,'正在生成下一轮试验设计');
  try{downloadResponse(await request('doe',{problem:readProblem(),samples:30,centers:3}));$('run-status').textContent='已导出 30 个设计点与 3 个中心重复点。';}catch(error){toast(error.message);}finally{setBusy(false,null);}
});
document.querySelector('.workspace-shell').addEventListener('dragover',event=>event.preventDefault());
document.querySelector('.workspace-shell').addEventListener('drop',event=>{event.preventDefault();importFile(event.dataTransfer.files[0]);});

async function main() {
  $('run-button').disabled=true;
  try {
    const response=await fetch('./assets/examples.json');if(!response.ok)throw new Error('示例数据读取失败');state.examples=await response.json();selectExample('battery');
    const e=state.examples.exchanger;
    scatter($('hero-chart'),{predicted:e.preview,observed:e.table.data.map(values=>Object.fromEntries(e.table.columns.map((c,i)=>[c,values[i]]))),x:'heat_W',y:'dp_Pa',height:270,interactive:false});
  }catch(error){$('run-status').textContent='页面资源未能加载，请通过网站地址打开或刷新重试。';toast(error.message);}
}
main();
