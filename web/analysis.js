/* Scientific views derived exclusively from the current engine output. */
const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const fmt=value=>!finite(value)?'—':value===0?'0':Math.abs(value)<.0001?value.toExponential(2):Number(value.toPrecision(5)).toLocaleString('en-US',{maximumFractionDigits:5});
const blank=message=>`<div class="analysis-empty"><span>—</span><p>${esc(message)}</p></div>`;
let current;
function cards(id,items){$(id).innerHTML=items.map(([name,value,note])=>`<div class="analysis-scorecard"><span>${esc(name)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`).join('');}
function table(id,rows,columns,empty='完成计算后显示本次记录。'){
  if(!rows?.length){$(id).innerHTML=`<p class="table-empty-note">${esc(empty)}</p>`;return;}
  const keys=columns||Object.keys(rows[0]);
  const names={case:'样本标识',_source_row:'原始行号',_data_role:'数据用途',seed:'随机种子',pareto_count:'候选数量',utility:'推荐效用'};
  $(id).innerHTML=`<table><thead><tr>${keys.map(k=>`<th scope="col">${esc(names[k]||k)}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,100).map(row=>`<tr>${keys.map(k=>`<td>${esc(typeof row[k]==='number'?fmt(row[k]):typeof row[k]==='boolean'?(row[k]?'是':'否'):row[k]??'—')}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function pointPlot(id,records,{x,y,xlabel,ylabel,identity=false,zero=false,nonnegative=false}){
  const all=records.filter(r=>finite(r[x])&&finite(r[y]));
  if(!all.length){$(id).innerHTML=blank('尚无可绘制的本次计算记录。');return;}
  const W=600,H=290,L=72,R=24,T=20,B=56;
  let xmin=Math.min(...all.map(r=>r[x])),xmax=Math.max(...all.map(r=>r[x]));
  let ymin=Math.min(...all.map(r=>r[y])),ymax=Math.max(...all.map(r=>r[y]));
  if(identity){xmin=ymin=Math.min(xmin,ymin);xmax=ymax=Math.max(xmax,ymax);}
  if(zero){ymin=Math.min(0,ymin);ymax=Math.max(0,ymax);}
  const padx=(xmax-xmin||Math.max(Math.abs(xmin)*.05,1))*.08;
  const pady=(ymax-ymin||Math.max(Math.abs(ymin)*.05,1))*.1;
  xmin-=padx;xmax+=padx;ymin-=pady;ymax+=pady;
  if(nonnegative){xmin=Math.max(0,xmin);ymin=Math.max(0,ymin);}
  const X=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),Y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);
  let grid='';
  for(let i=0;i<=4;i++){
    const xx=L+(W-L-R)*i/4,yy=T+(H-T-B)*i/4;
    grid+=`<path d="M${L} ${yy}H${W-R}" stroke="#e7edf4"/><text x="${L-12}" y="${yy+4}" text-anchor="end">${esc(fmt(ymax-(ymax-ymin)*i/4))}</text><text x="${xx}" y="${H-B+23}" text-anchor="middle">${esc(fmt(xmin+(xmax-xmin)*i/4))}</text>`;
  }
  const reference=identity?`<path d="M${X(Math.max(xmin,ymin))} ${Y(Math.max(xmin,ymin))}L${X(Math.min(xmax,ymax))} ${Y(Math.min(xmax,ymax))}" stroke="#8496ad" stroke-dasharray="5 5"/>`:zero?`<path d="M${L} ${Y(0)}H${W-R}" stroke="#8496ad" stroke-dasharray="5 5"/>`:'';
  const points=all.slice(0,2000).map(r=>`<circle cx="${X(r[x])}" cy="${Y(r[y])}" r="${r.role==='holdout'?4:3}" fill="${r.role==='holdout'?'#c78546':'#1672ce'}" fill-opacity=".7" stroke="white" stroke-width=".5"><title>${esc((r.role==='holdout'?'独立留出':'本次记录')+` · ${xlabel}: ${fmt(r[x])} · ${ylabel}: ${fmt(r[y])}`)}</title></circle>`).join('');
  $(id).innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(xlabel+'与'+ylabel)}"><g font-family="Arial,sans-serif" fill="#6b7b91" font-size="11">${grid}<path d="M${L} ${T}V${H-B}H${W-R}" fill="none" stroke="#cdd8e6"/>${reference}${points}<text x="${(W+L-R)/2}" y="${H-8}" text-anchor="middle">${esc(xlabel)}</text><text transform="translate(16 ${(H+T-B)/2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text></g></svg>`;
}
export function renderProblemSnapshot(problem,rows,result,busy){
  const mode={shuffled:'随机 K 折',group:'分组验证',time:'时间前向验证'}[problem.validation]||problem.validation;
  const items=[['设计变量',`${problem.variables.length} 个`,problem.variables.map(v=>v.kind==='integer'?'整数':v.kind==='discrete'?'集合':'连续').filter((v,i,a)=>a.indexOf(v)===i).join(' / ')],['研究响应',`${problem.objectives.length} 个目标`,`${problem.responses.length} 个辅助响应 · ${problem.constraints.length} 项约束`],['模型与验证',problem.model,`${mode} · ${problem.folds} 折`],['搜索设置',`${problem.population} × ${problem.generations}`,`${problem.runs} 次运行 · 种子 ${problem.seed}`]];
  $('problem-snapshot').innerHTML=items.map(([name,value,note])=>`<div><span>${esc(name)}</span><b>${esc(value)}</b><small>${esc(note)}</small></div>`).join('')+`<div class="snapshot-status"><span class="status-dot ${result?'complete':busy?'working':''}"></span><b>${busy?'计算中':result?'本次计算完成':'配置 / 示例'}</b><small>${rows.length} 个完整输入样本</small></div>`;
}
function parallelPlot(problem,rows,result){
  const selected=rows.slice(0,8),targets=problem.objectives;
  if(!selected.length||!targets.length){$('parallel-chart').innerHTML=blank('完成优化后，对照多个目标的权衡。');$('parallel-caption').textContent='';return;}
  const raw=current.rows;
  const lows=result?.config.anchor_minimum||targets.map(t=>Math.min(...raw.map(r=>Number(r[t.column])).filter(Number.isFinite)));
  const highs=result?.config.anchor_maximum||targets.map(t=>Math.max(...raw.map(r=>Number(r[t.column])).filter(Number.isFinite)));
  const benefit=(row,j)=>targets[j].direction==='max'?(row[targets[j].column]-lows[j])/(highs[j]-lows[j]||1):(highs[j]-row[targets[j].column])/(highs[j]-lows[j]||1);
  const values=selected.flatMap(row=>targets.map((_,j)=>benefit(row,j))).filter(finite);
  if(!values.length){$('parallel-chart').innerHTML=blank('这些响应尚无有效数值。');return;}
  const low=Math.min(0,...values),high=Math.max(1,...values),W=Math.max(700,targets.length*130),H=265,L=65,R=55,T=27,B=62;
  const X=j=>targets.length===1?W/2:L+j*(W-L-R)/(targets.length-1),Y=v=>H-B-(v-low)/(high-low)*(H-T-B);
  let axes='';
  for(let i=0;i<=4;i++){const v=low+(high-low)*i/4;axes+=`<path d="M${L} ${Y(v)}H${W-R}" stroke="#edf1f6"/><text x="${L-13}" y="${Y(v)+4}" text-anchor="end">${fmt(v)}</text>`;}
  axes+=targets.map((t,j)=>`<path d="M${X(j)} ${T}V${H-B}" stroke="#d4deea"/><text x="${X(j)}" y="${H-B+24}" text-anchor="middle">${esc(t.column)}</text><text x="${X(j)}" y="${H-B+42}" text-anchor="middle" fill="#8995a5">${esc((t.direction==='min'?'↓ 最小化':'↑ 最大化')+(t.unit?' · '+t.unit:''))}</text>`).join('');
  const lines=selected.map((row,i)=>{const points=targets.map((_,j)=>`${X(j)},${Y(benefit(row,j))}`).join(' ');return `<polyline points="${points}" fill="none" stroke="${i===0?'#0a66c2':'#8cadcf'}" stroke-opacity="${i===0?1:.42}" stroke-width="${i===0?2.6:1.4}"><title>${esc(`推荐排序 ${row['推荐排序']??i+1}`)}</title></polyline>${i===0?targets.map((_,j)=>`<circle cx="${X(j)}" cy="${Y(benefit(row,j))}" r="4" fill="#0a66c2"/>`).join(''):''}`;}).reverse().join('');
  $('parallel-chart').innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="前八个候选的多目标相对表现"><g fill="#6b7b91" font-size="11" font-family="Arial,sans-serif">${axes}${lines}</g></svg>`;
  $('parallel-caption').textContent=`显示前 ${selected.length} 个${result?'本次预测':'合成预览'}候选，深蓝为首选。数值按${result?'冻结训练':'完整示例输入'}范围转换为相对表现，越高越好；超出 0–1 的改善保留原值。此图不是置信度或额外排序指标。`;
}
function renderValidation(){
  const result=current?.result,target=$('validation-target').value;
  if(!result){cards('validation-scorecards',[['CV R²','—','运行后显示折外检验'],['CV RMSE','—','基于当前响应'],['CV MAE','—','基于当前响应'],['独立留出','—','请在配置中指定留出']]);['validation-chart','residual-chart','model-comparison-chart','holdout-evidence'].forEach(id=>$(id).innerHTML=blank('完成一次计算后显示真实检验记录。'));return;}
  const metric=result.metrics.find(r=>r['目标']===target),unit=[...current.problem.objectives,...current.problem.responses].find(t=>t.column===target)?.unit||'';
  const cv=result.validation.filter(r=>r['目标']===target&&r['用于验证']!==false&&finite(r['实测值'])&&finite(r['交叉验证预测值'])).map(r=>({truth:r['实测值'],prediction:r['交叉验证预测值'],role:'cv'}));
  const holdout=result.holdout.filter(r=>r['目标']===target&&finite(r['实测值'])&&finite(r['冻结模型预测值'])).map(r=>({truth:r['实测值'],prediction:r['冻结模型预测值'],role:'holdout'}));
  cards('validation-scorecards',[['CV R²',fmt(metric?.['交叉验证R2']),`${cv.length} 个折外检验记录`],['CV RMSE',fmt(metric?.['交叉验证RMSE']),unit||'当前响应的原始单位'],['CV MAE',fmt(metric?.['交叉验证MAE']),metric?.['选用模型']||'—'],['独立留出',String(holdout.length),'橙色点 · 不参与训练与选型']]);
  const records=[...cv,...holdout];
  pointPlot('validation-chart',records,{x:'truth',y:'prediction',xlabel:`实测 ${target}`,ylabel:'模型预测',identity:true});
  pointPlot('residual-chart',records.map(r=>({...r,residual:r.prediction-r.truth})),{x:'truth',y:'residual',xlabel:`实测 ${target}`,ylabel:`残差${unit?' / '+unit:''}`,zero:true});
  const comparisons=result.model_comparison.filter(r=>r['目标']===target&&finite(r['归一化RMSE'])).sort((a,b)=>a['归一化RMSE']-b['归一化RMSE']);
  const max=Math.max(0,...comparisons.map(r=>r['归一化RMSE']));
  $('model-comparison-chart').innerHTML=comparisons.length?`<div class="model-bars">${comparisons.map(r=>`<div class="model-bar ${r['模型']===metric?.['选用模型']?'selected':''}"><span>${esc(r['模型'])}${r['模型']===metric?.['选用模型']?'<small>当前模型</small>':''}</span><div><i style="width:${max?100*r['归一化RMSE']/max:0}%"></i></div><b>${esc(fmt(r['归一化RMSE']))}</b></div>`).join('')}</div><p class="analysis-caption">同一响应内按 CV NRMSE 比较；${comparisons.length===1?'当前设置仅评估一个模型，选择 Auto 可比较全部模型。':'最小验证误差用于选型，不能作为独立泛化误差。'}</p>`:blank('没有可用模型比较记录。');
  const independent=result.holdout_metrics.find(r=>r['目标']===target);
  $('holdout-evidence').innerHTML=independent?`<div class="holdout-callout"><b>${esc(target)} · 独立留出 ${esc(independent['留出数'])} 个</b><span>RMSE ${esc(fmt(independent['留出RMSE']))}${esc(unit?' '+unit:'')} · MAE ${esc(fmt(independent['留出MAE']))} · 训练域内 ${esc(independent['训练域内数'])} 个</span></div>`:`<p class="table-empty-note">本次没有独立留出检验。可在“模型与验证”中指定留出批次；CV 报告不能代替独立检验。</p>`;
}
function renderDiagnostics(){
  const result=current.result;
  if(!result){cards('diagnostic-scorecards',[['输入表可行','—','根据实测响应判定'],['预测候选','—','根据代理响应判定'],['数据清理','—','查看处理记录'],['独立运行','—','按随机种子比较']]);['coverage-chart','constraint-table','cleaning-table','run-comparison-chart','run-table','preference-table'].forEach(id=>$(id).innerHTML=blank('完成计算后查看本次研究诊断。'));$('run-analysis-label').textContent='等待计算';$('run-interpretation').textContent='';return;}
  const accepted=result.constraints.filter(r=>r['可行']===true).length;
  const feasible=result.predicted.filter(r=>finite(r['约束违反量'])&&r['约束违反量']<=1e-10).length;
  cards('diagnostic-scorecards',[['输入表可行',`${accepted} / ${result.constraints.length}`,'实测响应 + 搜索域检查'],['预测候选可行',`${feasible} / ${result.predicted.length}`,'基于代理模型，仍需复核'],['数据清理记录',String(result.cleaning.length),'剔除与重复设计处理'],['独立运行',String(result.runs.length),'查看每个种子的推荐效用']]);
  pointPlot('coverage-chart',result.predicted,{x:'推荐排序',y:'最近样本距离',xlabel:'当前偏好排序',ylabel:'最近样本距离',zero:true,nonnegative:true});
  table('constraint-table',result.constraints,['case','_source_row','_data_role','搜索域内','最近样本距离','约束违反量','可行',...Object.keys(result.constraints[0]||{}).filter(k=>/^约束\d/.test(k))],'本次输入表没有诊断记录。');
  table('cleaning-table',result.cleaning,null,'本次没有需要单独记录的数据清理项。重复设计的处理说明可在“结果追溯”查看。');
  pointPlot('run-comparison-chart',result.runs,{x:'seed',y:'utility',xlabel:'随机种子',ylabel:'单次推荐效用'});
  table('run-table',result.runs,['seed','pareto_count','utility',...current.problem.variables.map(v=>v.column),...current.problem.objectives.map(t=>t.column)]);
  $('run-analysis-label').textContent=`${result.runs.length} 个实际随机种子`;
  $('run-interpretation').textContent=result.runs.length<2?'本次仅运行一次，不能据此判断跨种子的稳定性；将独立运行次数设为 2–5 可进行比较。':'各次使用同一问题、代理模型与冻结训练尺度，仅随机种子不同。效用与候选数反映本次有限预算的结果，不构成全局最优保证。';
  table('preference-table',result.sensitivity,['结果类型','偏好',...current.problem.objectives.map(t=>t.column+' 权重'),...current.problem.objectives.map(t=>t.column),'理想点距离']);
}
export function renderScientificPanels(problem,rows,result,preview){
  current={problem,rows,result};
  const names=result?.metrics.map(r=>r['目标'])||[...problem.objectives,...problem.responses].map(t=>t.column),previous=$('validation-target').value;
  $('validation-target').innerHTML=names.map(name=>`<option value="${esc(name)}">${esc(name)}</option>`).join('');
  if(names.includes(previous))$('validation-target').value=previous;
  parallelPlot(problem,result?.predicted||preview,result);renderValidation();renderDiagnostics();
}
$('validation-target').addEventListener('change',renderValidation);
