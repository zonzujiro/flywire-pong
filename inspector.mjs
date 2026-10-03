import {GROUPS,GROUP_NAMES,validateMetadata,decoderContributions,actionSensitivity,weightChange,connections} from './inspection.mjs';

const $=id=>document.getElementById(id);
const names=['UP','STAY','DOWN'];
const esc=value=>String(value??'Unknown').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=value=>{if(value===null||value===undefined||value==='')return 'Unknown';const numeric=Number(value);return Number.isFinite(numeric)?numeric===0?'0':Math.abs(numeric)<.00005?numeric.toExponential(1):numeric.toFixed(4):'Unknown';};
const signed=value=>value>0?'positive':value<0?'negative':'';
const color=(value,max=1)=>{
  const amount=Math.min(1,Math.abs(value)/Math.max(max,1e-12));
  const low=[52,69,74],high=value>=0?[127,226,208]:[255,156,134];
  return `rgb(${low.map((v,i)=>Math.round(v+(high[i]-v)*amount)).join(',')})`;
};

export class Inspector {
  static async load(next) {
    const [modelResponse,dataResponse]=await Promise.all([fetch('./model.json'),fetch('./inspector-data.json')]);
    if(!modelResponse.ok||!dataResponse.ok)throw Error('Could not load inspection metadata');
    const bytes=await modelResponse.arrayBuffer(),data=await dataResponse.json();
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
    if(digest!==data.model_sha256)throw Error('Inspector model hash mismatch');
    const model=JSON.parse(new TextDecoder().decode(bytes));validateMetadata(model,data);
    return new Inspector(model,data,next);
  }

  constructor(model,data,next) {
    this.model=model;this.data=data;this.selected=model.output_indices[0];this.step=12;this.baseline=0;this.mode='activity';this.filter='all';this.target=0;
    this.canvas=$('network-graph');this.ctx=this.canvas.getContext('2d');this.positions=[];
    this.deltas=weightChange(model,data);
    GROUPS.forEach((group,g)=>{
      const indices=data.nodes.flatMap((n,i)=>n.selection_group===group?[i]:[]);
      indices.forEach((n,j)=>this.positions[n]=[85+g*171,76+(j+.5)*580/indices.length]);
      const optgroup=document.createElement('optgroup');optgroup.label=GROUP_NAMES[g];
      indices.forEach(n=>{const option=document.createElement('option');option.value=n;option.textContent=this.label(n);optgroup.append(option);});
      $('neuron-select').append(optgroup);
    });
    data.baselines.forEach((b,i)=>{const option=document.createElement('option');option.value=i;option.textContent=b.label;$('weight-baseline').append(option);});
    model.output_indices.forEach((n,i)=>$('dn-name-'+i).textContent=data.nodes[n].cell_type||'Unknown');
    $('simulation-step').max=model.internal_steps;
    $('simulation-step').oninput=e=>this.setStep(Number(e.target.value));
    $('step-back').onclick=()=>this.setStep(Math.max(0,this.step-1));
    $('step-forward').onclick=()=>this.setStep(Math.min(model.internal_steps,this.step+1));
    $('next-decision').onclick=async()=>{$('next-decision').disabled=true;await next();};
    $('neuron-select').onchange=e=>this.select(Number(e.target.value));
    $('find-neuron').onclick=()=>this.find();
    $('neuron-search').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();this.find();}};
    $('color-mode').onchange=e=>{this.mode=e.target.value;$('influence-control').hidden=this.mode!=='influence';this.render();};
    $('influence-action').onchange=e=>{this.target=Number(e.target.value);this.render();};
    $('weight-baseline').onchange=e=>{this.baseline=Number(e.target.value);this.deltas=weightChange(model,data,this.baseline);this.render();this.renderConnections();this.renderWeightChanges();};
    $('graph-filter').onchange=e=>{this.filter=e.target.value;this.drawGraph();};
    $('graph-zoom').onchange=e=>{this.canvas.style.width=(100*Number(e.target.value))+'%';};
    this.canvas.onclick=e=>{
      const rect=this.canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*1200/rect.width,y=(e.clientY-rect.top)*700/rect.height;
      let closest=-1,distance=15;
      this.positions.forEach(([nx,ny],n)=>{const d=Math.hypot(x-nx,y-ny);if(this.visible(n)&&d<distance){closest=n;distance=d;}});
      if(closest>=0)this.select(closest);
    };
    $('connection-rows').onclick=e=>{const button=e.target.closest('button[data-neuron]');if(button)this.select(Number(button.dataset.neuron));};
    $('weight-rows').onclick=e=>{const button=e.target.closest('button[data-neuron]');if(button)this.select(Number(button.dataset.neuron));};
    $('download-decision').onclick=()=>this.download();
    this.renderConnections();this.renderWeightChanges();
  }

  label(n){const node=this.data.nodes[n];return `${node.cell_type||'Unknown type'} · ${node.root_id}`;}
  visible(n){const g=this.data.nodes[n].selection_group;return this.filter==='all'||(this.filter==='visual'&&GROUPS.indexOf(g)<5)||(this.filter==='descending'&&g==='descending');}
  select(n){this.selected=n;$('neuron-select').value=n;this.render();this.renderConnections();}
  find(){
    const query=$('neuron-search').value.trim().toLowerCase();
    if(!query){$('search-result').textContent='Enter a FlyWire ID or cell type.';return;}
    const matches=this.data.nodes.flatMap((n,i)=>n.root_id.includes(query)||(n.cell_type||'').toLowerCase().includes(query)?[i]:[]);
    if(matches.length){this.select(matches[0]);$('search-result').textContent=`${matches.length} match${matches.length===1?'':'es'}. Showing the first; use the neuron selector for others.`;}
    else $('search-result').textContent='No matching neuron in this selected network.';
  }

  show(snapshot){
    if(snapshot.checkpoint_sha256!==this.model.checkpoint_sha256)throw Error('Decision checkpoint mismatch');
    if(snapshot.states.length!==this.model.internal_steps+1||snapshot.states.some(s=>s.length!==this.model.neuron_count))throw Error('Invalid decision trace');
    if(snapshot.probabilities.some((p,i)=>p!==snapshot.chosen_probabilities[i]))throw Error('Inspection differs from actual decision');
    this.snapshot=snapshot;this.sensitivities=new Map();
    $('inspection-status').textContent=`Game seed ${snapshot.seed} · Decision ${snapshot.decision} · Sampled ${snapshot.action==='NOOP'?'STAY':snapshot.action} · Score before this action: fly ${snapshot.points_for}, Atari ${snapshot.points_against}. The game above shows the frame after that action.`;
    this.drawInput('previous-input',snapshot.previous_frame);this.drawInput('current-input',snapshot.input_frame);
    this.setStep(this.model.internal_steps);
  }

  drawInput(id,buffer){
    const canvas=$(id),rgb=new Uint8Array(buffer),rgba=new Uint8ClampedArray(160*210*4);
    for(let i=0,j=0;i<rgb.length;i+=3,j+=4){rgba[j]=rgb[i];rgba[j+1]=rgb[i+1];rgba[j+2]=rgb[i+2];rgba[j+3]=255;}
    const source=document.createElement('canvas');source.width=160;source.height=210;source.getContext('2d').putImageData(new ImageData(rgba,160,210),0,0);
    const [x0,y0,x1,y1]=this.model.vision.crop;canvas.getContext('2d').drawImage(source,x0,y0,x1-x0,y1-y0,0,0,canvas.width,canvas.height);
  }

  setStep(step){this.step=step;$('simulation-step').value=step;$('step-label').textContent=`${step} / ${this.model.internal_steps}`;$('step-back').disabled=step===0;$('step-forward').disabled=step===this.model.internal_steps;this.render();}
  gradients(){if(!this.sensitivities.has(this.target))this.sensitivities.set(this.target,actionSensitivity(this.model,this.snapshot.states,this.target));return this.sensitivities.get(this.target);}

  render(){if(!this.snapshot)return;this.drawGraph();this.renderDetails();this.renderReadout();this.drawHistory();}
  drawGraph(){
    if(!this.snapshot)return;
    const ctx=this.ctx,m=this.model,state=this.snapshot.states[this.step];
    const values=this.mode==='influence'?this.gradients()[this.step]:state;
    const max=this.mode==='influence'?Math.max(...Array.from(values,Math.abs)):1;
    const deltaMax=Math.max(...this.deltas.map(Math.abs));
    $('graph-legend').textContent=this.mode==='influence'?`Sensitivity to ${names[this.target]} score at update ${this.step}: teal = positive, coral = negative, gray = zero. Colors scaled to this update's maximum (${number(max)}).`:this.mode==='weights'?`Edges show trained minus baseline weight: teal = increase, coral = decrease; nodes show activity. Baseline: ${this.data.baselines[this.baseline].label}.`:'Neuron activity: teal = positive, coral = negative, gray = zero. Scale −1 to +1. Thin background lines are recorded connections.';
    ctx.clearRect(0,0,1200,700);
    const drawEdge=(e,highlight=false)=>{
      const pre=m.pre[e],post=m.post[e];if(!this.visible(pre)||!this.visible(post))return;
      const [x1,y1]=this.positions[pre],[x2,y2]=this.positions[post];
      ctx.strokeStyle=this.mode==='weights'?color(this.deltas[e],deltaMax):highlight?'#aac5c9':'#60767c';
      ctx.globalAlpha=highlight?.85:this.mode==='weights'?.24:.065;ctx.lineWidth=highlight?1.25:.6;
      ctx.beginPath();if(pre===post)ctx.arc(x1+6,y1-5,5,0,Math.PI*2);else{ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);}ctx.stroke();
      if(highlight&&pre!==post){const angle=Math.atan2(y2-y1,x2-x1),x=x2-6*Math.cos(angle),y=y2-6*Math.sin(angle);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-5*Math.cos(angle-.45),y-5*Math.sin(angle-.45));ctx.moveTo(x,y);ctx.lineTo(x-5*Math.cos(angle+.45),y-5*Math.sin(angle+.45));ctx.stroke();}
    };
    for(let e=0;e<m.edge_count;e++)drawEdge(e);
    connections(m,this.selected).forEach(e=>drawEdge(e,true));
    ctx.globalAlpha=1;
    this.positions.forEach(([x,y],n)=>{if(!this.visible(n))return;ctx.fillStyle=color(values[n],max);ctx.beginPath();ctx.arc(x,y,n===this.selected?6:2.3,0,2*Math.PI);ctx.fill();if(n===this.selected){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();}});
    ctx.fillStyle='#a4b3b9';ctx.font='15px system-ui';ctx.textAlign='center';
    GROUPS.forEach((g,i)=>{const count=this.data.nodes.filter(n=>n.selection_group===g).length;ctx.fillText(GROUP_NAMES[i],85+i*171,30);ctx.font='12px system-ui';ctx.fillText(`${count} neurons`,85+i*171,49);ctx.font='15px system-ui';});
  }

  renderDetails(){
    const n=this.selected,node=this.data.nodes[n],state=this.snapshot.states[this.step];$('neuron-select').value=n;
    $('selected-name').textContent=node.cell_type||'Unknown cell type';
    const regions=[...new Set(connections(this.model,n).map(e=>this.data.edge_regions[e]))].join(', ');
    const input=this.model.input_indices.indexOf(n),v=this.model.vision;
    const rows=[['FlyWire ID',node.root_id],['Source class',node.cell_class||node.super_class||'Unknown'],['Selection group',GROUP_NAMES[GROUPS.indexOf(node.selection_group)]],['Side',node.side||'Unknown'],['Activity',number(state[n])],['Synapse regions',regions||'None']];
    if(node.known_nt)rows.push(['Annotated transmitter',node.known_nt]);
    else if(node.top_nt)rows.push(['Predicted transmitter',`${node.top_nt}${node.top_nt_conf===null?'':` (confidence ${number(node.top_nt_conf)})`}`]);
    else rows.push(['Transmitter','Unknown']);
    if(input>=0){const mapping=this.data.visual_inputs[input];rows.push(['Visual column',`${mapping.column_id} — inferred via annotated lamina targets`],['Eye-grid coordinate',`(${mapping.p}, ${mapping.q})`],['Screen grid',`(${v.grid_x[input]}, ${v.grid_y[input]}) — engineering projection`],['Engineered channel',v.roles[input]],['Sensory input',number(this.snapshot.sensory[input])]);}
    if(this.mode==='influence')rows.push([`${names[this.target]} sensitivity`,number(this.gradients()[this.step][n])]);
    $('neuron-details').innerHTML=rows.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('');
    $('flywire-lookup').href='https://codex.flywire.ai/app/search?'+new URLSearchParams({dataset:'fafb',filter_string:node.root_id});
  }

  renderReadout(){
    const contributions=decoderContributions(this.model,this.snapshot.states[this.step]);
    $('readout-terms').innerHTML=contributions.map((c,a)=>`<tr><td>${names[a]}</td>${c.terms.map(v=>`<td class="${signed(v)}">${number(v)}</td>`).join('')}<td>${number(c.bias)}</td><td>${number(c.terms.reduce((sum,v)=>sum+v,c.bias))}</td></tr>`).join('');
  }

  drawHistory(){
    const canvas=$('neuron-history'),ctx=canvas.getContext('2d'),series=this.snapshot.states.map(s=>s[this.selected]);ctx.clearRect(0,0,480,150);
    ctx.strokeStyle='#455c63';ctx.beginPath();ctx.moveTo(25,75);ctx.lineTo(460,75);ctx.stroke();
    ctx.strokeStyle='#7fe2d0';ctx.lineWidth=2;ctx.beginPath();series.forEach((v,i)=>{const x=25+i*435/this.model.internal_steps,y=75-v*55;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    ctx.fillStyle='#edf4f3';ctx.beginPath();ctx.arc(25+this.step*435/this.model.internal_steps,75-series[this.step]*55,4,0,2*Math.PI);ctx.fill();ctx.font='12px system-ui';ctx.fillStyle='#a4b3b9';ctx.fillText('Activity −1 … +1',25,15);ctx.fillText('Update 0',25,142);ctx.fillText('Update '+this.model.internal_steps,395,142);
  }

  neuronButton(n){return `<button data-neuron="${n}">${esc(this.label(n))}</button>`;}
  renderConnections(){
    const edges=connections(this.model,this.selected).sort((a,b)=>Math.abs(this.deltas[b])-Math.abs(this.deltas[a]));
    $('connection-count').textContent=`(${edges.length} rows)`;
    $('connection-rows').innerHTML=edges.slice(0,100).map(e=>{const outgoing=this.model.pre[e]===this.selected,other=outgoing?this.model.post[e]:this.model.pre[e];return `<tr><td>${outgoing?'OUT →':'IN ←'} ${this.neuronButton(other)}</td><td>${esc(this.data.edge_regions[e])}</td><td>${this.data.edge_synapses[e]}</td><td>${number(this.data.baselines[this.baseline].weights[e])}</td><td>${number(this.model.weights[e])}</td><td class="${signed(this.deltas[e])}">${number(this.deltas[e])}</td></tr>`;}).join('');
    $('connection-limit').textContent=`Baseline: ${this.data.baselines[this.baseline].label}. ${edges.length>100?'Showing the 100 largest absolute changes; the graph retains all connections.':'All connection rows shown, sorted by absolute change.'}`;
  }

  renderWeightChanges(){
    const sorted=this.deltas.map((_,e)=>e).sort((a,b)=>Math.abs(this.deltas[b])-Math.abs(this.deltas[a]));
    $('weight-summary').textContent=`${this.deltas.filter(d=>d!==0).length.toLocaleString()} of ${this.model.edge_count.toLocaleString()} connection weights changed from ${this.data.baselines[this.baseline].label}. Showing the 20 largest changes. These are trained software weights, not measured biological strength changes.`;
    $('weight-rows').innerHTML=sorted.slice(0,20).map(e=>`<tr><td>${this.neuronButton(this.model.pre[e])} → ${this.neuronButton(this.model.post[e])}</td><td>${esc(this.data.edge_regions[e])}</td><td>${number(this.data.baselines[this.baseline].weights[e])}</td><td>${number(this.model.weights[e])}</td><td class="${signed(this.deltas[e])}">${number(this.deltas[e])}</td></tr>`).join('');
  }

  download(){
    const s=this.snapshot;
    const encodeFrame=buffer=>{const bytes=new Uint8Array(buffer);let text='';for(let i=0;i<bytes.length;i+=8192)text+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(text);};
    const result={format_version:1,checkpoint_sha256:s.checkpoint_sha256,seed:s.seed,decision:s.decision,action:s.action,probabilities:s.probabilities,logits:s.logits,
      frame_width:160,frame_height:210,frame_encoding:'base64 RGB uint8',previous_frame:encodeFrame(s.previous_frame),input_frame:encodeFrame(s.input_frame),
      root_ids:this.model.root_ids,sensory:s.sensory,input_indices:this.model.input_indices,states:s.states.map(a=>Array.from(a)),
      note:'State 0 is zero initialization; states 1–12 are synchronous artificial model updates. Input images precede the chosen action.'};
    const url=URL.createObjectURL(new Blob([JSON.stringify(result)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`flywire-decision-${s.seed}-${s.decision}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
}
