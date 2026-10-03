const $=id=>document.getElementById(id),ctx=$('board').getContext('2d');
const source=document.createElement('canvas');source.width=160;source.height=210;
const sourceCtx=source.getContext('2d');ctx.imageSmoothingEnabled=false;
const worker=new Worker(new URL('./game-worker.js',import.meta.url));
let running=true,busy=false,latest=null,nextGameAt=0,pendingReset=false,sequence=0;
let inspector=null,inspectionGeneration=0;
const pending=new Map();
function closeInspector(){inspectionGeneration++;$('inspector').hidden=true;}
function pause(value){running=value;if(value)closeInspector();$('toggle').textContent=running?'Pause':'Play';if(latest)status();}
function fail(message){pause(false);$('error').textContent='Could not run Pong: '+message+'. Try reloading in a browser with WebAssembly support.';}
worker.onmessage=({data})=>{const request=pending.get(data.id);if(!request)return;pending.delete(data.id);data.error?request.reject(Error(data.error)):request.resolve(data.inspection||data.state);};
worker.onerror=error=>{fail(error.message);for(const p of pending.values())p.reject(Error(error.message));pending.clear();};
function status(){
  const s=latest;
  $('status').textContent=s.done?(s.outcome==='win'?'Fly network wins!':s.outcome==='loss'?'Atari opponent wins.':'Game ended early.'):(running?'Playing':'Paused')+' · Game seed '+s.seed;
}
function draw(s){
  latest=s;const rgb=new Uint8Array(s.frame),rgba=new Uint8ClampedArray(160*210*4);
  for(let i=0,j=0;i<rgb.length;i+=3,j+=4){rgba[j]=rgb[i];rgba[j+1]=rgb[i+1];rgba[j+2]=rgb[i+2];rgba[j+3]=255;}
  sourceCtx.putImageData(new ImageData(rgba,160,210),0,0);
  ctx.drawImage(source,0,34,160,160,0,0,160,160);
  $('for').textContent=s.points_for;$('against').textContent=s.points_against;
  $('wins').textContent=s.wins;$('losses').textContent=s.losses;
  $('action').textContent=s.action==='NOOP'?'STAY':s.action||'READY';
  s.probabilities.forEach((v,i)=>{$('p'+i).textContent=Math.round(v*100)+'%';$('b'+i).style.width=(v*100)+'%';});
  document.querySelectorAll('.prob').forEach(e=>e.classList.toggle('chosen',e.dataset.action===s.action));
  $('architecture').textContent=s.neurons.toLocaleString()+' neurons · '+s.edges.toLocaleString()+' directed connections';status();
  $('inspect').disabled=s.decisions===0;
}
function rpc(type,steps=1){const id=++sequence;const promise=new Promise((resolve,reject)=>pending.set(id,{resolve,reject}));worker.postMessage({id,type,steps});return promise;}
async function request(type,steps=1){
  busy=true;
  try{const state=await rpc(type,steps);draw(state);return state;}
  catch(error){fail(error.message);}
  finally{busy=false;}
}
$('toggle').onclick=()=>pause(!running);
$('reset').onclick=()=>{closeInspector();pendingReset=true;};
document.addEventListener('keydown',e=>{if(['SELECT','BUTTON','INPUT','TEXTAREA'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();pause(!running);}if(e.key.toLowerCase()==='r'){closeInspector();pendingReset=true;}});
$('inspect').onclick=()=>showInspection();
$('close-inspector').onclick=closeInspector;
async function showInspection(){
  pause(false);$('inspector').hidden=false;
  $('inspector').scrollIntoView({behavior:'smooth',block:'start'});
  const generation=++inspectionGeneration;
  $('inspection-status').textContent='Loading the recorded decision…';
  try {
    while(busy)await new Promise(resolve=>setTimeout(resolve,10));
    if(generation!==inspectionGeneration)return;
    busy=true;
    if(!inspector){const {Inspector}=await import('./inspector.mjs');inspector=await Inspector.load(async()=>{
      if(busy||!latest||latest.done)return;
      await request('step');await showInspection();
    });}
    const snapshot=await rpc('inspect');
    if(generation===inspectionGeneration){inspector.show(snapshot);$('next-decision').disabled=latest.done;}
  }catch(error){if(generation===inspectionGeneration)$('inspection-status').textContent='Inspector unavailable: '+error.message;}
  finally{busy=false;}
}
// Hidden tabs stop advancing the game; the page never needs a remote backend.
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause(false);});
async function tick(){
  const start=performance.now(),speed=Number($('speed').value);
  if(pendingReset&&!busy&&latest){pendingReset=false;nextGameAt=0;await request('reset');}
  else if(running&&!busy&&latest&&!document.hidden){
    if(latest.done){if(!nextGameAt)nextGameAt=Date.now()+2200;if(Date.now()>=nextGameAt){nextGameAt=0;await request('reset');}}
    else await request('step',Math.max(1,speed));
  }
  setTimeout(tick,Math.max(1,(speed<1?1000/15:1000/30)-(performance.now()-start)));
}
await request('init');
if(latest){$('toggle').disabled=$('reset').disabled=$('speed').disabled=false;pause(true);tick();}
