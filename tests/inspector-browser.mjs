import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const url=process.env.SITE_URL||'http://127.0.0.1:8876/';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1080}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  await page.goto(url);await page.locator('#inspect:enabled').waitFor({timeout:45000});
  await page.getByRole('button',{name:'Inspect network',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('inspection-status').textContent.includes('Sampled')).catch(async error=>{console.error(await page.locator('#inspection-status').textContent(),errors);throw error;});
  assert.equal(await page.locator('#toggle').textContent(),'Play');
  assert.equal(await page.locator('#neuron-select option').count(),1230);
  await page.locator('#simulation-step').fill('0');assert.equal(await page.locator('#step-label').textContent(),'0 / 12');
  assert.match(await page.locator('#neuron-details').textContent(),/Activity0/);
  await page.getByRole('button',{name:'Next update',exact:true}).click();assert.equal(await page.locator('#step-label').textContent(),'1 / 12');
  await page.locator('#simulation-step').fill('6');
  await page.locator('#color-mode').selectOption('influence');await page.locator('#influence-action').selectOption('2');
  assert.match(await page.locator('#graph-legend').textContent(),/Sensitivity to DOWN/);
  await page.locator('#neuron-search').fill('DNp15');await page.getByRole('button',{name:'Find neuron',exact:true}).click();
  assert.equal(await page.locator('#selected-name').textContent(),'DNp15');
  assert.match(await page.locator('#flywire-lookup').getAttribute('href'),/filter_string=720575940623431351/);
  assert.ok(await page.locator('#connection-rows tr').count()>0);
  await page.locator('#color-mode').selectOption('weights');await page.locator('#weight-baseline').selectOption('1');
  assert.match(await page.locator('#graph-legend').textContent(),/Before final PPO/);
  await page.locator('#graph-filter').selectOption('visual');await page.locator('#graph-zoom').selectOption('2');
  await page.locator('#graph-filter').selectOption('all');await page.locator('#graph-zoom').selectOption('1');
  const decision=await page.locator('#inspection-status').textContent();
  await page.getByRole('button',{name:'Next game decision',exact:true}).click();
  await page.waitForFunction(old=>document.getElementById('inspection-status').textContent.includes('Sampled')&&document.getElementById('inspection-status').textContent!==old,decision);
  const downloadPromise=page.waitForEvent('download');await page.locator('#download-decision').click();
  const download=await downloadPromise;fs.mkdirSync('artifacts',{recursive:true});await download.saveAs('artifacts/decision.json');
  const exported=JSON.parse(fs.readFileSync('artifacts/decision.json'));
  assert.equal(exported.states.length,13);assert.equal(exported.states[0].length,1230);
  assert.ok(exported.states[0].every(v=>v===0));assert.ok(exported.root_ids.every(v=>typeof v==='string'));
  assert.equal(Buffer.from(exported.input_frame,'base64').length,160*210*3);
  await page.locator('#color-mode').selectOption('activity');await page.locator('#simulation-step').fill('6');
  await page.locator('#inspector').screenshot({path:'artifacts/inspector-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile inspector overflows');
  await page.locator('#inspector').screenshot({path:'artifacts/inspector-mobile.png'});
  await page.getByRole('button',{name:'Play',exact:true}).click();assert.ok(await page.locator('#inspector').isHidden());
  await page.waitForFunction(()=>document.getElementById('status').textContent.includes('Playing'));

  // Same game actions/frames with and without repeated inspection requests.
  const invariant=await page.evaluate(async()=>{
    const workers=[new Worker(new URL('./game-worker.js',location.href)),new Worker(new URL('./game-worker.js',location.href))];
    try {
      const rpc=(worker,type,steps=8)=>new Promise((resolve,reject)=>{worker.onmessage=({data})=>data.error?reject(Error(data.error)):resolve(data.inspection||data.state);worker.postMessage({id:1,type,steps});});
      let states=await Promise.all(workers.map(w=>rpc(w,'init')));
      const equal=(a,b)=>{
        if(a.seed!==b.seed||a.decisions!==b.decisions||a.action!==b.action||a.done!==b.done||a.points_for!==b.points_for||a.points_against!==b.points_against||JSON.stringify(a.probabilities)!==JSON.stringify(b.probabilities))throw Error('Inspection changed game state');
        const aa=new Uint8Array(a.frame),bb=new Uint8Array(b.frame);if(aa.some((v,i)=>v!==bb[i]))throw Error('Inspection changed game frame');
      };
      for(let i=0;i<40;i++){
        states=await Promise.all(workers.map(w=>rpc(w,'step')));equal(...states);
        const snapshot=await rpc(workers[1],'inspect');
        if(JSON.stringify(snapshot.probabilities)!==JSON.stringify(states[1].probabilities))throw Error('Inspection probability mismatch');
        if(snapshot.decision!==states[1].decisions)throw Error('Inspection decision mismatch');
      }
      const decisions=states[0].decisions;
      await rpc(workers[1],'reset');
      let resetClearsTrace=false;try{await rpc(workers[1],'inspect');}catch{resetClearsTrace=true;}
      if(!resetClearsTrace)throw Error('Reset retained old inspection');
      return {decisions,identical:true,resetClearsTrace};
    } finally{workers.forEach(w=>w.terminate());}
  });
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,url,neuron_count:1230,updates:12,downloaded_trace:true,mobile:true,invariant,errors}));
} finally{await browser.close();}
