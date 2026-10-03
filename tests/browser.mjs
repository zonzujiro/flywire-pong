import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const url=process.env.SITE_URL||'http://127.0.0.1:8876/';
const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
try {
  const page=await browser.newPage({viewport:{width:1280,height:960}});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  await page.goto(url);
  await page.getByRole('button',{name:'Pause',exact:true}).waitFor({timeout:45000});
  await page.waitForFunction(()=>['UP','DOWN','STAY'].includes(document.getElementById('action').textContent));
  assert.match(await page.locator('#architecture').textContent(),/1,230 neurons.*8,515/);
  await page.getByRole('button',{name:'Pause',exact:true}).click();
  const paused=await page.locator('#board').evaluate(c=>c.toDataURL());
  await page.waitForTimeout(400);
  assert.equal(await page.locator('#board').evaluate(c=>c.toDataURL()),paused);
  const seed=await page.locator('#status').textContent();
  await page.getByRole('button',{name:'New game'}).click();
  await page.waitForFunction(old=>document.getElementById('status').textContent!==old,seed);
  await page.locator('#speed').selectOption('4');
  await page.getByRole('button',{name:'Play',exact:true}).click();
  await page.waitForFunction(()=>Number(document.getElementById('for').textContent)+Number(document.getElementById('against').textContent)>0,{},{timeout:45000});
  await page.getByRole('button',{name:'Pause',exact:true}).click();
  fs.mkdirSync('artifacts',{recursive:true});
  await page.screenshot({path:'artifacts/desktop.png',fullPage:true});
  const score={fly:Number(await page.locator('#for').textContent()),atari:Number(await page.locator('#against').textContent())};
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile page overflows');
  await page.screenshot({path:'artifacts/mobile.png',fullPage:true});
  assert.equal(await page.locator('#error').textContent(),'');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,url,score,checked:['WASM load','neural actions','pause','new game','speed','native points','mobile layout'],errors}));
} finally {await browser.close();}
