import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {gunzipSync} from 'node:zlib';
import {FlyPolicy,sampleAction,seededRandom} from '../network.mjs';
const raw=fs.readFileSync(new URL('../model.json',import.meta.url));
const fixturePath=process.argv[2] || new URL('./parity.json.gz',import.meta.url);
const fixtureBytes=fs.readFileSync(fixturePath);
const fixture=JSON.parse(String(fixturePath).endsWith('.gz')?gunzipSync(fixtureBytes):fixtureBytes);
assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),fixture.model_sha256);
const policy=new FlyPolicy(JSON.parse(raw));
let maxSensory=0,maxState=0,maxLogits=0,maxProb=0;
const error=(actual,expected)=>Math.max(...Array.from(actual,(v,i)=>Math.abs(v-expected[i])));
for(const c of fixture.cases) {
  const sensory=policy.encode(Buffer.from(c.previous,'base64'),Buffer.from(c.current,'base64'),c.frame_width);
  maxSensory=Math.max(maxSensory,error(sensory,c.sensory));
  const result=policy.forward(sensory);
  maxState=Math.max(maxState,error(policy.state,c.state));
  maxLogits=Math.max(maxLogits,error(result.logits,c.logits));
  maxProb=Math.max(maxProb,error(result.probabilities,c.probabilities));
}
assert.ok(maxSensory<2e-6,`Sensory mismatch ${maxSensory}`);
assert.ok(maxState<2e-5,`State mismatch ${maxState}`);
// Float32 tanh/dot kernels differ slightly between Torch and JavaScript.
assert.ok(maxLogits<1e-3,`Logit mismatch ${maxLogits}`);
assert.ok(maxProb<2e-4,`Probability mismatch ${maxProb}`);
const a=seededRandom(7),b=seededRandom(7);
for(let i=0;i<100;i++)assert.equal(a(),b());
assert.equal(sampleAction([1,0,0],()=>.5),0);
assert.equal(sampleAction([0,0,1],()=>.5),2);
console.log(JSON.stringify({passed:true,cases:fixture.cases.length,maxSensory,maxState,maxLogits,maxProb}));
