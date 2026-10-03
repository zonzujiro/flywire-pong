import assert from 'node:assert/strict';
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import crypto from 'node:crypto';
import {FlyPolicy} from '../network.mjs';
import {validateMetadata,actionSensitivity,decoderContributions,weightChange,connections} from '../inspection.mjs';

const raw=fs.readFileSync(new URL('../model.json',import.meta.url));
const model=JSON.parse(raw),data=JSON.parse(fs.readFileSync(new URL('../inspector-data.json',import.meta.url)));
const fixture=JSON.parse(gunzipSync(fs.readFileSync(new URL('./inspection.json.gz',import.meta.url))));
assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),fixture.model_sha256);
validateMetadata(model,data);
assert.throws(()=>validateMetadata(model,{...data,checkpoint_sha256:'wrong'}));
const wrong=structuredClone(data);wrong.nodes[0].root_id='1';assert.throws(()=>validateMetadata(model,wrong));
assert.ok(model.root_ids.every(id=>typeof id==='string'&&/^\d+$/.test(id)));
const policy=new FlyPolicy(model);
let maxStateError=0,maxRelativeGradientError=0;
for(const c of fixture.cases){
  const plain=policy.forward(c.sensory),trace=policy.forward(c.sensory,true);
  assert.deepEqual(trace.logits,plain.logits);assert.deepEqual(trace.probabilities,plain.probabilities);
  for(let t=0;t<trace.states.length;t++)for(let n=0;n<model.neuron_count;n++)maxStateError=Math.max(maxStateError,Math.abs(trace.states[t][n]-c.states[t][n]));
  for(const a of [0,2]){
    const gradients=actionSensitivity(model,trace.states,a);
    for(let t=0;t<gradients.length;t++)for(let n=0;n<model.neuron_count;n++){
      const expected=c.gradients[a][t][n];maxRelativeGradientError=Math.max(maxRelativeGradientError,Math.abs(gradients[t][n]-expected)/(1+Math.abs(expected)));
    }
  }
  const terms=decoderContributions(model,trace.states.at(-1));
  terms.forEach((c,a)=>assert.ok(Math.abs(c.bias+c.terms.reduce((s,v)=>s+v,0)-trace.logits[a])<1e-3));
}
assert.ok(maxStateError<2e-5,`State mismatch: ${maxStateError}`);
assert.ok(maxRelativeGradientError<2e-4,`Autograd mismatch: ${maxRelativeGradientError}`);
const fresh=weightChange(model,data,0),ppo=weightChange(model,data,1);
assert.equal(fresh.length,model.edge_count);assert.notDeepEqual(fresh,ppo);
const node=model.output_indices[0];assert.deepEqual(connections(model,node),model.pre.flatMap((pre,e)=>pre===node||model.post[e]===node?[e]:[]));

// Independent finite-difference check on a graph with a negative edge and feedback.
const toy={neuron_count:3,edge_count:3,pre:[0,1,2],post:[1,2,1],weights:[.3,-.4,.2],leak:.2,internal_steps:3,output_indices:[2],action_weight:[[.7],[-.2],[.4]],action_bias:[0,0,0]};
const update=state=>{const next=state.map(x=>toy.leak*x);for(let e=0;e<3;e++)next[toy.post[e]]+=state[toy.pre[e]]*toy.weights[e];next[0]+=.5;return next.map(Math.tanh);};
const states=[[0,0,0]];for(let t=0;t<3;t++)states.push(update(states.at(-1)));
const derivative=actionSensitivity(toy,states,0),eps=1e-6;
for(let t=0;t<=3;t++)for(let n=0;n<3;n++){
  const score=offset=>{let s=states[t].slice();s[n]+=offset;for(let k=t;k<3;k++)s=update(s);return s[2]*(.7-(-.2+.4)/2);};
  assert.ok(Math.abs((score(eps)-score(-eps))/(2*eps)-derivative[t][n])<1e-8);
}
console.log(JSON.stringify({passed:true,python_cases:fixture.cases.length,maxStateError,maxRelativeGradientError,finite_difference:true,metadata_identity:true,tracing_preserves_inference:true}));
