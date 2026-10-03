// Diagnostics on an immutable recorded decision; these never choose game actions.
export const GROUPS=['photoreceptor','lamina','medulla','motion','visual_projection','central','descending'];
export const GROUP_NAMES=['Light inputs','Lamina','Medulla','Motion cells','Visual projection','Central','Descending'];

export function validateMetadata(model,data) {
  if(data.format_version!==1||data.checkpoint_sha256!==model.checkpoint_sha256)throw Error('Inspector checkpoint mismatch');
  if(data.nodes.length!==model.neuron_count||data.edge_synapses.length!==model.edge_count||data.edge_regions.length!==model.edge_count)throw Error('Inspector graph size mismatch');
  data.nodes.forEach((node,i)=>{if(node.root_id!==model.root_ids[i]||!GROUPS.includes(node.selection_group))throw Error('Inspector neuron identity mismatch');});
  if(!data.baselines.length||data.baselines.some(b=>b.weights.length!==model.edge_count||b.weights.some(w=>!Number.isFinite(w))))throw Error('Invalid weight baseline');
  if(data.visual_inputs.length!==model.input_indices.length||data.visual_inputs.some((v,i)=>v.neuron_index!==model.input_indices[i]||v.root_id!==model.root_ids[v.neuron_index]))throw Error('Inspector visual mapping mismatch');
}

export function decoderContributions(model,state) {
  return model.action_weight.map((weights,a)=>({bias:model.action_bias[a],
    terms:model.output_indices.map((n,i)=>state[n]*weights[i])}));
}

// Derivative of the target logit minus the mean of the other two logits.
// Float rounding is treated as identity, as in ordinary autodiff.
export function actionSensitivity(model,states,target) {
  if(!Number.isInteger(target)||target<0||target>2||states.length!==model.internal_steps+1)throw Error('Invalid influence request');
  const gradients=states.map(s=>new Float64Array(s.length));
  model.output_indices.forEach((n,i)=>{
    gradients.at(-1)[n]=model.action_weight[target][i]-model.action_weight.reduce((sum,w,a)=>sum+(a===target?0:w[i]/2),0);
  });
  for(let t=states.length-2;t>=0;t--) {
    const derivative=Float64Array.from(gradients[t+1],(g,n)=>g*(1-states[t+1][n]**2));
    for(let n=0;n<derivative.length;n++)gradients[t][n]=model.leak*derivative[n];
    for(let e=0;e<model.edge_count;e++)gradients[t][model.pre[e]]+=model.weights[e]*derivative[model.post[e]];
  }
  return gradients;
}

export function weightChange(model,data,baseline=0) {
  if(!data.baselines[baseline])throw Error('Invalid baseline');
  return model.weights.map((w,e)=>w-data.baselines[baseline].weights[e]);
}

export function connections(model,node) {
  return model.pre.flatMap((pre,e)=>pre===node||model.post[e]===node?[e]:[]);
}
