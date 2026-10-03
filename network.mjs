// Direct sparse inference. Every edge is exported from the saved FlyWire graph.
const f = Math.fround;
const clip = x => Math.max(-1, Math.min(1, x));

// Match NumPy's float32 pairwise sum order for image means.
function floatSum(a,start=0,n=a.length) {
  if(n<8){let sum=-0;for(let i=0;i<n;i++)sum=f(sum+a[start+i]);return sum;}
  if(n<=128){
    const r=Array.from(a.subarray(start,start+8));let i=8;
    for(;i<n-n%8;i+=8)for(let j=0;j<8;j++)r[j]=f(r[j]+a[start+i+j]);
    let sum=f(f(f(r[0]+r[1])+f(r[2]+r[3]))+f(f(r[4]+r[5])+f(r[6]+r[7])));
    for(;i<n;i++)sum=f(sum+a[start+i]);return sum;
  }
  let half=Math.floor(n/2);half-=half%8;
  return f(floatSum(a,start,half)+floatSum(a,start+half,n-half));
}

export class FlyPolicy {
  constructor(model) {
    if (model.format_version !== 1) throw Error('Unsupported network export');
    this.m = model;
    this.v = model.vision;
    this.pre = Int32Array.from(model.pre);
    this.post = Int32Array.from(model.post);
    this.weights = Float32Array.from(model.weights);
    this.bias = Float32Array.from(model.biases);
    this.state = new Float32Array(model.neuron_count);
    this.next = new Float32Array(model.neuron_count);
    this.incoming = new Float32Array(model.neuron_count);
    this.fx = this.v.field_x.map(a => Float32Array.from(a));
    this.fy = this.v.field_y.map(a => Float32Array.from(a));
  }

  gray(rgb, width) {
    const [x0,y0,x1,y1] = this.v.crop, w=x1-x0, h=y1-y0;
    const gray = new Float32Array(w*h);
    for (let y=0;y<h;y++) for(let x=0;x<w;x++) {
      const i=((y+y0)*width+x+x0)*3;
      // Pillow RGB -> L fixed-point coefficients and rounding.
      gray[y*w+x]=f(((19595*rgb[i]+38470*rgb[i+1]+7471*rgb[i+2]+32768)>>16)/255);
    }
    return gray;
  }

  pool(rgb,width) {
    const gray=this.gray(rgb,width), [x0,y0,x1,y1]=this.v.crop;
    const w=x1-x0,h=y1-y0, gw=this.v.width,gh=this.v.height;
    const background=f(floatSum(gray)/gray.length);
    const pos=new Float32Array(gw*h),neg=new Float32Array(gw*h);
    for(let gx=0;gx<gw;gx++) for(let y=0;y<h;y++) {
      let p=0,n=0;
      for(let x=0;x<w;x++) {
        const difference=f(gray[y*w+x]-background),factor=this.fx[gx][x];
        p=Math.max(p,f(Math.max(difference,0)*factor));
        n=Math.max(n,f(Math.max(-difference,0)*factor));
      }
      pos[gx*h+y]=p;neg[gx*h+y]=n;
    }
    const result=new Float32Array(gw*gh);
    for(let gy=0;gy<gh;gy++) for(let gx=0;gx<gw;gx++) {
      let p=0,n=0;
      for(let y=0;y<h;y++) {
        p=Math.max(p,f(pos[gx*h+y]*this.fy[gy][y]));
        n=Math.max(n,f(neg[gx*h+y]*this.fy[gy][y]));
      }
      result[gy*gw+gx]=Math.max(0,Math.min(1,f(f(background+p)-n)));
    }
    const mean=f(floatSum(result)/result.length);
    for(let i=0;i<result.length;i++)result[i]=f(result[i]-mean);
    return result;
  }

  encode(previous,current,width=160) {
    const now=this.pool(current,width),old=this.pool(previous,width);
    let motion;
    if(this.v.roles.includes('mixed')) {
      const before=this.gray(previous,width),after=this.gray(current,width);
      const [x0,y0,x1,y1]=this.v.crop,w=x1-x0,h=y1-y0,gw=this.v.width,gh=this.v.height;
      const delta=Float32Array.from(after,(value,i)=>f(Math.abs(Math.round(value*255)-Math.round(before[i]*255))/255));
      const horizontal=new Float32Array(gw*h);
      for(let gx=0;gx<gw;gx++)for(let y=0;y<h;y++) {
        let max=0;for(let x=0;x<w;x++)max=Math.max(max,f(delta[y*w+x]*this.fx[gx][x]));
        horizontal[gx*h+y]=max;
      }
      motion=new Float32Array(gw*gh);
      for(let gy=0;gy<gh;gy++)for(let gx=0;gx<gw;gx++) {
        let max=0;for(let y=0;y<h;y++)max=Math.max(max,f(horizontal[gx*h+y]*this.fy[gy][y]));
        motion[gy*gw+gx]=max;
      }
    }
    const sensory=new Float32Array(this.m.input_indices.length);
    for(let i=0;i<sensory.length;i++) {
      const cell=this.v.grid_y[i]*this.v.width+this.v.grid_x[i];
      const role=this.v.roles[i];
      if(role==='current')sensory[i]=clip(f(now[cell]*this.v.contrast_gain));
      else if(role==='previous')sensory[i]=clip(f(old[cell]*this.v.contrast_gain));
      else if(role==='change')sensory[i]=clip(f(f(now[cell]-old[cell])*this.v.motion_gain));
      else if(role==='mixed')sensory[i]=clip(f(f(now[cell]*this.v.contrast_gain)+f(motion[cell]*this.v.motion_gain)));
      else throw Error('Unknown visual channel');
    }
    return sensory;
  }

  forward(sensory) {
    this.state.fill(0); // The verified policy resets activity each decision.
    for(let pass=0;pass<this.m.internal_steps;pass++) {
      this.incoming.fill(0);
      for(let e=0;e<this.weights.length;e++) {
        const receiver=this.post[e];
        this.incoming[receiver]=f(this.incoming[receiver]+f(this.state[this.pre[e]]*this.weights[e]));
      }
      // Held sensory drive enters only the annotated visual input slots.
      const drive=new Float32Array(this.state.length);
      for(let i=0;i<sensory.length;i++)drive[this.m.input_indices[i]]+=sensory[i];
      for(let n=0;n<this.state.length;n++) {
        const total=f(f(f(f(this.m.leak*this.state[n])+this.incoming[n])+drive[n])+this.bias[n]);
        this.next[n]=f(Math.tanh(total));
      }
      [this.state,this.next]=[this.next,this.state];
    }
    const logits=new Float32Array(3);
    for(let action=0;action<3;action++) {
      let value=0;
      for(let i=0;i<this.m.output_indices.length;i++)value=f(value+f(this.state[this.m.output_indices[i]]*this.m.action_weight[action][i]));
      logits[action]=f(value+this.m.action_bias[action]);
    }
    const max=Math.max(...logits),probabilities=Array.from(logits,x=>Math.exp(x-max));
    const sum=probabilities.reduce((a,b)=>a+b,0);
    return {logits:Array.from(logits),probabilities:probabilities.map(x=>x/sum)};
  }

  step(previous,current,width) {
    return this.forward(this.encode(previous,current,width));
  }
}

// Reproducible browser sampling; a different RNG from Torch, same distribution.
export function seededRandom(seed) {
  let state=seed>>>0 || 1;
  return ()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
}
export function sampleAction(probabilities,random) {
  let value=random();
  for(let a=0;a<probabilities.length-1;a++){value-=probabilities[a];if(value<0)return a;}
  return probabilities.length-1;
}
