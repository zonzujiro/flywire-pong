/* ALE and neural inference stay off the page's UI thread. */
importScripts('./vendor/ale.js');
let module,ale,policy,model,random,previous,current,seed=20261002;
let done=false,outcome=null,action=null,probabilities=[0,0,0];
let pointsFor=0,pointsAgainst=0,decisions=0,wins=0,losses=0;
const ACTIONS=[3,0,4]; // ALE RIGHT / NOOP / LEFT -> right paddle UP / STAY / DOWN.
const NAMES=['UP','NOOP','DOWN'];
let sample,makeRandom,Policy,lastDecision,diagnosticPolicy;

function reset(){
  seed++;
  if(ale)ale.delete();
  ale=new module.ALEInterface();
  ale.setInt('random_seed',seed);
  ale.setInt('frame_skip',model.environment.frameskip);
  ale.setInt('max_num_frames_per_episode',108000);
  ale.setFloat('repeat_action_probability',model.environment.repeat_action_probability);
  ale.setBool('color_averaging',false);
  ale.loadROM('/roms/pong.bin');
  ale.resetGame();
  ale.act(1); // Initial FIRE, exactly as in the Python Pong wrapper.
  if(ale.getScreenWidth()!==160||ale.getScreenHeight()!==210)throw Error('Unexpected Pong frame size');
  previous=current=new Uint8Array(ale.getScreenRGB());
  random=makeRandom(seed);
  pointsFor=pointsAgainst=decisions=0;done=false;outcome=null;action=null;probabilities=[0,0,0];
  lastDecision=null;
}

function advance(steps){
  if(!Number.isInteger(steps)||steps<1||steps>8)throw Error('Invalid step count');
  for(let i=0;i<steps&&!done;i++){
    const result=policy.step(previous,current,160);
    probabilities=result.probabilities;
    action=sample(probabilities,random);
    lastDecision={previous,current,seed,decision:decisions+1,action:NAMES[action],
      probabilities:probabilities.slice(),points_for:pointsFor,points_against:pointsAgainst};
    const reward=ale.act(ACTIONS[action]);
    previous=current;current=new Uint8Array(ale.getScreenRGB());
    pointsFor+=Math.max(reward,0);pointsAgainst+=Math.max(-reward,0);decisions++;
    done=ale.gameOver()||ale.gameTruncated();
    if(done){
      outcome=ale.gameTruncated()?'truncated':pointsFor>pointsAgainst?'win':'loss';
      wins+=Number(outcome==='win');losses+=Number(outcome==='loss');
    }
  }
}

self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      const net=await import('./network.mjs');
      sample=net.sampleAction;makeRandom=net.seededRandom;Policy=net.FlyPolicy;
      const response=await fetch('./model.json');if(!response.ok)throw Error('Could not load saved network');
      model=await response.json();policy=new net.FlyPolicy(model);
      module=await createALEModule({locateFile:file=>new URL('./vendor/'+file,self.location.href).href,
                                   print:()=>{},printErr:message=>console.warn(message)});
      reset();
    }else if(data.type==='inspect'){
      if(!lastDecision)throw Error('Play a decision before inspecting');
      diagnosticPolicy??=new Policy(model);
      const sensory=diagnosticPolicy.encode(lastDecision.previous,lastDecision.current,160);
      const result=diagnosticPolicy.forward(sensory,true);
      // Separate policy instance and copied frames: no ALE/RNG/action changes.
      const previousFrame=lastDecision.previous.slice(),inputFrame=lastDecision.current.slice();
      self.postMessage({id:data.id,inspection:{...result,seed:lastDecision.seed,decision:lastDecision.decision,
        action:lastDecision.action,chosen_probabilities:lastDecision.probabilities,
        points_for:lastDecision.points_for,points_against:lastDecision.points_against,
        previous_frame:previousFrame.buffer,input_frame:inputFrame.buffer,
        checkpoint_sha256:model.checkpoint_sha256}},[previousFrame.buffer,inputFrame.buffer]);
      return;
    }else if(data.type==='step')advance(data.steps);
    else if(data.type==='reset')reset();
    else throw Error('Unknown game command');
    const frame=current.slice();
    self.postMessage({id:data.id,state:{frame:frame.buffer,seed,points_for:pointsFor,points_against:pointsAgainst,
      decisions,done,outcome,action:action===null?null:NAMES[action],probabilities,wins,losses,
      neurons:model.neuron_count,edges:model.edge_count,checkpoint_sha256:model.checkpoint_sha256}},[frame.buffer]);
  }catch(error){self.postMessage({id:data.id,error:error.message||String(error)});}
};
