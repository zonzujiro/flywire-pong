"""Generate Python reference states and action sensitivities for the inspector."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import torch


def export(root, website):
    sys.path.insert(0, str(root))
    from model.factory import make_policy
    from vision.encoder import RetinaEncoder
    receipt = json.loads((root / "docs/pong-winning-checkpoint.json").read_text())
    saved = torch.load(root / receipt["selected_checkpoint"], map_location="cpu", weights_only=True)
    config = saved["config"]
    encoder = RetinaEncoder(root / config["mapping_path"], root / config["vision_config"])
    policy = make_policy(config, encoder)
    policy.load_state_dict(saved["model"])
    torch.set_num_threads(1)
    original = json.loads(gzip.decompress((website / "tests/parity.json.gz").read_bytes()))
    cases = []
    for index in (2, 15, 28):
        sensory = torch.tensor(original["cases"][index]["sensory"]).unsqueeze(0)
        states = [policy.connectome.initial_state(1).requires_grad_(True)]
        for _ in range(policy.internal_steps):
            states.append(policy.connectome.step(states[-1], sensory))
        logits = policy.action_head(policy.connectome.readout(states[-1]))[0]
        gradients = {}
        for action in (0, 2):
            score = logits[action] - sum(logits[a]/2 for a in range(3) if a != action)
            gradients[str(action)] = [g.detach()[0].tolist() for g in torch.autograd.grad(score, states, retain_graph=True)]
        cases.append(dict(parity_case=index, sensory=sensory[0].tolist(), states=[s.detach()[0].tolist() for s in states], gradients=gradients))
    output = dict(model_sha256=original["model_sha256"], checkpoint_sha256=receipt["sha256"], cases=cases,
                  definition="d(target logit - mean(other logits))/d(activity at each update)")
    (website / "tests/inspection.json.gz").write_bytes(gzip.compress(json.dumps(output, separators=(",", ":")).encode(), mtime=0))
    print(json.dumps(dict(cases=len(cases), updates=policy.internal_steps, references="PyTorch autograd")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--website", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    export(args.research_root, args.website)
