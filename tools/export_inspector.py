"""Export biological metadata and pinned weight baselines from the research repo."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def export(research_root, website):
    model = json.loads((website / "model.json").read_text())
    root = research_root
    receipt = json.loads((root / "docs/pong-winning-checkpoint.json").read_text())
    selected = root / receipt["selected_checkpoint"]
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == model["checkpoint_sha256"]
    trained = torch.load(selected, map_location="cpu", weights_only=True)["model"]
    nodes = pq.read_table(root / receipt["graph_dir"] / "nodes.parquet").to_pylist()
    edges = pq.read_table(root / receipt["graph_dir"] / "edges_by_neuropil.parquet").to_pylist()
    assert [str(n["root_id"]) for n in nodes] == model["root_ids"]
    assert [e["local_pre_index"] for e in edges] == model["pre"]
    assert [e["local_post_index"] for e in edges] == model["post"]
    baselines = []
    for label, relative in (
        ("Fresh synapse-count initialization", "data/processed/training/pong_winning_grid11_v2/initial.pt"),
        ("Before final PPO (already taught)", "data/processed/training/pong_winning_grid11_corrections/ppo/checkpoint_initial.pt"),
    ):
        path = root / relative
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = saved["model"]
        for key in ("root_ids", "pre_index", "post_index", "synapse_count", "input_index", "output_index"):
            assert torch.equal(state[f"connectome.{key}"], trained[f"connectome.{key}"]), key
        assert saved["config"].get("incoming_gain_bound") is None
        if relative.endswith("/initial.pt"):
            assert saved["config"]["fresh_biological_initialization"]
            syn = state["connectome.synapse_count"]
            raw = torch.log1p(syn)
            incoming = torch.zeros(model["neuron_count"]).index_add_(0, state["connectome.post_index"], raw)
            expected = .8 * raw / incoming[state["connectome.post_index"]].clamp_min(1e-12)
            assert torch.equal(expected, state["connectome.edge_weight"])
        baselines.append(dict(label=label, source=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              weights=state["connectome.edge_weight"].tolist()))
    fields = ("cell_type", "cell_class", "super_class", "side", "known_nt", "top_nt", "top_nt_conf")
    mapping = json.loads((root / receipt["graph_dir"] / "retinotopy.json").read_text())
    result = dict(format_version=1, checkpoint_sha256=model["checkpoint_sha256"],
                  model_sha256=hashlib.sha256((website / "model.json").read_bytes()).hexdigest(),
                  nodes=[{**{key: n[key] for key in fields}, "root_id": str(n["root_id"]),
                          "selection_group": n["selected_stage"]} for n in nodes],
                  edge_synapses=[int(e["syn_count"]) for e in edges],
                  edge_regions=[e["neuropil"] for e in edges], baselines=baselines,
                  visual_inputs=[dict(neuron_index=e["local_index"], root_id=str(e["root_id"]),
                                      column_id=e["column_id"], p=e["p"], q=e["q"],
                                      lamina_target_ids=[str(r) for r in e["lamina_target_root_ids"]])
                                 for e in mapping["entries"]], visual_mapping_method=mapping["biological_status"])
    assert np.array_equal(np.asarray(model["weights"], dtype=np.float32), trained["connectome.edge_weight"].numpy())
    (website / "inspector-data.json").write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(dict(neurons=len(nodes), edges=len(edges), baselines=len(baselines), verified=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--website", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    export(args.research_root, args.website)
