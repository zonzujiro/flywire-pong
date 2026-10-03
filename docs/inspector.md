# Inspect an actual Pong decision

Click **Inspect network** below the game. This pauses play and retrieves the last neural decision. The small images show the exact previous/current RGB frames used by the encoder. The main Pong board shows the frame after that action, two emulator frames later. The inspector identifies the seed, decision number, sampled command, and score before the action.

## Walk through the calculation

Use **Previous update**, **Next update**, or the slider. Update 0 is the zero starting state. Updates 1 through 12 recompute every selected neuron using the recorded incoming edges and the held sensory input. The final action is sampled only after update 12. Moving the slider does not advance Pong.

**Next game decision** advances the emulator by one real neural command, then records the new calculation. **Play** closes the inspector and resumes normal playback. Inspection and diagnostic requests never sample an action, call ALE, change the game's random generator, or change its policy weights.

## Neurons and connectivity

The graph shows all 1,230 selected neurons and 8,515 directed connection rows. Colors show signed state values between −1 and +1. Layout groups are the circuit extraction's engineering stages, not anatomical coordinates. **Show → Visual groups** isolates the selected light-input, lamina, medulla, motion, and visual-projection populations. Zoom enlarges the scrollable graph.

Click a dot, choose from the keyboard-accessible neuron selector, or search by cell type or full FlyWire ID. Selecting a neuron highlights its recorded incoming/outgoing connections, shows its source labels and incident synapse regions, and plots its activity across all updates. Region labels belong to synapses, not an invented single region for a neuron. Missing labels remain unknown. The external lookup uses the [official Codex search query format](https://blog.flywire.ai/2025/10/29/codex-query-master-workshop/); current Codex may have changed individual IDs since the pinned FAFB783 dataset.

For visual inputs, details separate the inferred biological column, anatomical eye-grid coordinate, engineered screen-grid coordinate, and engineered current/previous/change channel. Column placement was inferred using measured links to source-annotated lamina cells and cross-checked against all relevant targets in the full graph.

## Output contributions and influence

The readout table shows the numerical contribution of each of the two descending states to each action logit, plus the learned action bias. It is evaluated at the shown update; only update 12 is used for play. Positive activity or a large contribution alone does not guarantee that action is selected: logits become probabilities, then an action is sampled.

**Influence on action score** computes local derivatives through the twelve synchronous updates. For selected action `a`, the inspected score is:

```text
score = logit[a] − mean(logits of the other two actions)
influence[update, neuron] = derivative of score with respect to that activity
```

Positive sensitivity means a small increase in this state favors that action relative to its alternatives. Negative sensitivity disfavors it. Sensitivity colors are scaled to the maximum absolute derivative at the displayed update. This is a local differential analysis of this artificial model, treating Float32 rounding as identity as ordinary autodiff does. It is not a causal ablation result, a prediction about large perturbations, or a measured function of a biological neuron.

## Connection changes

Two baselines are exported from saved checkpoints, verifying identical IDs, connectivity, synapse counts and input/output indexing:

1. **Fresh synapse-count initialization:** the existing untrained state from the winning graph's lineage. Each initial weight equals `0.8 * log1p(synapse count) / sum_incoming(log1p(synapse count))`, checked against the stored tensor. This is an engineering initialization, not a measured biological strength.
2. **Before final PPO (already taught):** the preserved initialization of the final PPO stage. It already contains prior learning; changes relative to it isolate that last stage's weight differences.

The connection table retains separate neuron-pair/region rows and lists original synapse counts, region, baseline weight, trained weight and signed difference. For large neighborhoods it lists the 100 largest absolute changes and explicitly reports the limit. The graph keeps all connections. A second table shows the 20 largest changes across the entire graph. Learned weight signs do not establish biological excitation/inhibition.

## Export and verify

**Download this decision's activity** saves a JSON record containing all 13 state vectors, original neuron IDs as strings, sensory input, logits/probabilities, sampled action, checkpoint hash, seed/decision number and both exact RGB frames encoded in base64.

`npm test` checks the original 30 Python inference cases and three additional PyTorch traces. The latter check every state at every update and autograd sensitivities for UP and DOWN. An independent finite-difference test covers recurrence, a negative edge and feedback.

`npm run test:browser` tests controls and inspection on desktop/mobile, including the downloaded trace. A separate comparison runs two identical browser games for 320 decisions, with forty inspection calls in only one game, and requires identical RGB frames, native scores, action probabilities, sampled commands and game counters. Reset must discard the old inspection.

To regenerate inspection metadata or Python reference fixtures using the existing research environment, run the appropriate script in `tools/` with `--research-root`. Fixture generation expects the research repository to be the working directory because its model factory reads the original relative configuration paths. Regeneration verifies the selected model hash and biological indexing; it does not retrain the model.

## Scientific boundaries

**Known from FlyWire:** IDs, source cell annotations, directed connections, synapse counts and synapse-region labels. Transmitter predictions are labeled predictions; source annotations are labeled separately.

**Biologically motivated approximation:** photoreceptor placement inferred from connections to annotated visual columns.

**Engineering choices:** graph selection/grouping/layout, image projection and channel assignments, neuron dynamics, action decoder, initial/trainable weights, and gradient-based influence analysis. None is presented as a discovered fly behavior.
