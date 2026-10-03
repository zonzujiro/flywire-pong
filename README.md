# FlyWire Pong

[Watch the network play](https://zonzujiro.github.io/flywire-pong/)

The trained FlyWire network controls the **right paddle** against Atari's built-in computer player. The game emulator and neural calculations run locally in your browser. GitHub Pages only serves static files; no Python server, remote inference service, or account is required.

## Controls

- Play / Pause, or Space.
- New game, or R. Abandoned games are not counted as wins or losses.
- Speed: half speed through 4x. Speed changes playback, not the policy or emulator frame skip.
- Completed games advance automatically. Hidden tabs pause.

The score above the arena counts points in the current game. The side panel counts completed games in the current page session. Reloading clears these counters. Action bars are probabilities; actions are sampled, rather than always taking the largest bar.

## Network inspector

Click **Inspect network** to pause and inspect the last decision. Step through its 12 internal updates, click/search individual neurons, inspect source annotations and directed connections, compare fresh or pre-PPO weights with trained weights, and explore local influence on each action score. **Next game decision** advances one neural command; **Play** resumes normal playback. You can download the complete trace with FlyWire IDs and both input images. See [the inspector guide and scientific limits](docs/inspector.md).

The inspector uses the exact images and frozen checkpoint from the recorded choice. Repeated inspection calls leave the emulator and action random generator untouched. Its miniature input images precede the chosen action; the main game board shows the subsequent frame. Graph layout is schematic, and influence is a local sensitivity calculation of our artificial model.

## What the network is

- 1,230 computational states correspond to selected FlyWire FAFB783 neurons; their original neuron IDs remain strings in `model.json`.
- 8,515 directed edge rows carry the exported trained connection weights. All recorded rows are retained, including multiple rows for the same neuron pair. No dense replacement of the biological graph is introduced.
- 355 selected visual inputs receive our encoder's brightness and temporal information over 114 mapped columns on an 11x11 grid.
- Twelve synchronous recurrent updates per decision, held sensory input, and activity reset between decisions match the selected Python policy.
- Two descending readouts, DNp15 and DNp01, feed the exported small action decoder. The network provides every paddle action; there is no ball-following fallback.
- The selected training lineage used supervised initialization followed by PPO. This website performs inference only.

**Known from FlyWire:** the selected neurons and directed connectivity. **Biologically motivated approximation:** inferred placement of selected visual inputs using connectivity and column annotations. **Engineering choices:** image encoding, neuron dynamics, update count, training, and the paddle action decoder. This is an artificial network using biological topology, not a simulation of the fly's complete biological brain.

## Results and scope

The selected checkpoint won **43 of 60** completed clean-Pong games in Python (23/30 and 20/30). A later Python check won 20/30. The original receipt is retained in `verification/python-checkpoint.json`. These results do not establish a causal advantage of fly topology and are **not a measured browser win rate**.

Browser inference matches 30 recorded Python frame pairs, including uncertain action choices. Maximum absolute action-probability difference is 0.000007871. See `verification/port.json` and the executable parity test. The browser uses ALE 0.12.0 instead of native ALE 0.12.1, a different random-number generator for action sampling, and direct ALE seeds rather than Gymnasium's seed conversion. Exact trajectories are consequently not expected to match Python evaluations.

## Run locally

Any static web server works. For example, from this folder:

```sh
python -m http.server 8876
```

Open http://localhost:8876/. Python here only serves files; it does not run the neural network. Opening `index.html` directly as a `file:` URL does not provide the worker/module environment this site needs.

## Verify

Node.js 22 or newer:

```sh
npm ci
npm test
```

The included compressed fixtures bind to the exact SHA256 of `model.json` and compare the encoder, every recurrent neuron state, logits, and action probabilities against Python calculations. No emulator installation is needed for this test.

For the browser smoke test, first start the static server. By default it uses installed Microsoft Edge:

```sh
npm run test:browser
```

Alternatively, `npx playwright install chromium` and set `BROWSER_CHANNEL=chromium`. Set `SITE_URL` to test another deployment. This test checks emulator loading, neural decisions, real point scoring, pause, reset, speed, and mobile overflow; it writes screenshots under ignored `artifacts/`. It is not a win-rate benchmark.

## Files

| File | Purpose |
| --- | --- |
| `index.html`, `app.mjs` | Responsive interface and playback controls |
| `game-worker.js` | Runs the emulator and network away from the UI thread |
| `network.mjs` | Sparse recurrent inference and retina encoder |
| `model.json` | Immutable trained model export and neuron IDs |
| `inspector.mjs`, `inspection.mjs`, `inspector-data.json` | Decision inspection, sensitivity analysis, biological metadata and pinned weight baselines |
| `tests/` | Python parity fixtures and browser checks |
| `verification/` | Checkpoint provenance and measured port errors |
| `vendor/` | Unmodified official ALE WASM package and corresponding source |

The model was exported from the research project with `scripts/export_browser_policy.py`, which verifies the selected checkpoint hash before exporting effective weights, neuron biases, encoder arrays, and the action decoder. This standalone repository contains the deployable inference artifact, not the original dataset download or training project.

## Deployment

GitHub Pages serves the `master` branch at `/`. All runtime URLs are relative so workers and assets work under `/flywire-pong/`. `.nojekyll` requests static serving. `node_modules`, screenshots, raw connectome downloads, and Python caches are excluded from version control.

## Sources and license

- [FlyWire Codex](https://codex.flywire.ai/).
- [Official ALE WebAssembly documentation](https://ale.farama.org/wasm/).
- [ALE source, v0.12.0](https://github.com/Farama-Foundation/Arcade-Learning-Environment/tree/v0.12.0).

The browser application code is provided under GPL-2.0; see `LICENSE`. Third-party emulator files retain their upstream license and bundled data. See `vendor/NOTICE.md` for provenance and the included corresponding-source archive. This is an independent experiment, not an official FlyWire or Atari product.
