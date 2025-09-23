# TensorMath Node for ComfyUI

TensorMath brings programmable tensor arithmetic and sophisticated attention scheduling to ComfyUI. It implements the Prompt Memetic Engineering (PME) workflow: prompts are treated as composable vectors that can be blended, attenuated, and animated over the diffusion timeline with precise math.

Whether you want to fade stylistic cues in and out, average multiple token embeddings, or build fully scripted prompt behaviours, TensorMath provides the parser, evaluator, schedulers, and UI assets needed to do it in a repeatable way.

## Why Prompt Memetic Engineering?

Traditional prompt engineering is a manual craft. PME reframes it as a controllable system:

- **Explicit semantics:** Tokens become addressable vectors that can be added, subtracted, or weighted just like tensors.
- **Temporal control:** Scheduling envelopes let you evolve concepts across sampling steps for cinematic reveals or subtle emphasis changes.
- **Reusability:** The same expressions can be reused, versioned, and shared just like code, enabling collaborative prompt design.

Read the full vision in [docs/papers/prompt_memetic_engineering.md](docs/papers/prompt_memetic_engineering.md).

## Feature Highlights

- **Prompt Math DSL** with bracketed expressions (`[[ ... ]]`) supporting addition, subtraction, multiplication, constants, and per-token schedules.
- **Attention schedulers** including linear, smooth, ease-in/out curves plus custom factories that map tokens to time-weighted envelopes.
- **Frontend integration** that ships a ready-made node editor widget and JSON configuration for ComfyUI web builds.
- **Torch shim** (`torch.py`) that falls back to a NumPy-backed API so automated tests can run even without PyTorch installed.
- **Extensible architecture** organised as a proper Python package (`tensor_math`) for easy maintenance and reuse in other nodes.

## Repository Layout

```
__init__.py                  # ComfyUI entry point exposing nodes + web assets
web/                         # Browser assets (JS/CSS) and generated config
 tensor_math/                # Python package for parser, evaluator, scheduler, nodes
 tests/                      # Unit tests exercising the math and scheduling stack
 docs/                       # Guides, examples, coursework, and whitepaper
 torch.py                    # Optional shim for environments without real PyTorch
```

## Requirements

- ComfyUI (latest main branch recommended)
- Python 3.12 (ships with the official ComfyUI Windows release; other platforms should match the version ComfyUI uses)
- PyTorch 2.x (recommended for inference; the torch shim is available for experimentation)
- NumPy 1.26+ (required when relying on the bundled torch shim)
- `pytest` (installed via `requirements-dev.txt`) for running the automated test suite

## Installation (ComfyUI Custom Node)

1. **Locate your ComfyUI install.** On Windows the default path is usually `ComfyUI\` next to `run_nvidia_gpu.bat`.
2. **Clone or copy TensorMath.** Place the repository inside `ComfyUI/custom_nodes/` (for example `ComfyUI/custom_nodes/TensorMath`). The folder should now contain:
   - `__init__.py`
   - `tensor_math/`
   - `web/`
3. **Install Python dependencies.** Activate the ComfyUI Python environment and run:
   ```bash
   pip install -r requirements.txt    # pulls in NumPy for the shim
   pip install torch                  # optional but recommended for production
   pip install -r requirements-dev.txt  # installs pytest for the test suite
   ```
4. **Restart ComfyUI.** The new nodes appear under `conditioning/tensor math`.

### Updating an Existing Install

1. Pull or copy the latest files into the same `custom_nodes/TensorMath` directory.
2. Regenerate the web config after adding or removing schedules:
   ```bash
   py -3.12 -c "import json, pathlib; from tensor_math.prompt_math.evaluator import build_frontend_config; pathlib.Path('web/prompt_math_config.json').write_text(json.dumps(build_frontend_config(), indent=2) + '\n')"
   ```
3. Refresh the ComfyUI browser tab (or clear the browser cache) so the editor JS loads the updated configuration.

## Quickstart Workflow

1. **Add a `Prompt Math - Evaluate` node** from `conditioning/tensor math`.
2. **Provide token vectors.** Use a `Python` node or your preferred embedding source to feed a dictionary such as:
   ```python
   {
       "cat": clip_text_encode("cat"),
       "dog": clip_text_encode("dog"),
       "pad": clip_text_encode("")
   }
   ```
   Any tensors or callables returning tensors are accepted; see [Input Expectations](#prompt-math---evaluate) below.
3. **Set the expression** to a Prompt Math formula, for example:
   ```
   [[ [cat] * 0.5 + [dog] * 0.5 @ fade_in(0.2, 0.8) ]]
   ```
4. **Wire the outputs.** Connect `tensor` to the downstream conditioning input and expose `schedule_payload` to scripts or custom attention nodes that understand the metadata.

## Node Reference

### Prompt Math - Evaluate

- **Category:** `conditioning/tensor math`
- **Inputs:**
  - `expression` (`STRING`, multiline): Prompt Math DSL expression.
  - `token_vectors` (`DICT`): Mapping of token names to tensors, callables returning tensors, or dictionaries containing `tensor` / `value` / `embedding` keys.
  - `encoder` (`STRING`, optional): Name forwarded to the token lookup (defaults to `clip_l`).
  - `pad_token` (`STRING`, optional): Key used if the evaluator needs a padding vector; defaults to generating a zero tensor matching the first embedding.
- **Outputs:**
  - `tensor` (`TENSOR`): Result of evaluating the Prompt Math expression.
  - `schedule_payload` (`DICT`): Metadata bundle with `encoder` and a `schedules` list. Each entry contains token name, start/end, curve type, indices, and any attached metadata.

### Prompt Math - Frontend Config

- **Category:** `conditioning/tensor math`
- **Outputs:**
  - `config` (`DICT`): The same structure as `build_frontend_config()`—schedule metadata, sampler discovery results, and template snippets for UI integrations.

Use this node if you need to expose the schedule catalogue to other custom widgets or sync front-end controls with back-end registrations.

## Prompt Math DSL

Expressions always live inside double brackets: `[[ ... ]]`. Within that scope you can:

- Reference embeddings as `[token_name]`.
- Combine tensors with `+`, `-`, `*`. Arithmetic follows left-to-right evaluation with explicit brackets for grouping.
- Apply scalar weights by multiplying with floats (e.g. `[style] * 0.6`).
- Chain schedules with the `@` suffix: `[token] @ fade_in(0.1, 0.8, "ease_in_out")`.

### Supported Schedules

Built-in schedule factories include:

| Name      | Description                              | Default parameters |
|-----------|------------------------------------------|--------------------|
| `fade_in` | Ramp weight from 0 → 1 between start/end | `start=0.0`, `end=1.0`, optional `curve`
| `fade_out`| Ramp weight from 1 → 0                   | `start=0.0`, `end=1.0`, optional `curve`

Additional helpers (e.g. emphasis, bell curves, pulses) can be registered via `ScheduleFactory`; see [Extending Schedules](#extending-schedules).

### Schedule Curves

Curves correspond to members of `CurveType`: `linear`, `smooth`, `ease_in`, `ease_out`, `ease_in_out`, `constant`. When omitted the factory defaults to `linear`.

### Expression Tips

- Mix numeric literals: `[[ 0.7 * [photo] + 0.3 * [painting] ]]`.
- Nest expressions for clarity: `[[ ([day] + [night]) * 0.5 ]]`.
- Provide fallback tensors (`pad_token`) for rare misspelled tokens to avoid zero-length outputs.

## Extending Schedules

Register custom schedules by subclassing `ScheduleFactory` or by injecting builders at runtime:

```python
from tensor_math.prompt_math.evaluator import ScheduleFactory
from tensor_math.prompt_math.scheduling import create_fade_in_schedule

factory = ScheduleFactory()

def emphasis(token_expr, token_indices, call):
    return create_fade_in_schedule(token_expr, token_indices, start_time=call.args[0], end_time=call.args[1])

factory.register("emphasis", emphasis, metadata={
    "label": "Emphasis",
    "direction": "increase",
    "defaults": {"start": 0.0, "end": 0.5},
    "parameters": [
        {"name": "start", "type": "float", "required": True},
        {"name": "end", "type": "float", "required": True},
    ],
})
```

- Registered schedules automatically appear in `build_frontend_config()`.
- After adding a new factory call, regenerate `web/prompt_math_config.json` using the command from the [Updating](#updating-an-existing-install) section.

## Frontend Assets

The `web/` directory contains:

- `prompt_math/prompt_math_editor.js` and `.css`: assets for an optional client-side editor.
- `prompt_math_config.json`: generated registry consumed by the editor.

When bundling ComfyUI, copy the `web/` folder alongside the node. If you serve ComfyUI through a reverse proxy, ensure the JSON file is accessible relative to the script URL (the loader falls back to `web/prompt_math_config.json`).

## Testing & Development

- **Run tests:** `py -3.12 -m pytest`
- **Torch shim:** If PyTorch is unavailable the bundled `torch.py` provides a minimal NumPy-backed API so the suite still runs. Set `PROMPT_MATH_FORCE_STUB=1` to always use the shim.
- **Repo conventions:**
  - Package code lives under `tensor_math/`
  - Web assets under `web/`
  - Docs in `docs/`

### Working Without PyTorch

The shim supports the tensor operations used by the tests but is not meant for production generation. Install real PyTorch for accurate inference inside ComfyUI.

## Preparing a Registry Submission

ComfyUI's official registry expects a few files at the repository root:

1. **`info.json`** - update the placeholder values with the final author name, repository URL, and version before submitting.
2. **`requirements.txt` / `requirements-dev.txt`** - keep runtime dependencies (NumPy or others) and optional development tools listed so ComfyUI can install them automatically.
3. **Documentation** - the README and docs/ folder should stay in sync with the published features. Include screenshots or sample workflows if desired.
4. **Tags and description** - make sure the values in `info.json` match what you want to appear in the registry listing.

After everything is up to date, follow the instructions in the [ComfyUI Registry](https://github.com/comfyanonymous/ComfyUI/wiki/Custom-Node-Registry) to open a submission PR.
## Troubleshooting

- **Node missing from ComfyUI:** Verify the folder name inside `custom_nodes` matches the repository (`TensorMath`) and that `__init__.py` is in the root.
- **`token_vectors` errors:** Ensure the upstream node outputs a Python `dict` where each value is a tensor or resolves to one. Strings or plain numbers must be converted first.
- **Schedule not appearing in UI:** Regenerate `web/prompt_math_config.json` and clear the browser cache so the front-end picks up the change.
- **`ModuleNotFoundError: tensor_math`:** Restart ComfyUI after installing, or check that the `tensor_math` directory is deployed alongside `__init__.py`.
- **PyTorch CUDA warnings:** They originate from your ComfyUI environment; TensorMath is agnostic to CUDA. Update your PyTorch install if needed.

## Contributing

Pull requests are welcome! Please:

1. Run `py -3.12 -m pytest` (install `pytest` first if necessary).
2. Add or update documentation for new schedules, nodes, or config options.
3. Keep public interfaces backwards compatible where possible.

For major changes, open an issue describing the proposed behaviour so we can align on design.

## License & Credits

TensorMath Node is released under the MIT License (see `LICENSE`). It builds upon the open ComfyUI ecosystem and the wider research community exploring prompt programming. Contributions or scheduling presets referenced from third parties should include attribution in commit messages or documentation.






