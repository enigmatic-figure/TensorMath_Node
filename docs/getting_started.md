# Getting Started with TensorMath

Welcome! This guide walks you from first install to a working Prompt Math workflow inside ComfyUI. If you only need a high-level overview, start with the repository [README](../README.md). When you are ready to build your first schedule-driven prompt, follow the steps below.

## 1. Prerequisites

- ComfyUI cloned or downloaded from the official repository
- Python 3.12 (Bundled with the Windows release. On Linux/macOS use the same Python version ComfyUI uses.)
- Optional: PyTorch 2.x to run inference with GPU acceleration

If you cannot install PyTorch immediately, the included `torch.py` shim lets you run tests and simple experiments using NumPy under the hood.

## 2. Installation

1. **Download the node.** Clone the repository or unzip the release archive.
2. **Copy it into ComfyUI.** Place the folder inside `ComfyUI/custom_nodes/` (for example `ComfyUI/custom_nodes/TensorMath`). The folder should contain `__init__.py`, the `tensor_math/` package, and the `web/` directory.
3. **Install dependencies.** Activate the Python environment ComfyUI ships with and run:
   ```bash
   pip install -r requirements.txt
   pip install torch                 # optional but recommended for production
   pip install -r requirements-dev.txt  # installs pytest for the smoke tests
   ```
4. **Restart ComfyUI.** Launch your normal ComfyUI entry point (`run_nvidia_gpu.bat`, `python main.py`, etc.). Open the UI in a browser and confirm the nodes appear under `conditioning/tensor math`.

## 3. Verify the Installation

Add a new node by right-clicking the canvas and searching for `Prompt Math`. You should see two nodes:

- `Prompt Math - Evaluate`
- `Prompt Math - Frontend Config`

If the nodes are missing, review the troubleshooting section in the README (common fixes include making sure the directory name is correct and restarting ComfyUI).

## 4. Build Your First Workflow

Follow this minimal example to produce a blended embedding and a registered schedule:

1. **Insert a `Prompt Math - Evaluate` node.**
2. **Supply token vectors.** The node expects a Python dictionary where each key is a token string and each value resolves to a tensor. Quick ways to provide one:
   - Use the built-in `Python` node with code similar to:
     ```python
     {
         "cat": clip_text_encode("cat"),
         "dog": clip_text_encode("dog"),
         "pad": clip_text_encode("")
     }
     ```
   - Connect the output of a custom embedding loader if you already index tokens elsewhere.
3. **Configure the expression.** Set the `expression` input to:
   ```
   [[ [cat] * 0.5 + [dog] * 0.5 @ fade_in(0.2, 0.8, "smooth") ]]
   ```
   This starts with an equal blend of the two tokens, then increases the "dog" component using the `fade_in` schedule between 20% and 80% of the sampling timeline with a smooth easing curve.
4. **Route the outputs.**
   - Connect `tensor` to the conditioning input of your sampler (e.g. to a `KSampler` node).
   - Optionally inspect `schedule_payload` by plugging it into a `Preview` or custom Python node to validate the metadata.
5. **Run the flow.** Generate an image. You should see the prompt transition manifest as the diffusion progresses.

## 5. Explore the Prompt Math DSL

- Wrap expressions in double brackets: `[[ ... ]]`.
- Reference tokens using square brackets: `[token_name]`.
- Combine tokens with `+`, `-`, `*`, and scalar literals.
- Add schedules with the `@` operator: `[token] @ fade_in(0.1, 0.9, "ease_in_out")`.
- Nest expressions when combining multiple operations: `[[ ([cat] - [dog]) * 0.6 ]]`.

Refer back to the README for a full list of built-in schedules and schedule curves.

## 6. Regenerate the Front-End Configuration (Optional)

When you register additional schedule builders via `ScheduleFactory`, rebuild the JavaScript configuration so the browser UI stays in sync:

```bash
py -3.12 -c "import json, pathlib; from tensor_math.prompt_math.evaluator import build_frontend_config; pathlib.Path('web/prompt_math_config.json').write_text(json.dumps(build_frontend_config(), indent=2) + '\n')"
```

Reload the ComfyUI browser tab to pick up the new metadata.

## 7. Next Steps

- Continue with [Coursework 1: Introduction to Scheduling](coursework/introduction_to_scheduling.md) for guided practice.
- Browse the [examples](examples/simple_fade.md) for ready-made expressions.
- Read the [FAQ](faq.md) for troubleshooting and design guidance.
- Dive into [hints and tips](hints_and_tips.md) to learn advanced blending strategies.

Happy prompting!
