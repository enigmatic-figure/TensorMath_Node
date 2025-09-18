# TensorMath_Node

A collection of scheduling, parsing, and math utilities that bring programmable tensor manipulation to ComfyUI custom nodes.

## Highlights
- **Scheduling toolkit:** Hermite and easing curves, normalized timestep helpers, and an `AttentionScheduler` for stacking fade-in/fade-out envelopes.
- **Robust parser & evaluator:** Understands nested prompt-math expressions, optional schedules, and supports blending tokens with registered schedules.
- **Extended math helpers:** Vector blending, weighted statistics, masking, and normalization routines tuned for attention workflows.
- **Torch compatibility shim:** Ships with a lightweight `torch` fallback that defers to the real library when installed, keeping tests runnable on minimal environments.

## Getting Started
1. Ensure `python3.12` is available (ComfyUI already bundles Python on Windows builds).
2. Install dependencies if you want the real PyTorch backend: `pip install torch`.
3. Drop the `src/prompt_math` folder into your ComfyUI custom nodes directory and place the Python modules alongside your node implementation.

## Running the Test Suite
```bash
py -3.12 -m pytest
```

## Customisation Tips
- Use `ScheduleFactory.register("my_curve", builder)` to add bespoke attention envelopes.
- Combine `MaskingOperations.apply_mask` with `NormalizationOperations.layer_norm` to create context-aware prompt strengths.
- The parser accepts additive (`+`), subtractive (`-`), and multiplicative (`*`) combinations—perfect for weighting tokens dynamically.
- Regenerate the front-end schedule palette with `py -3.12 -c "import json, pathlib; from prompt_math_eval_extended import build_frontend_config; pathlib.Path('prompt_math_config.json').write_text(json.dumps(build_frontend_config(), indent=2) + '\n')"` whenever you add new `ScheduleFactory` builders.

## Support
Contributions are welcome. Add new scheduling curves, tensor utilities, or ComfyUI node wrappers and extend the tests in `tests/test_prompt_math.py` to keep everything green.
