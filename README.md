# TensorMath_Node: The Prompt Memetic Engineering Toolkit

**TensorMath_Node** is a collection of scheduling, parsing, and math utilities that bring programmable tensor manipulation to ComfyUI. It elevates prompt engineering to a new level, allowing you to manipulate the concepts in your prompts with mathematical precision.

This new paradigm, which we call "Prompt Memetic Engineering," allows for a whole new level of creativity and control. Instead of just writing "a cat and a dog," you can say "50% cat and 50% dog," or even make the "cat" part of the prompt fade in over time.

## Documentation

We have created a comprehensive set of documentation to help you get started with TensorMath_Node and to explore the concepts behind Prompt Memetic Engineering.

*   **[White Paper: An Introduction to Prompt Memetic Engineering](docs/papers/prompt_memetic_engineering.md)**
    *   Learn about the theory and vision behind the project.
*   **[Getting Started Guide](docs/getting_started.md)**
    *   Your first steps to installing and using TensorMath_Node.
*   **[Coursework](docs/coursework/introduction_to_scheduling.md)**
    *   Guided lessons to help you master the core features.
*   **[Examples](docs/examples/simple_fade.md)**
    *   Practical examples that you can use in your own projects.
*   **[Frequently Asked Questions (FAQ)](docs/faq.md)**
    *   Answers to common questions.
*   **[Hints and Tips](docs/hints_and_tips.md)**
    *   Advanced techniques for power users.

## For Developers

This section contains more technical information for developers who want to contribute to the project or customize it for their own needs.

### Highlights

*   **Scheduling toolkit:** Hermite and easing curves, normalized timestep helpers, and an `AttentionScheduler` for stacking fade-in/fade-out envelopes.
*   **Robust parser & evaluator:** Understands nested prompt-math expressions, optional schedules, and supports blending tokens with registered schedules.
*   **Extended math helpers:** Vector blending, weighted statistics, masking, and normalization routines tuned for attention workflows.
*   **Torch compatibility shim:** Ships with a lightweight `torch` fallback that defers to the real library when installed, keeping tests runnable on minimal environments.

### Getting Started (for Developers)

1.  Ensure `python3.12` is available (ComfyUI already bundles Python on Windows builds).
2.  Install dependencies if you want the real PyTorch backend: `pip install torch`.
3.  Drop the `src/prompt_math` folder into your ComfyUI custom nodes directory and place the Python modules alongside your node implementation.

### Running the Test Suite

```bash
py -3.12 -m pytest
```

### Customisation Tips

*   Use `ScheduleFactory.register("my_curve", builder)` to add bespoke attention envelopes.
*   Combine `MaskingOperations.apply_mask` with `NormalizationOperations.layer_norm` to create context-aware prompt strengths.
*   The parser accepts additive (`+`), subtractive (`-`), and multiplicative (`*`) combinations—perfect for weighting tokens dynamically.
*   Regenerate the front-end schedule palette with `py -3.12 -c "import json, pathlib; from prompt_math_eval_extended import build_frontend_config; pathlib.Path('prompt_math_config.json').write_text(json.dumps(build_frontend_config(), indent=2) + '\n')"` whenever you add new `ScheduleFactory` builders.

### Support and Contributions

Contributions are welcome. Add new scheduling curves, tensor utilities, or ComfyUI node wrappers and extend the tests in `tests/test_prompt_math.py` to keep everything green.
