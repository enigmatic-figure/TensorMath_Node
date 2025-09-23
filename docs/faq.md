# Frequently Asked Questions

This FAQ collects common questions from users integrating TensorMath into ComfyUI workflows. If you cannot find the answer you need, open an issue or reach out on the community channels listed in the main README.

## General

**Q: What does TensorMath actually do?**  
It evaluates Prompt Math expressions—small pieces of tensor algebra—against your token embeddings and returns both the blended tensor and the schedules required to reproduce the behaviour during sampling.

**Q: Do I need to understand PyTorch to use it?**  
No. You only need a source of embeddings (Clip text encoder, custom loaders, etc.). The node handles tensor math internally. If you are developing new schedulers or operations, familiarity with PyTorch or NumPy helps.

**Q: Can I use TensorMath outside ComfyUI?**  
Yes, the Python package inside `tensor_math/` has no ComfyUI dependency. You can import it in standalone scripts to parse expressions, evaluate schedules, or build custom tooling. The provided nodes are simply convenience wrappers for ComfyUI.

## Compatibility & Requirements

**Q: Which versions of ComfyUI are supported?**  
The project targets the current main branch. Older releases may work, but new ComfyUI breaking changes are tracked against main.

**Q: Is PyTorch required?**  
The node prefers real PyTorch, but the repository ships a NumPy-based shim (`torch.py`) used by the tests. You can experiment with simple expressions using the shim, yet production workflows should install PyTorch for correct device placement and performance.

**Q: What pip packages should I install?**  
Run `pip install -r requirements.txt` to pull in NumPy (needed when you rely on the bundled torch shim). For development and testing, also run `pip install -r requirements-dev.txt` and install PyTorch if you plan to generate images locally.

**Q: Does it support CUDA and ROCm builds of PyTorch?**  
Yes. TensorMath uses high-level tensor operations that work across CPU, CUDA, and ROCm backends chosen by the underlying PyTorch installation.

## Using the Nodes

**Q: What does the `token_vectors` input expect?**  
A Python dictionary mapping string keys to tensors or callables returning tensors. If your embeddings come from another node that outputs a tensor directly, wrap it in a dictionary before feeding the evaluator.

**Q: What happens when a token is missing?**  
If the lookup returns `None`, the evaluator uses the `pad_token` (when provided) or falls back to a zero tensor shaped like the first embedding. A missing pad causes an error so the issue is explicit.

**Q: How do I reuse the schedule payload?**  
Connect the `schedule_payload` output to your own Python or JSON node. Each entry contains token metadata, timings, curve information, and original indices. Custom attention pipelines can iterate through the list and apply weights accordingly.

## Prompt Math Expressions

**Q: Is whitespace important?**  
Whitespace is ignored except within quoted strings. Feel free to format complex expressions across multiple lines for readability inside the multiline text box.

**Q: Are there built-in functions beyond `fade_in` and `fade_out`?**  
The default factory exposes those two. Additional helpers (emphasis, bells, pulses, etc.) are easy to register via `ScheduleFactory.register`. Update the front-end config after registration so the UI knows about them.

**Q: Can I schedule the result of a whole expression instead of a single token?**  
Yes. Wrap the sub-expression in brackets, give it a name via `ScheduleFactory`, or introduce a helper node that registers the schedule against synthetic tokens. The evaluator collects schedules wherever the AST contains a `@` suffix.

## Troubleshooting

**Q: The node does not show up after copying the folder.**  
Double-check the install path: `ComfyUI/custom_nodes/TensorMath`. Restart both the backend and browser. If you renamed the folder, update the internal import paths accordingly.

**Q: The editor still lists old schedules after I changed the Python side.**  
Regenerate `web/prompt_math_config.json` with `build_frontend_config()` and hard-refresh your browser to clear cached assets.

**Q: PyTorch complains about device mismatches.**  
Make sure all incoming embeddings share the same device (CPU/GPU). You can call `.to(device)` before storing them in the token dictionary.

**Q: `ModuleNotFoundError: comfy.samplers` during testing.**  
The tests stub out the Comfy modules automatically. If you run into import errors, ensure you cleaned previous stubs with `clear_comfy_stub()` as shown in `tests/test_prompt_math.py` or set `PYTHONPATH` appropriately.

## Development

**Q: How do I add a new schedule function?**  
Create a builder that returns a `TokenSchedule` and register it via `ScheduleFactory.register(name, builder, metadata=...)`. Add unit tests, update the docs, and regenerate the front-end config.

**Q: Are contributions welcome?**  
Absolutely. See the [contributing section](../README.md#contributing) of the README for guidelines and testing requirements.

Still have questions? Open an issue or submit a discussion thread so we can expand the documentation for everyone.




