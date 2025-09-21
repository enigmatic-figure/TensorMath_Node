# Frequently Asked Questions (FAQ)

## Q: What is ComfyUI?

ComfyUI is a powerful and modular node-based user interface for Stable Diffusion. TensorMath_Node is a "custom node" designed to run within the ComfyUI environment. You will need to have ComfyUI installed to use this tool.

## Q: Can I use this with other tools besides ComfyUI?

Currently, TensorMath_Node is designed specifically for ComfyUI. The core Python scripts for parsing and scheduling are independent, so they could theoretically be adapted for other Stable Diffusion user interfaces, but this would require custom integration work.

## Q: Where can I find a list of all the available schedules?

The `prompt_math_eval_extended.py` file contains a call to `build_frontend_config()`, which generates a `prompt_math_config.json` file. This file contains a list of all registered schedules. We plan to add a more user-friendly reference page for this in the future.

## Q: What happens if my weights don't add up to 1?

You don't need to make your weights add up to 1! The system will automatically normalize the weights for you at each step of the generation. For example, if you have `(cat:1), (dog:1)`, it will be treated as `(cat:0.5), (dog:0.5)`. This allows you to think in terms of ratios rather than exact percentages.

## Q: Can I nest expressions?

Yes! The parser supports nested expressions. For example, you can do `((cat:0.5), (dog:0.5)):0.8), (bird:0.2)` to create a blend of a cat and a dog, and then blend that resulting concept with a bird. This allows for extremely complex and fine-grained control over your prompts.
