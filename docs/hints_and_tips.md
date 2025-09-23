# Hints and Tips

Already comfortable with the basics? The ideas below help you squeeze more control out of Prompt Math expressions.

## Blend Strategically

- **Normalise weights mentally.** Expressions are evaluated exactly as written, so `[[ [cat] + [dog] ]]` is equivalent to a 50/50 blend. Scale terms explicitly when you want bias (`[[ 0.8 * [cat] + 0.2 * [dog] ]]`).
- **Subtract with care.** Negative weights are powerful but can amplify noise. When subtracting `[trees]` from a forest scene, clamp the result with `NormalizationOperations` downstream if artifacts appear.

## Pair Schedules for Narrative Control

- **Crossfades:** Use complementary `fade_in` and `fade_out` schedules on two tokens to create cinematic transitions.
- **Pulses:** Register a custom pulse schedule to create rhythmic emphasis (great for music-video style prompts). Combine it with multiplication to modulate only a subset of the expression.
- **Envelope Stacking:** Multiple schedules on the same token multiply together. Chain a `fade_in` with a short emphasis burst to get a rapid highlight at a precise moment.

## Manage Token Libraries

- Store embeddings in a dedicated `Python` node or load them from disk so they are easy to reuse across flows.
- Cache heavy embeddings with ComfyUI's save/load nodes to avoid recomputing encodes.
- Keep a neutral or blank embedding (`pad`) on hand so the evaluator never has to fall back to zeros unintentionally.

## Inspect Schedule Output

- The `schedule_payload` output is JSON-friendly. Plug it into a `Python` node and print the contents while iterating through the diffusion timeline to understand how weights evolve.
- When debugging custom schedules, feed the payload into `ScheduleEvaluator` manually to confirm values at specific timesteps.

## Compose with Other Nodes

- Combine TensorMath with control nets or LoRA loaders: generate conditioning with TensorMath, then mix it with LoRA modifiers using downstream math nodes.
- Use TensorMath in latent image-to-image chains to keep prompt emphasis aligned with image strength schedules.

## Keep the Front-End in Sync

- Any time you add or modify schedules in Python, regenerate the config and clear browser caches. Inconsistent metadata is the most common cause of UI glitches.
- If you ship TensorMath as part of a larger custom node pack, ensure the `web/` folder is copied verbatim so the editor assets remain available.

Experiment, iterate, and document the expressions that work best for you—sharing those recipes back with the community helps everyone build stronger prompt libraries.

