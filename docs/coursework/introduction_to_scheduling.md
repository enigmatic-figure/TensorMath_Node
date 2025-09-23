# Coursework 1: Introduction to Scheduling

This lesson expands the quickstart by focusing on one core capability of Prompt Math—**scheduling**. You will learn what schedules are, why they matter, and how to author your own time-aware expressions.

## 1. Understanding Scheduling

Diffusion models generate images over multiple steps. Scheduling lets you change a token's weight as those steps progress. Instead of a static blend, you can:

- Fade a concept in or out.
- Pulse attention for emphasis.
- Chain envelopes so different concepts dominate at different phases.

Mathematically, a schedule is an easing curve sampled across the `[0, 1]` diffusion timeline. The evaluator converts the curve into per-token weights and stores them in the `schedule_payload` output.

## 2. Anatomy of a Schedule Expression

A scheduled token looks like this:

```
[[ [token_name] @ fade_in(0.2, 0.8, "smooth") ]]
```

Component summary:

- `[token_name]` — the embedding you want to manipulate.
- `@` — attaches schedule metadata to the preceding token.
- `fade_in` — the schedule factory to use (`fade_out` is also built in).
- `(0.2, 0.8, "smooth")` — arguments passed to the factory. In this case the token ramps from weight 0 at 20% of the timeline to weight 1 at 80% using a smooth Hermite curve.

## 3. Guided Exercise

Follow the steps below in ComfyUI.

1. **Set up the node.** Add `Prompt Math - Evaluate` and feed it a token library containing `"robot"` and `"portrait"` embeddings.
2. **Create a fade-in.** Use this expression:
   ```
   [[ [robot] @ fade_in(0.1, 0.7) ]]
   ```
   Observe how the subject acquires robotic features as sampling proceeds.
3. **Add a counterweight.** Update the expression to:
   ```
   [[ [portrait] * 0.7 + [robot] * 0.3 @ fade_in(0.1, 0.7, "ease_in_out") ]]
   ```
   Now the portrait stays dominant while the robotic style gently emerges.
4. **Experiment with timing.** Move the start/end values closer together to create a rapid transition (e.g. `fade_in(0.45, 0.55)`).

## 4. Inspecting the Result

Connect `schedule_payload` to a `Python` node and print the contents:

```python
for schedule in schedule_payload["schedules"]:
    print(schedule["token"], schedule["start"], schedule["end"], schedule["curve"])
```

You should see the timing metadata align with your expression. Use the values to coordinate other nodes—noise schedules, LoRA strengths, or custom attention controllers—so everything follows the same tempo.

## 5. Practice Prompts

Try authoring expressions for the scenarios below:

1. **Day-to-night landscape.** Crossfade `[daytime_forest]` and `[night_sky]` using `fade_out` and `fade_in` respectively.
2. **Highlight details mid-sampling.** Keep `[architecture]` at full strength but spike `[neon_lights]` between 60% and 75% of the diffusion timeline.
3. **Multi-stage style transfer.** Start with `[sketch]`, transition to `[oil_painting]`, then finish with `[photograph]` using three consecutive schedules.

Document the expressions that work well; they make great building blocks for future workflows.

## 6. Next Lesson Preview

In the next coursework module we will explore composing multiple schedules across complex expressions, sharing schedule factories, and exposing sliders in the ComfyUI front end for real-time adjustments.

