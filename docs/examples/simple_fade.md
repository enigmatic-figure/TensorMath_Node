# Example: Simple Fade

This walkthrough shows how to fade one concept into another using the built-in schedules.

## Scenario

Blend a `sunny day` scene into a `starry night` as sampling progresses.

## Expression

```
[[ [sunny_day] @ fade_out(0.0, 0.7, "ease_out") + [starry_night] @ fade_in(0.3, 1.0, "ease_in") ]]
```

## How It Works

1. **`[sunny_day] @ fade_out(0.0, 0.7, "ease_out")`**
   - Starts at full strength.
   - Uses an ease-out curve so the energy drops gently.
   - Reaches zero by 70% of the diffusion timeline.
2. **`[starry_night] @ fade_in(0.3, 1.0, "ease_in")`**
   - Begins contributing around 30% into sampling.
   - Accelerates towards the end, producing a dramatic reveal of the night sky.
3. **Addition** blends both tensors, handing off control as the schedules cross.

## Steps in ComfyUI

1. Provide token vectors for `sunny_day`, `starry_night`, and a neutral `pad` token.
2. Set the expression on the `Prompt Math - Evaluate` node to the snippet above.
3. Connect the `tensor` output to your sampler and generate an image.
4. Optional: inspect `schedule_payload` to confirm the start/end times and curve types match expectations.

The resulting image should shift from a warm daylight palette to a cool nocturnal scene as the diffusion steps unfold, demonstrating how schedules can orchestrate visual storytelling.
