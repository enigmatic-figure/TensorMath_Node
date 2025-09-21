# Example: Simple Fade

This example demonstrates how to create a smooth transition between two concepts. We will fade from a "sunny day" to a "starry night."

## Prompt

```
(sunny day:ease-out:1:0), (starry night:ease-in:0:1)
```

## Breakdown

This prompt consists of two parts, separated by a comma:

1.  `(sunny day:ease-out:1:0)`
    *   `sunny day`: The first concept.
    *   `ease-out`: The schedule type. This means the influence of "sunny day" will start strong and then fade out.
    *   `1`: The starting weight. "sunny day" has full influence at the beginning of the generation.
    *   `0`: The ending weight. "sunny day" has no influence at the end of the generation.

2.  `(starry night:ease-in:0:1)`
    *   `starry night`: The second concept.
    *   `ease-in`: The schedule type. This means the influence of "starry night" will start weak and then build up.
    *   `0`: The starting weight. "starry night" has no influence at the beginning.
    *   `1`: The ending weight. "starry night" has full influence at the end.

## Expected Result

The generated image should be a blend of a sunny day and a starry night. The overall structure of the image will likely be determined in the early steps (when "sunny day" is dominant), while the details and lighting will be increasingly influenced by "starry night" as the generation progresses. This can create beautiful, surreal landscapes with the colors of a sunset and the stars of a night sky.
