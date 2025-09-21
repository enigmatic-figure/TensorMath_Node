# Hints and Tips

Ready to take your prompt engineering to the next level? Here are a few hints and tips for getting the most out of TensorMath_Node.

## Tip: Use Negative Weights to Subtract Concepts

You can use negative weights to "subtract" a concept from your prompt. For example, if you're generating a forest scene but want to reduce the number of trees, you could try a prompt like:

`(forest:1), (trees:-0.5)`

This can be a powerful way to fine-tune your images and remove unwanted elements. Be aware that this is an experimental feature, and the results can sometimes be unpredictable.

## Tip: Combine with other Custom Nodes

TensorMath_Node is just one piece of the puzzle. You can combine it with other custom nodes to create incredibly complex and powerful workflows. For example, you could use a "latent noise" node to create a base image, and then use TensorMath_Node to apply a scheduled style to it.

## Tip: The Power of Zero

Don't underestimate the power of a zero weight. You can use a schedule to completely remove a concept at a certain point in the generation. For example, `(raining:1:hermite:1:0)` will create a scene that starts with rain and ends with no rain at all. This can be more effective than trying to add a "not raining" concept with a negative weight.

## Tip: Discovering New Schedules

Want to see how the schedules are defined? Take a look at the `prompt_math_extended_functions.py` file. You'll find the Python functions that define the shape of each schedule. You can even add your own custom schedules by following the examples in that file and registering them with the `ScheduleFactory`.
