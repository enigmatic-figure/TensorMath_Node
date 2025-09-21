# Coursework 1: Introduction to Scheduling

Welcome to the first lesson in our TensorMath_Node coursework series. In this lesson, you'll learn about one of the most powerful features of the toolkit: **Scheduling**.

## 1. What is Scheduling?

In the "Getting Started" guide, you learned how to give a concept a fixed weight, like `(cat:0.5)`. Scheduling takes this a step further: it allows you to **change the weight of a concept over the course of the image generation process.**

Think of the image generation as happening in a series of steps. With scheduling, you can tell the model to pay more attention to "cat" at the beginning of the process and less attention at the end, or vice-versa.

## 2. Why is Scheduling Useful?

Scheduling unlocks a huge range of creative possibilities:

*   **Smooth Transitions:** You can create smooth blends between different concepts. Imagine an image that starts as a sketch and slowly "paints" itself into a photograph.
*   **Concept Emphasis:** You can emphasize certain details at different stages. For example, you could have the broad strokes of a landscape appear first, with the fine details filling in later.
*   **Evolving Narratives:** You can create images that tell a story. A character's expression could subtly shift from happy to sad, or a serene landscape could gradually transform into a stormy one.

## 3. Your First Scheduled Prompt

Let's try it out. We're going to create a prompt where the concept of a "robot" fades in over the course of the generation.

1.  **Add a TensorMath_Node** to your workflow.
2.  **In the text box, type the following:**
    ```
    (robot:linear:0:1)
    ```
3.  **Generate an image.**

Let's break down what this prompt does:

*   `robot`: This is the concept we're manipulating.
*   `linear`: This is the **schedule type**. `linear` means the weight will change at a constant rate.
*   `0`: This is the **starting weight**. At the beginning of the generation, "robot" will have a weight of 0.
*   `1`: This is the **ending weight**. At the end of the generation, "robot" will have a weight of 1.

So, this prompt tells the model to "fade in" the concept of a robot over the course of the generation. Try experimenting with other concepts and see what you can create!

## 4. Available Schedules

TensorMath_Node comes with a variety of built-in schedules to give you fine-grained control over your prompts. Some of the most common ones include:

*   `linear`: A constant rate of change.
*   `ease-in`: Starts slow and then speeds up.
*   `ease-out`: Starts fast and then slows down.
*   `hermite`: A smooth, S-shaped curve.

You can find a full list of available schedules in the node's documentation (which we'll build out later!).

## 5. Next Steps

Congratulations, you've now learned the basics of scheduling! The best way to learn more is to experiment. Try combining multiple scheduled concepts in a single prompt, or try using different schedule types to see how they affect the output.

In the next lesson, we'll cover how to blend multiple schedules together to create even more complex and interesting effects.
