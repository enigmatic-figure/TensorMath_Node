# Getting Started with TensorMath_Node

Welcome to TensorMath_Node! This guide will help you install the tool and take your first steps into the world of "Prompt Memetic Engineering."

## 1. What is TensorMath_Node?

In simple terms, TensorMath_Node is a powerful calculator for your prompts in ComfyUI. It lets you control the concepts in your prompts with mathematical precision. Instead of just writing "a cat and a dog," you can say "50% cat and 50% dog," or even make the "cat" part of the prompt fade in over time.

This approach, which we call "Prompt Memetic Engineering," allows for a whole new level of creativity and control. If you're interested in the science behind it, you can read our white paper: [An Introduction to Prompt Memetic Engineering](papers/prompt_memetic_engineering.md).

## 2. Installation

Getting TensorMath_Node running in ComfyUI is easy. Just follow these steps:

1.  **Navigate to your ComfyUI installation.**
2.  Find the `custom_nodes` directory. It's usually located at `ComfyUI/custom_nodes/`.
3.  **Download the TensorMath_Node files.** You can do this by cloning the repository or downloading the ZIP file.
4.  **Place the files in the right location.**
    *   Copy the `src/prompt_math` folder into your `custom_nodes` directory.
    *   Copy the Python files (`prompt_math_eval_extended.py`, `prompt_math_extended_functions.py`, etc.) from the root of the project into the `custom_nodes` directory alongside the `prompt_math` folder.
5.  **Restart ComfyUI.**

That's it! The TensorMath nodes should now be available in your ComfyUI node menu.

## 3. Your First Prompt

Let's try a simple example.

1.  **Add a TensorMath_Node** to your workflow. You can find it by right-clicking, selecting "Add Node," and looking for "TensorMath."
2.  **Connect it to your model's prompt input.**
3.  **In the TensorMath_Node's text box, type the following:**
    ```
    (cat:0.5), (dog:0.5)
    ```
4.  **Generate an image.**

You should see an image that is a blend of a cat and a dog. You've just created your first engineered prompt! You controlled the "strength" of the "cat" and "dog" concepts in your prompt.

## 4. What's Next?

You've only scratched the surface of what's possible. To continue your journey, we recommend checking out the following resources:

*   **[Coursework](coursework/introduction_to_scheduling.md):** Our guided lessons will walk you through the core features of TensorMath_Node, one concept at a time.
*   **[Examples](examples/simple_fade.md):** Explore our collection of practical examples to see how you can use TensorMath_Node in your own workflows.
*   **[FAQ](faq.md):** Find answers to common questions.
*   **[Hints and Tips](hints_and_tips.md):** Learn some advanced tricks and techniques.

Happy engineering!
