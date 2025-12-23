# Instructor Guide: Neural Networks - Concepts and Evolution

**Module Time:** 120 minutes

---

**What This Module is About:**
This module serves as the bridge between classical Machine Learning and Deep Learning. It demystifies the "Black Box" of neural networks by breaking them down into their mathematical components and placing them in historical context.

**Core Objectives:**
- Trace the historical evolution of DL to understand *why* it is dominant now.
- Deconstruct a neural network into inputs, weights, biases, and activation functions.
- Connect abstract architecture to concrete applications in Vision and NLP.

**Must-Hit Topics:**
- The difference between Biological and Artificial Neurons (don't overdo the biology).
- The definition of **Non-linearity** (Activation Functions).
- The 3 ingredients of the DL boom: Data, Compute, Algorithms.
- The distinction between Input, Hidden, and Output layers.

---

## 1. From Perceptrons to SOTA
**(Time: ~25 minutes)**

**Purpose:** To prevent students from viewing AI as magic by showing the incremental scientific progress and the reliance on hardware/data.

**Talking Points:**
-   **The XOR Problem:** Use this to explain why early AI failed. A single straight line cannot separate an XOR pattern. This justifies the need for "Hidden Layers."
-   **The "Winter":** Explain that AI research lost funding because it over-promised and under-delivered. Cautionary tale for today.
-   **Why Now?** Ask the class: "The math existed in the 80s. Why didn't we have ChatGPT in 1990?" (Answer: No internet for data, no GPUs for compute).

**Teaching Tips:**
-   Use the **Mermaid Timeline** in the slides to visualize the gaps in time.
-   Don't memorize dates; focus on the *shifts* in technology.

---

## 2. What is a Neural Network?
**(Time: ~30 minutes)**

**Purpose:** To establish the mental model of a neural network as a series of matrix multiplications passed through non-linear functions.

**Talking Points:**
-   **The Anatomy:** Draw a neuron on the whiteboard. Label inputs as "ingredients" and weights as "the recipe quantities."
-   **The Bias:** Explain bias as the "threshold." Even if inputs are 0, do we still want the neuron to fire?
-   **Activation Functions:** This is the most critical concept. Explain `ReLU` (Rectified Linear Unit) as essentially "If negative, return 0; if positive, return x." It's simple but powerful.

**Teaching Tips:**
-   **Analogy:** Compare a neural net to a pricing model for a house.
    -   *Inputs:* Sqft, Bedrooms.
    -   *Weights:* Value per sqft.
    -   *Bias:* Base price of the land.
-   *Check for Understanding:* Ask "What happens if we remove all activation functions from a 100-layer network?" (Answer: It collapses into a single linear regression).

---

## 3. Deep Learning Applications
**(Time: ~20 minutes)**

**Purpose:** To help students identify *when* to use Deep Learning versus classical ML.

**Talking Points:**
-   **Structured vs. Unstructured Data:** Classical ML (Random Forest, XGBoost) often wins on spreadsheets. DL wins on Images, Text, Audio.
-   **Feature Extraction:**
    -   *Old Way:* Humans manually wrote code to detect "corners" or "edges" in images.
    -   *DL Way:* The network *learns* to detect corners and edges in the hidden layers.

**Teaching Tips:**
-   Show examples of failures. Show an image where a slight rotation confuses a standard algorithm, but a CNN handles it easily.

---

## 4. Lab: "From Linear to Neural" - a simple feedforward neural network using PyTorch.
**(Time: ~35 minutes)**

**Purpose:** Replace logistic regression with a simple feedforward neural network using PyTorch.

Refer: [Lab Instructor Guide](https://git.generalassemb.ly/applied-ai-and-deep-learning-in-action/Lab-2_1-From-Linear-to-Neural/blob/main/instructor-guide.md)

---

## Wrap-Up Reflection
**(Time: ~10 minutes)**

**Key Takeaways:**
-   Deep Learning is efficient largely due to GPUs and Big Data, not just "better math."
-   We are currently in the "Transformer Era" (founded in 2017), moving toward Generative AI.