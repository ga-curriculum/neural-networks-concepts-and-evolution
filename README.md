# Neural Networks: Concepts and Evolution

**Session Time:** 120 minutes

---
## Table of Contents 

1. Introduction to Neural Networks  
2. The Evolution of Neural Networks  
3. Anatomy of a Neural Network  
4. Activation Functions  
5. Forward and Backpropagation  
6. Loss Functions and Optimization  
7. Neural Networks in Action (PyTorch Implementation)  
8. Lab: From Linear to Neural — Building a Simple Feedforward Network  
9. Wrap-Up and Reflection  
10. Resources   

---

## Learning Objectives

By the end of this module, you'll be able to:
1. Explain what a neural network is and how it models human brain-inspired computation.
2. Trace the historical evolution of neural networks from perceptrons to modern deep learning architectures.
3. Understand the structure and functioning of layers, neurons, and activation functions.
4. Describe how forward propagation, loss functions, and backpropagation work together to train a model.
5. Build, train, and evaluate a simple feedforward neural network using **PyTorch**.
6. Compare the performance of a neural network to a traditional machine learning model (e.g., logistic regression).
---

## Session Breakdown

| Segment | Topic | Duration |
|----------|--------|----------|
| 1 | Introduction to Neural Networks | 10 min |
| 2 | The Evolution of Neural Networks | 10 min |
| 3 | Anatomy of a Neural Network | 15 min |
| 4 | Activation Functions | 10 min |
| 5 | Forward and Backpropagation | 20 min |
| 6 | Loss Functions and Optimization | 10 min |
| 7 | Neural Networks in Action (PyTorch Implementation) | 25 min |
| 8 | Lab: From Linear to Neural | 15 min |
| 9 | Wrap-Up and Reflection | 5 min |

---

### 1. Introduction to Neural Networks (10 min)

Neural Networks (NNs) are computational models inspired by the human brain. They are designed to recognize complex patterns through interconnected nodes (neurons) that process input data and make predictions.

**Simple Analogy:**  
Just as our brains learn by strengthening certain neural connections, artificial neural networks adjust their internal weights based on errors during training.

**Diagram:**  
![Simple Neural Network](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/800px-Artificial_neural_network.svg.png)

---

### 2. The Evolution of Neural Networks

| Era | Key Concept | Innovation | Example |
|------|--------------|-------------|----------|
| **1950s–60s** | Perceptron | Single-layer binary classifier | Rosenblatt’s Perceptron |
| **1980s** | Backpropagation | Multi-layer learning | Feedforward Networks |
| **2000s** | Deep Learning | Multi-layer networks + GPUs | CNNs, RNNs |
| **2010s–2020s** | Modern AI | Transformers, LLMs | GPT, BERT |

**Discussion Prompt:**  
Ask learners: “Why did neural networks decline in the 1970s and return to prominence decades later?”

(Hint: computation power, data availability, and better algorithms.)

---

### 3. Anatomy of a Neural Network

A **neural network** consists of:
- **Input Layer:** Receives data (e.g., image pixels, text embeddings).  
- **Hidden Layers:** Extracts patterns through weighted connections.  
- **Output Layer:** Produces final predictions.

**Mathematical Representation:**
\[
y = f(Wx + b)
\]
Where:
- \( W \) = weights  
- \( b \) = bias  
- \( f \) = activation function  

**Visual Representation:**
![Neural Network Layers](https://miro.medium.com/v2/resize:fit:1200/1*oY5D6oajm2M7YgB1h3bN5A.png)

---

### 4. Activation Functions

Activation functions introduce non-linearity, allowing networks to learn complex relationships.

| Function | Formula | Use Case | Visualization |
|-----------|----------|----------|----------------|
| **Sigmoid** | \( f(x) = \frac{1}{1 + e^{-x}} \) | Binary classification | S-shaped curve |
| **ReLU** | \( f(x) = \max(0, x) \) | Deep networks | Fast and efficient |
| **Tanh** | \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | Normalized output | Zero-centered |
| **Softmax** | \( f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} \) | Multi-class classification | Probabilities sum to 1 |

**Code Example:**

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-1.0, 0.0, 1.0])
print("Sigmoid:", torch.sigmoid(x))
print("ReLU:", F.relu(x))
print("Tanh:", torch.tanh(x))
```

---

### 5. Forward and Backpropagation

**Forward Propagation:**  
The process of passing input data through the network to produce an output.

**Backpropagation:**  
The process of updating weights by calculating the gradient of the loss function.

**Formula for Weight Update:**
\[
w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}
\]
Where:
- \( \eta \) = learning rate  
- \( L \) = loss function  

**Visualization:**
![Backpropagation Process](https://upload.wikimedia.org/wikipedia/commons/8/8c/Backpropagation_of_error.svg)

**Interactive Concept:**  
Ask learners: “What happens if the learning rate is too high or too low?”

---

### 6. Loss Functions and Optimization

**Loss Function:** Measures how far predictions are from actual values.  
**Optimizer:** Algorithm that adjusts weights to minimize loss.

| Task | Common Loss | Optimizer |
|------|--------------|------------|
| Binary Classification | Binary Cross-Entropy | SGD, Adam |
| Multi-Class Classification | Categorical Cross-Entropy | Adam |
| Regression | Mean Squared Error | RMSProp, Adam |

**Code Example:**

```python
import torch
import torch.nn as nn

loss_fn = nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
```

---
   
### 7. Neural Networks in Action (PyTorch Implementation)

**Build a simple feedforward network using PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).view(-1, 1)

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print("Final Loss:", loss.item())
```
---
### 8. Lab: From Linear to Neural — Building a Simple Feedforward Network

**Objective:**  
Transform a logistic regression model into a neural network and compare performance.

**Activities:**
1. Revisit the logistic regression baseline from Module 2.  
2. Build a one-hidden-layer neural network using PyTorch.  
3. Visualize training loss over iterations.  
4. Evaluate model accuracy and discuss results.  

**Extension Idea:**  
Add one more hidden layer and observe how the model performance changes.

---

## Wrap-Up and Reflection

**Reflection Prompts:**
1. How do neural networks differ fundamentally from traditional ML models?  
2. What role do activation functions and backpropagation play in learning?  
3. How did adding hidden layers affect performance?  
4. How might this knowledge prepare you for CNNs and more advanced architectures?

**Instructor Tip:**  
Encourage learners to summarize today’s model in their GitHub notebooks and experiment with additional activation functions at home.

---

## Resources 
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
- [Neural Networks and Deep Learning (Michael Nielsen Book)](http://neuralnetworksanddeeplearning.com/)  
- [3Blue1Brown — Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)  
- [CS231n: Neural Networks Notes (Stanford)](http://cs231n.github.io/neural-networks-1/)  
- [Activation Functions Explained (Towards Data Science)](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)  
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/)  
- [Visualizing Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)
