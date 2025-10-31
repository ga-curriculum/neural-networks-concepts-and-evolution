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
## 5. Data Handling and Libraries (20 min)

**Focus:** Core libraries for AI and ML workflows.

| Library | Purpose | Example |
|----------|----------|----------|
| **NumPy** | Numerical computation | `np.mean(arr)` |
| **Pandas** | Data manipulation | `df.describe()` |
| **Matplotlib** | Visualization | `plt.plot()` |

**Follow-Along Code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.randn(100)
df = pd.DataFrame({'values': data})
plt.hist(df['values'], bins=20)
plt.title("Random Data Distribution")
plt.show()
Copy code
```
---
## 6. Introduction to Git and Version Control (15 min)

Version control helps track changes, collaborate, and revert errors efficiently.

**Key Commands**
```bash
git init
git add .
git commit -m "Initial commit"
```
**Concepts:**

1.Repository: Project folder with version history

2.Commit: Snapshot of code changes

3.Branch: Separate workspace for experiments

---
## 7. Working with GitHub (10 min)

**Connecting to GitHub**
```git remote add origin https://github.com/username/repo.git
git push -u origin main
```

**Branching Workflow Example**
```git checkout -b feature-branch
# Work on code
git commit -am "Added data visualization section"
git checkout main
git merge feature-branch
```

**In-Class Collaboration Activity:**

Students pair up and fork a practice repository to make and merge changes.

---

## 8. Lab: Baseline Builder – Logistic Regression (Setup) (10 min)

**Objective:** Implement your first ML model using Scikit-learn.

```from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Baseline Accuracy:", accuracy_score(y_test, pred))
```


**Discussion:**

1. What does the model learn?
2. How can we improve this baseline in later modules (hint: neural networks)?

---
   
## 9. Wrap-Up and Reflection (5 min)

**Group Discussion:**
1. What challenges did you face while setting up your environment?
2. Which Python concept felt most familiar or new to you?
3. How does Git enhance collaboration in AI projects?

**Instructor Notes:**
1. Reiterate learning outcomes achieved.
2. Preview next module: Deep Learning with PyTorch.

---

## Resources 
- [Python Official Docs](https://docs.python.org/3/)  
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)  
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)  
- [Git Documentation](https://git-scm.com/doc)  
- [GitHub Guides](https://guides.github.com/)  
- [General Assembly Student Setup Guide](https://generalassemb.ly)
