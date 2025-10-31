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

## 1. Introduction to Python for AI (15 min)

Python is the **lingua franca of artificial intelligence and deep learning** due to its simplicity, readability, and powerful ecosystem of libraries.

| Concept | Description | Example |
|----------|--------------|----------|
| High-level language | Easy-to-read syntax, close to English | `print("Hello AI!")` |
| Interpreted | No compilation needed | Run directly in terminal or Jupyter |
| Versatile | Used in AI, web dev, automation | TensorFlow, Flask, Pandas |

**Example: Basic Python Syntax**

```python
# Variables and types
name = "AI Learner"
hours = 32
print(f"Welcome {name}, to {hours} hours of AI training!")
```
---

## 3. Setting Up the Development Environment (15 min)

Ensure all required tools are installed and verified.

✅ Required Tools

✅Python 3.9+

✅Jupyter Notebooks or VS Code

✅Git and GitHub account

✅Key Libraries: NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch

✅Zoom, Slack, and Chrome (for course delivery)

 **Sanity Check**

Run these commands to confirm setup:
```python --version
jupyter --version
git --version
pip list
```
Activity:
Learners open Jupyter, create a new notebook titled "Module1_Setup.ipynb", and run:
```
import numpy, pandas, sklearn
print("✅ Environment setup complete!")
```
---

## 4. Python Essentials for Machine Learning (20 min)
**Key Topics**
1. Variables, data types, lists, dictionaries, and loops
2. Writing and calling functions
3. Handling exceptions and debugging

**Example:**
```
def normalize(data):
    """Return normalized values between 0 and 1."""
    return (data - data.min()) / (data.max() - data.min())

import pandas as pd
scores = pd.Series([10, 15, 20, 25])
print(normalize(scores))
```

**Mini Exercise:**
Write a function that squares each number in a list and returns the new list.
```
numbers = [2, 4, 6]
# Output: [4, 16, 36]
```
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
