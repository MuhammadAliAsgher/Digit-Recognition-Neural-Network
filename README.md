# 🧮 Handwritten Digit Recognition with a Two-Layer Neural Network

## 📘 Overview

This project implements a two-layer neural network from scratch to classify handwritten digits from the **MNIST dataset**, as part of the **Kaggle Digit Recognizer** competition.

The neural network is built using **only NumPy**, without high-level frameworks like TensorFlow or PyTorch, to demonstrate a deep understanding of neural network fundamentals.

The model architecture:
- **Input layer**: 784 pixels (28x28 images)
- **Hidden layer**: 10 units with ReLU activation
- **Output layer**: 10 units with softmax activation

It was trained using gradient descent, achieving:
- 🏋️‍♂️ **Training accuracy**: 84.53%
- 🧪 **Dev set accuracy**: 83.40%

Training stopped at 490 iterations due to Kaggle resource limits. The project includes data preprocessing, model training, evaluation, visualizations, and leaderboard submission.

🔗 **Links**:
- 📘 Kaggle Notebook: [Digit Recognition Neural Network](https://www.kaggle.com/code/muhammadaliasghar01/digit-recognition-neural-network)
- 💻 GitHub Repository: [Digit-Recognition-Neural-Network](https://github.com/MuhammadAliAsgher/Digit-Recognition-Neural-Network)

---

## 🔑 Key Features

### 🧠 Neural Network Architecture
- **Input layer**: 784 units (28x28 pixel images)
- **Hidden layer**: 10 units with ReLU activation
- **Output layer**: 10 units with softmax activation (for digits 0–9)

### ⚙️ Implementation Details
- Built **from scratch using NumPy**
- Includes:
  - Forward and backward propagation
  - Gradient descent optimization
  - Numerical stability for softmax

### 📊 Training and Evaluation
- Trained on **41,000 samples**
- Evaluated on **1,000-sample dev set**
- Accuracy:
  - 🏋️ Training: 84.53%
  - 🧪 Dev: 83.40%

### 📈 Visualizations
- Learning curve (accuracy vs. iterations)
- Sample digit predictions from training set
- Confusion matrix and misclassified examples

### 🏆 Kaggle Submission
- Generated predictions for **28,000 test images**
- Created `submission.csv`
- **Expected test accuracy**: ~83%

---

## 📋 Results

| Dataset  | Accuracy |
|----------|----------|
| Training | 84.53%   |
| Dev      | 83.40%   |

---

## 🔍 Observations

- Model **generalizes well** with only slight overfitting.
- **4 and 9** are often misclassified due to similar shapes.
- Despite early stopping at 490 iterations (due to Kaggle limits), the model performed **strongly**.

---

## 📁 Repository Contents

- `digit-recognition-neural-network.ipynb`: Jupyter Notebook with full implementation
- 📎 **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/muhammadaliasghar01/digit-recognition-neural-network)

---

## 🚀 How to Run

### ✅ Prerequisites

- Python **3.11+**
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🧾 Steps

### 1. Clone the Repository

```bash
git clone https://github.com/MuhammadAliAsgher/Digit-Recognition-Neural-Network.git
cd Digit-Recognition-Neural-Network
```
