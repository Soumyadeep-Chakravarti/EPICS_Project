# Models Explained

This document explains each machine learning algorithm used in our project in simple terms.

## Overview

We use 4 different classification algorithms, plus an ensemble that combines them all:

| Model | Type | Key Idea |
|-------|------|----------|
| Logistic Regression | Linear | Draw a line to separate classes |
| Decision Tree | Tree-based | Ask yes/no questions |
| SVM | Kernel-based | Find the widest gap between classes |
| Random Forest | Ensemble of trees | Many trees vote together |
| Voting Ensemble | Meta-ensemble | All 4 models vote together |

## 1. Logistic Regression

### Simple Explanation

Despite its name, Logistic Regression is used for **classification**, not regression. It draws a mathematical "line" (or hyperplane in higher dimensions) to separate benign from malignant cases.

### How It Works

```
                    Malignant
                        |
    Feature 2           |  x x x
        ^               | x x x x
        |           x x |x x
        |         x x x |
        |       x x x   |
        |     o o o ----|---- Decision Boundary
        |   o o o o     |
        | o o o o       |
        +-----------------> Feature 1
              Benign
```

1. Takes the 30 input features
2. Multiplies each by a learned weight
3. Adds them up and passes through a "sigmoid" function
4. Output is a probability between 0 and 1

### The Math (Simplified)

```
Step 1: z = w1*x1 + w2*x2 + ... + w30*x30 + b

Step 2: probability = 1 / (1 + e^(-z))

Step 3: if probability > 0.5 -> Malignant
        if probability <= 0.5 -> Benign
```

### Why We Use It

- **Fast**: Trains in milliseconds
- **Interpretable**: Weights tell us which features matter
- **Works well**: Achieves 98.25% accuracy on our data!
- **Provides probabilities**: We can see confidence levels

### Configuration in Our Code

```python
LogisticRegression(
    max_iter=5000,    # Maximum training iterations
    C=1.0,            # Regularization strength
    random_state=42   # For reproducibility
)
```

### Our Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.25% |
| Precision | 98.61% |
| Recall | 98.61% |
| ROC-AUC | 0.9954 |

---

## 2. Decision Tree

### Simple Explanation

A Decision Tree makes decisions by asking a series of yes/no questions, like a flowchart. It's the most intuitive model to understand.

### How It Works

```
                    [Is worst_radius > 16.8?]
                           /        \
                         Yes         No
                         /            \
              [Is worst_texture    [Is mean_concavity
                  > 25.7?]            > 0.05?]
                 /     \              /      \
               Yes      No          Yes       No
               /         \          /          \
          Malignant    [...]    Malignant    Benign
```

1. Starts at the root with all data
2. Finds the best feature and threshold to split
3. Recursively splits until reaching a decision
4. Each leaf node gives a prediction

### Why Splits?

The tree chooses splits that best separate the classes using **Gini impurity**:

```
Gini = 1 - (p_benign)^2 - (p_malignant)^2

Perfect split (all one class): Gini = 0
Worst split (50/50): Gini = 0.5
```

### Why We Use It

- **Highly interpretable**: Can visualize the decision process
- **No scaling needed**: Works with raw feature values
- **Handles non-linear relationships**: Can capture complex patterns
- **Feature importance**: Shows which features matter most

### The Problem: Overfitting

Decision Trees can "memorize" training data instead of learning general patterns. This is why our Decision Tree has the lowest accuracy (91.23%).

### Configuration in Our Code

```python
DecisionTreeClassifier(
    max_depth=None,           # No limit on tree depth
    min_samples_split=2,      # Minimum samples to split
    random_state=42
)
```

### Our Results

| Metric | Score |
|--------|-------|
| Accuracy | 91.23% |
| Precision | 95.59% |
| Recall | 90.28% |
| ROC-AUC | 0.9157 |

---

## 3. Support Vector Machine (SVM)

### Simple Explanation

SVM finds the "best" line to separate classes - specifically, the line that has the **maximum margin** (widest gap) between the closest points of each class.

### How It Works

```
                    Malignant
                        
    Feature 2      x    |    x
        ^          x    |  x
        |            x  |x        <- Support Vectors
        |       --------|--------  <- Decision Boundary
        |            o  |o        <- Support Vectors  
        |          o    |  o
        |        o      |    o
        +-----------------> Feature 1
              Benign
              
        |<--- margin --->|
```

The "support vectors" are the points closest to the boundary - they "support" or define the margin.

### The Kernel Trick

What if the classes aren't linearly separable? SVM uses a **kernel** to transform the data into a higher dimension where they become separable.

We use the **RBF (Radial Basis Function)** kernel:

```
Before (not separable):          After kernel (separable):
                                      
    o o x x                          x x
    o o x x           -->        x x     x x
    o o x x                    o o o o o o o o
```

### Why We Use It

- **Excellent accuracy**: Tied for best at 98.25%
- **Works well with high dimensions**: Our 30 features are no problem
- **Robust to overfitting**: The margin maximization helps generalize
- **Handles non-linear data**: Thanks to the kernel trick

### Configuration in Our Code

```python
SVC(
    C=1.0,              # Regularization parameter
    kernel='rbf',       # Radial Basis Function kernel
    probability=True,   # Enable probability estimates
    random_state=42
)
```

### Our Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.25% |
| Precision | 98.61% |
| Recall | 98.61% |
| ROC-AUC | 0.9950 |

---

## 4. Random Forest

### Simple Explanation

Random Forest is a "wisdom of the crowd" approach - it builds **100 different decision trees** and lets them vote on the prediction.

### How It Works

```
Training Data
     |
     v
+----+----+----+----+
|    |    |    |    |
v    v    v    v    v
Tree Tree Tree ... Tree   (100 trees)
 1    2    3        100
 |    |    |         |
 v    v    v         v
 M    B    M   ...   M    (Individual predictions)
 
         |
         v
    Final Vote: Malignant (majority wins)
```

### Two Key Techniques

**1. Bagging (Bootstrap Aggregating)**
- Each tree is trained on a random subset of the data
- Samples are drawn with replacement
- This creates diverse trees

**2. Feature Randomness**
- At each split, only consider a random subset of features
- Prevents all trees from being identical
- Default: sqrt(30) = ~5 features per split

### Why We Use It

- **Reduces overfitting**: Averaging many trees smooths out individual errors
- **Feature importance**: Naturally measures which features matter
- **Robust**: Less sensitive to noisy data
- **No scaling needed**: Tree-based methods don't need normalized data

### Configuration in Our Code

```python
RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=None,       # No limit on tree depth
    random_state=42
)
```

### Our Results

| Metric | Score |
|--------|-------|
| Accuracy | 95.61% |
| Precision | 95.89% |
| Recall | 97.22% |
| ROC-AUC | 0.9939 |

### Feature Importance from Random Forest

Our Random Forest identified these as the most important features:

1. **worst concave points** (0.142)
2. **worst perimeter** (0.131)
3. **worst radius** (0.112)
4. **mean concave points** (0.098)
5. **worst area** (0.091)

---

## 5. Voting Ensemble

### Simple Explanation

The Voting Ensemble combines all 4 models into a single "committee" that makes decisions together.

### How It Works: Soft Voting

We use **soft voting**, which averages the probability predictions:

```
Sample X arrives
       |
       v
+------+------+------+------+
|      |      |      |      |
v      v      v      v      v
LR     DT     SVM    RF
|      |      |      |
v      v      v      v
0.95   0.80   0.92   0.88   <- P(Malignant)

Average = (0.95 + 0.80 + 0.92 + 0.88) / 4 = 0.89

Final: 89% confidence -> Malignant
```

### Why Soft Voting?

**Hard voting** just counts class votes (3 vs 1), but **soft voting** considers confidence:

```
Hard Voting:                    Soft Voting:
LR: Malignant (99%)            LR: 0.99
DT: Benign (51%)               DT: 0.49  
SVM: Malignant (98%)           SVM: 0.98
RF: Malignant (97%)            RF: 0.97
                               ────────────
Result: Malignant (3-1)        Avg: 0.86 -> Malignant

Both give same answer, but soft voting uses more information!
```

### Why We Use It

- **Reduces individual model errors**: One model's mistake can be corrected by others
- **More robust**: Less sensitive to any single model's weaknesses
- **Better calibrated probabilities**: Averaging smooths out extreme predictions
- **Combines diverse approaches**: Linear, tree-based, and kernel methods

### Configuration in Our Code

```python
VotingClassifier(
    estimators=[
        ('Logistic Regression', lr_model),
        ('Decision Tree', dt_model),
        ('SVM', svm_model),
        ('Random Forest', rf_model),
    ],
    voting='soft'  # Use probability averaging
)
```

### Our Results

| Metric | Score |
|--------|-------|
| Accuracy | 97.37% |
| Precision | 97.26% |
| Recall | 98.61% |
| ROC-AUC | 0.9954 |

---

## Model Comparison

### Accuracy Rankings

```
Logistic Regression ████████████████████████████████████████ 98.25%
SVM                 ████████████████████████████████████████ 98.25%
Ensemble            ███████████████████████████████████████  97.37%
Random Forest       ██████████████████████████████████████   95.61%
Decision Tree       █████████████████████████████████████    91.23%
```

### When to Use Each Model

| Model | Best For |
|-------|----------|
| Logistic Regression | When you need speed and interpretability |
| Decision Tree | When you need to explain decisions to non-technical stakeholders |
| SVM | When you have clean data and need high accuracy |
| Random Forest | When you want feature importance and robustness |
| Ensemble | When reliability matters most (medical diagnosis!) |

### Why Didn't Ensemble Beat Individual Models?

Surprisingly, our Ensemble (97.37%) didn't beat Logistic Regression (98.25%). This happens because:

1. **Decision Tree pulls down the average**: At 91.23%, it adds noise
2. **Data is nearly linear**: LR and SVM already perform near-perfectly
3. **Soft voting averages errors**: When 3 models are right but 1 is wrong with high confidence, it affects the result

In more complex problems, ensembles typically outperform individual models.

---

## Summary Table

| Model | Accuracy | Speed | Interpretability | Overfitting Risk |
|-------|----------|-------|------------------|------------------|
| Logistic Regression | 98.25% | Very Fast | High | Low |
| Decision Tree | 91.23% | Fast | Very High | High |
| SVM | 98.25% | Medium | Low | Low |
| Random Forest | 95.61% | Medium | Medium | Low |
| Ensemble | 97.37% | Slow | Medium | Very Low |

---

**Next**: Learn about the [Code Walkthrough](05_CODE_WALKTHROUGH.md) to see how these models are implemented.

*See [Glossary](09_GLOSSARY.md) for any unfamiliar terms.*
