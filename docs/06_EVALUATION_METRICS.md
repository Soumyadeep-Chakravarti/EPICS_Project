# Evaluation Metrics

This document explains all the metrics we use to measure model performance and why each one matters for medical diagnosis.

## Why Multiple Metrics?

A single number like "accuracy" doesn't tell the whole story. Consider this scenario:

```
Dataset: 100 patients
- 95 benign (healthy)
- 5 malignant (cancer)

A model that ALWAYS predicts "benign" achieves:
Accuracy = 95/100 = 95%  ← Looks great!

But it missed ALL 5 cancer cases!
```

This is why we use multiple metrics, especially for medical applications.

---

## The Confusion Matrix

Everything starts with the **confusion matrix** - a table showing prediction outcomes:

```
                      PREDICTED
                 Benign    Malignant
              ┌─────────┬───────────┐
    Benign    │   TN    │    FP     │
ACTUAL        │  (40)   │   (2)     │
              ├─────────┼───────────┤
    Malignant │   FN    │    TP     │
              │  (1)    │   (71)    │
              └─────────┴───────────┘
```

### The Four Outcomes

| Term | Meaning | Our Result | Good or Bad? |
|------|---------|------------|--------------|
| **TP** (True Positive) | Cancer detected correctly | 71 | Good |
| **TN** (True Negative) | Healthy identified correctly | 40 | Good |
| **FP** (False Positive) | Healthy wrongly flagged as cancer | 2 | Bad (causes anxiety) |
| **FN** (False Negative) | Cancer missed | 1 | Very Bad (missed diagnosis!) |

### Medical Context

In cancer detection:
- **False Negatives (FN) are worse than False Positives (FP)**
- Missing cancer = delayed treatment = potentially fatal
- False alarm = extra tests = stress, but patient is safe

---

## Metric 1: Accuracy

### What It Is

The percentage of all predictions that were correct.

### Formula

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct / Total
```

### Our Example

```
Accuracy = (71 + 40) / (71 + 40 + 2 + 1)
         = 111 / 114
         = 97.37%
```

### When It's Useful

- When classes are roughly balanced
- For a quick overall performance summary

### When It's Misleading

- With imbalanced datasets (like 95% vs 5%)
- When the cost of different errors varies

### Our Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 98.25% |
| SVM | 98.25% |
| Ensemble | 97.37% |
| Random Forest | 95.61% |
| Decision Tree | 91.23% |

---

## Metric 2: Precision

### What It Is

Of all patients we predicted as malignant, how many actually had cancer?

### Formula

```
Precision = TP / (TP + FP)
          = True Positives / All Positive Predictions
```

### Our Example

```
Precision = 71 / (71 + 2)
          = 71 / 73
          = 97.26%
```

### Interpretation

"When we say malignant, we're right 97.26% of the time."

### When High Precision Matters

- When false positives are costly
- When you want to be confident in positive predictions
- Example: Spam filter (don't want to delete important emails)

### Our Results

| Model | Precision |
|-------|-----------|
| Logistic Regression | 98.61% |
| SVM | 98.61% |
| Ensemble | 97.26% |
| Random Forest | 95.89% |
| Decision Tree | 95.59% |

---

## Metric 3: Recall (Sensitivity)

### What It Is

Of all patients who actually had cancer, how many did we catch?

### Formula

```
Recall = TP / (TP + FN)
       = True Positives / All Actual Positives
```

### Our Example

```
Recall = 71 / (71 + 1)
       = 71 / 72
       = 98.61%
```

### Interpretation

"We correctly identified 98.61% of all cancer cases."

### When High Recall Matters

- When missing a positive is dangerous
- **Critical for medical diagnosis!**
- Example: Cancer screening (don't want to miss any cases)

### The Trade-off

Precision and Recall often have an inverse relationship:

```
To increase Recall:  Lower the threshold → Catch more positives
                     But: More false positives → Lower Precision

To increase Precision: Raise the threshold → Be more selective
                       But: Miss some positives → Lower Recall
```

### Our Results

| Model | Recall |
|-------|--------|
| Logistic Regression | 98.61% |
| SVM | 98.61% |
| Ensemble | 98.61% |
| Random Forest | 97.22% |
| Decision Tree | 90.28% |

---

## Metric 4: F1 Score

### What It Is

The harmonic mean of Precision and Recall - a balanced measure.

### Formula

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Why Harmonic Mean?

The harmonic mean penalizes extreme imbalances:

```
Precision = 100%, Recall = 0%
Arithmetic mean: (100 + 0) / 2 = 50%  ← Seems okay
Harmonic mean: 2 * (1.0 * 0) / (1.0 + 0) = 0%  ← Shows the problem!
```

### Our Example

```
F1 = 2 * (0.9726 * 0.9861) / (0.9726 + 0.9861)
   = 2 * 0.9589 / 1.9587
   = 97.93%
```

### When to Use F1

- When you need to balance Precision and Recall
- When false positives and false negatives are both important
- With imbalanced datasets

### Our Results

| Model | F1 Score |
|-------|----------|
| Logistic Regression | 98.61% |
| SVM | 98.61% |
| Ensemble | 97.93% |
| Random Forest | 96.55% |
| Decision Tree | 92.86% |

---

## Metric 5: ROC-AUC

### What It Is

ROC-AUC measures how well the model can distinguish between classes across all possible thresholds.

### Understanding the ROC Curve

The ROC (Receiver Operating Characteristic) curve plots:
- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis**: True Positive Rate (TPR) = TP / (TP + FN) = Recall

```
    TPR
    1.0 ┤        ╭────────────
        │      ╭─╯
        │    ╭─╯
        │  ╭─╯         Our Model (AUC = 0.995)
        │╭─╯
    0.5 ┤         ╱
        │       ╱
        │     ╱    Random Classifier (AUC = 0.5)
        │   ╱
        │ ╱
    0.0 ┼────────────────────
        0.0       0.5      1.0  FPR
```

### What AUC Means

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classifier |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Fair |
| 0.5 | Random guessing |
| < 0.5 | Worse than random! |

### Why ROC-AUC Is Useful

1. **Threshold-independent**: Evaluates all possible cutoff points
2. **Class imbalance resistant**: Works well with unequal class sizes
3. **Comparison friendly**: Easy to compare different models

### Our Results

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.9954 |
| Ensemble | 0.9954 |
| SVM | 0.9950 |
| Random Forest | 0.9939 |
| Decision Tree | 0.9157 |

All models except Decision Tree achieve AUC > 0.99 - excellent!

---

## Metric 6: Cross-Validation Score

### What It Is

Performance averaged across multiple train/test splits.

### How 5-Fold CV Works

```
Fold 1: [TEST][Train][Train][Train][Train]  → 96.5%
Fold 2: [Train][TEST][Train][Train][Train]  → 97.8%
Fold 3: [Train][Train][TEST][Train][Train]  → 97.2%
Fold 4: [Train][Train][Train][TEST][Train]  → 96.9%
Fold 5: [Train][Train][Train][Train][TEST]  → 98.1%
                                              ─────────
                                    Mean:     97.3%
                                    Std:      ±0.6%
```

### Why Cross-Validation Matters

- **Reduces luck**: Single split might be "easy" or "hard"
- **Shows consistency**: Low std = reliable model
- **Uses all data**: Every sample gets tested once

### Interpreting Results

```
Model A: 95.0% ± 1.0%  ← Consistent
Model B: 96.0% ± 5.0%  ← Inconsistent (risky!)

Model A might be better for deployment despite lower mean.
```

### Our Results

| Model | CV Mean | CV Std |
|-------|---------|--------|
| Logistic Regression | 98.02% | ±1.28% |
| SVM | 97.14% | ±1.79% |
| Ensemble | 96.92% | ±1.76% |
| Random Forest | 95.38% | ±2.35% |
| Decision Tree | 90.99% | ±1.89% |

---

## Metrics Summary Table

| Metric | Question It Answers | Good For |
|--------|---------------------|----------|
| **Accuracy** | What % of all predictions are correct? | Overall performance |
| **Precision** | When we say positive, how often are we right? | Avoiding false alarms |
| **Recall** | What % of actual positives do we catch? | Not missing cases |
| **F1 Score** | Balance of Precision and Recall? | Overall positive detection |
| **ROC-AUC** | How well do we separate classes? | Model comparison |
| **CV Score** | How consistent is performance? | Reliability |

---

## Which Metric Matters Most?

For breast cancer detection, prioritize in this order:

### 1. Recall (Most Important!)
- Missing cancer is dangerous
- We want to catch ALL cancer cases
- False negatives can be fatal

### 2. ROC-AUC
- Shows overall discrimination ability
- Threshold-independent evaluation

### 3. F1 Score
- Balances Recall with Precision
- Good overall measure

### 4. Precision
- Reduces unnecessary biopsies
- Important but secondary to Recall

### 5. Accuracy
- General overview
- Can be misleading with imbalanced data

---

## Our Final Results

Here's how our models perform across all metrics:

```
                    Accuracy  Precision  Recall    F1      AUC
                    ────────  ─────────  ──────  ──────  ──────
Logistic Regression  98.25%    98.61%   98.61%  98.61%  0.9954
SVM                  98.25%    98.61%   98.61%  98.61%  0.9950
Ensemble             97.37%    97.26%   98.61%  97.93%  0.9954
Random Forest        95.61%    95.89%   97.22%  96.55%  0.9939
Decision Tree        91.23%    95.59%   90.28%  92.86%  0.9157
```

### Key Observations

1. **Best Overall**: Logistic Regression and SVM (tied)
2. **Best Recall**: LR, SVM, and Ensemble all at 98.61%
3. **Most Consistent**: Logistic Regression (lowest CV std)
4. **Worst Performer**: Decision Tree (prone to overfitting)

---

## Practical Interpretation

For our Ensemble model:

```
Out of 114 test patients:
├── 111 diagnosed correctly (97.37% accuracy)
├── 71 cancer cases caught (98.61% recall)
├── Only 1 cancer case missed (false negative)
└── Only 2 healthy patients got false alarms (false positive)
```

This is excellent performance for a medical diagnostic tool!

---

**Next**: Learn how to [Interpret Results](07_RESULTS_INTERPRETATION.md) by reading our charts and outputs.

*See [Glossary](09_GLOSSARY.md) for any unfamiliar terms.*
