# Project Overview

## What Did We Build?

We built a **machine learning system that detects breast cancer**. Given measurements from a cell sample, our system predicts whether the tumor is:

- **Benign** (non-cancerous, harmless)
- **Malignant** (cancerous, needs treatment)

Think of it like a very smart calculator that has learned from hundreds of examples what cancerous cells "look like" in terms of numbers.

## Why Does This Matter?

Breast cancer is one of the leading causes of death among women worldwide. The key to survival is **early detection**:

- If caught early: **99% survival rate**
- If caught late: survival rate drops significantly

Our system can help doctors by:
1. Providing a quick second opinion
2. Flagging suspicious cases for further review
3. Explaining WHY it made a prediction (which features mattered)

## How Does It Work? (The Big Picture)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Patient   │     │  Features   │     │   Machine   │     │ Prediction  │
│   Sample    │ --> │  Extracted  │ --> │  Learning   │ --> │  Benign or  │
│   (cells)   │     │ (30 numbers)│     │   Models    │     │  Malignant  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Step-by-Step:

1. **Data Collection**: We use a famous dataset (WDBC) with 569 real patient samples
2. **Feature Extraction**: Each sample has 30 measurements (size, shape, texture of cells)
3. **Data Preparation**: Split into training (80%) and testing (20%), normalize values
4. **Model Training**: Teach 4 different algorithms to recognize patterns
5. **Ensemble**: Combine all 4 models into one "committee" for better accuracy
6. **Evaluation**: Test on unseen data to measure real-world performance
7. **Visualization**: Generate charts to understand and explain results

## What Models Do We Use?

We use 4 different classification algorithms, each with its own approach:

| Model | Simple Explanation |
|-------|-------------------|
| **Logistic Regression** | Draws a mathematical line to separate classes |
| **Decision Tree** | Asks yes/no questions like a flowchart |
| **SVM** | Finds the widest possible gap between classes |
| **Random Forest** | 100 decision trees vote together |

Plus an **Ensemble** that combines all four for a final decision.

## Our Results

| Metric | Best Score | What It Means |
|--------|------------|---------------|
| **Accuracy** | 98.25% | 98 out of 100 predictions are correct |
| **ROC-AUC** | 0.997 | Almost perfect ranking ability |
| **Cross-Validation** | 97.14% | Consistent across different data splits |

The best performing models were **Logistic Regression** and **SVM**, both achieving 98.25% accuracy.

## Project Structure

Here's how our code is organized:

```
EPICS_Project/
│
├── main.py                 # The main script - run this to execute everything
│
├── src/                    # Source code (the brains)
│   ├── config.py           # Settings (test size, random seed, file paths)
│   ├── data_loader.py      # Loads the breast cancer dataset
│   ├── preprocess.py       # Splits and scales the data
│   ├── models.py           # Creates the 4 ML models
│   ├── ensemble.py         # Combines models into voting classifier
│   ├── evaluate.py         # Calculates accuracy, precision, recall, etc.
│   └── visualize.py        # Generates all the charts
│
├── results/                # Output folder
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── model_comparison.png
│   └── ... (more visualizations)
│
├── models/                 # Saved trained models
│   ├── ensemble.joblib     # The trained ensemble model
│   └── scaler.joblib       # The data normalizer
│
├── report/                 # LaTeX report for submission
│   ├── main.tex
│   └── build/main.pdf
│
└── docs/                   # You are here!
```

## Key Concepts to Understand

Before diving deeper, make sure you understand these terms:

- **Classification**: Sorting things into categories (benign vs malignant)
- **Training**: Teaching the model using known examples
- **Testing**: Checking how well the model works on new, unseen data
- **Features**: The input measurements (30 numbers per sample)
- **Labels**: The correct answers (0 = benign, 1 = malignant)

For detailed explanations, see [ML Concepts](03_ML_CONCEPTS.md).

## What Makes Our Project Good?

1. **High Accuracy**: 98.25% is excellent for medical diagnosis
2. **Explainable**: We show which features matter most
3. **Validated**: 5-fold cross-validation proves it's not just luck
4. **Production-Ready**: Saves models for reuse, has proper logging
5. **Well-Documented**: You're reading the proof!

## Next Steps

Now that you have the big picture:

1. Learn about the [medical context and dataset](02_BREAST_CANCER_BASICS.md)
2. Understand [ML fundamentals](03_ML_CONCEPTS.md)
3. See how [each model works](04_MODELS_EXPLAINED.md)

---

*See [Glossary](09_GLOSSARY.md) for any unfamiliar terms.*
