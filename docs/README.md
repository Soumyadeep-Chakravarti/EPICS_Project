# Breast Cancer Detection - Team Documentation

Welcome! This documentation is written for our team to understand the project. Whether you have an ML background or not, you'll find everything you need here.

## Who Is This For?

- Team members who want to understand how the project works
- Anyone preparing to explain the project (presentations, viva, etc.)
- Future maintainers of this codebase

## How to Read This Documentation

### If You're New to Machine Learning (Recommended Order)

1. [Project Overview](01_PROJECT_OVERVIEW.md) - Start here! The big picture
2. [Breast Cancer Basics](02_BREAST_CANCER_BASICS.md) - The medical context and dataset
3. [ML Concepts](03_ML_CONCEPTS.md) - Machine learning fundamentals
4. [Models Explained](04_MODELS_EXPLAINED.md) - Each algorithm in simple terms
5. [Evaluation Metrics](06_EVALUATION_METRICS.md) - How we measure success
6. [Code Walkthrough](05_CODE_WALKTHROUGH.md) - What each file does
7. [Results Interpretation](07_RESULTS_INTERPRETATION.md) - Understanding our outputs
8. [Report Guide](08_REPORT_GUIDE.md) - Section-by-section LaTeX report breakdown
9. [Glossary](09_GLOSSARY.md) - Quick reference for any term

### If You Already Know ML

Jump directly to what you need:
- [Code Walkthrough](05_CODE_WALKTHROUGH.md) - Architecture and implementation
- [Results Interpretation](07_RESULTS_INTERPRETATION.md) - Our findings
- [Glossary](09_GLOSSARY.md) - Quick term lookup

## Quick Links

| Document | What You'll Learn |
|----------|-------------------|
| [01 - Project Overview](01_PROJECT_OVERVIEW.md) | What we built, why, and how it fits together |
| [02 - Breast Cancer Basics](02_BREAST_CANCER_BASICS.md) | Medical context, dataset, features explained |
| [03 - ML Concepts](03_ML_CONCEPTS.md) | Training, testing, overfitting, scaling |
| [04 - Models Explained](04_MODELS_EXPLAINED.md) | LR, SVM, Decision Tree, Random Forest, Ensemble |
| [05 - Code Walkthrough](05_CODE_WALKTHROUGH.md) | What each `.py` file does, data flow |
| [06 - Evaluation Metrics](06_EVALUATION_METRICS.md) | Accuracy, Precision, Recall, F1, ROC-AUC |
| [07 - Results Interpretation](07_RESULTS_INTERPRETATION.md) | How to read our charts and numbers |
| [08 - Report Guide](08_REPORT_GUIDE.md) | Understanding the LaTeX report |
| [09 - Glossary](09_GLOSSARY.md) | A-Z technical terms |

## Our Results at a Glance

- **Best Accuracy**: 98.25% (Logistic Regression & SVM)
- **Dataset**: 569 samples, 30 features
- **Models**: 4 individual + 1 ensemble
- **Validation**: 5-fold cross-validation

## Project Structure

```
EPICS_Project/
├── main.py              # Run this to execute the pipeline
├── src/                 # Source code modules
│   ├── config.py        # Settings and logging
│   ├── data_loader.py   # Loads the dataset
│   ├── preprocess.py    # Splits and scales data
│   ├── models.py        # Creates ML models
│   ├── ensemble.py      # Combines models
│   ├── evaluate.py      # Measures performance
│   └── visualize.py     # Generates charts
├── results/             # Output visualizations
├── models/              # Saved trained models
├── report/              # LaTeX report files
└── docs/                # You are here!
```

## Need Help?

If something is unclear, check the [Glossary](09_GLOSSARY.md) first. If you're still stuck, ask a team member!

---

*Last updated: March 2026*
