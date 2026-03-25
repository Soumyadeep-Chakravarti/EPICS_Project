# Breast Cancer Basics

This document explains the medical context of our project and the dataset we use.

## What Is Breast Cancer?

Breast cancer occurs when cells in the breast grow uncontrollably. These abnormal cells can form a **tumor** - a lump of tissue.

### Two Types of Tumors

| Type | Description | Danger Level |
|------|-------------|--------------|
| **Benign** | Non-cancerous. Cells look relatively normal, don't spread to other parts of the body. | Low - usually not life-threatening |
| **Malignant** | Cancerous. Cells look abnormal, can invade nearby tissue and spread (metastasize). | High - requires treatment |

**Our system's job**: Look at cell measurements and predict whether a tumor is benign or malignant.

## How Is Breast Cancer Diagnosed?

In the real world, doctors use several methods:

1. **Mammogram**: X-ray of the breast
2. **Ultrasound**: Sound waves to create images
3. **Biopsy**: Remove a small tissue sample for examination
4. **Fine Needle Aspiration (FNA)**: Extract cells with a thin needle

Our dataset comes from **Fine Needle Aspiration** - a doctor inserts a thin needle into the tumor and extracts cells. These cells are then examined under a microscope.

## The WDBC Dataset

We use the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, one of the most famous datasets in machine learning.

### Quick Facts

| Property | Value |
|----------|-------|
| **Total Samples** | 569 patients |
| **Benign Cases** | 357 (62.7%) |
| **Malignant Cases** | 212 (37.3%) |
| **Features** | 30 measurements |
| **Source** | University of Wisconsin Hospitals |
| **Year** | 1995 |

### Why This Dataset?

- **Real medical data**: From actual patients, not synthetic
- **Well-documented**: Extensively studied and validated
- **Clean**: No missing values
- **Balanced enough**: Both classes have enough samples to learn from
- **Public**: Available in scikit-learn library

## The 30 Features Explained

Each sample has 30 numerical measurements. These come from analyzing the **cell nuclei** (the center of each cell) in the biopsy image.

### The 10 Base Measurements

For each cell nucleus, 10 characteristics are measured:

| # | Feature | What It Measures | Why It Matters |
|---|---------|-----------------|----------------|
| 1 | **Radius** | Distance from center to edge | Cancer cells tend to be larger |
| 2 | **Texture** | Variation in gray-scale values | Cancer cells have irregular texture |
| 3 | **Perimeter** | Total boundary length | Related to size and shape |
| 4 | **Area** | Size of the nucleus | Cancer cells are often bigger |
| 5 | **Smoothness** | Local variation in radius | Cancer cells have irregular edges |
| 6 | **Compactness** | Perimeter² / Area - 1 | Measures how circular the shape is |
| 7 | **Concavity** | Severity of concave portions | Cancer cells have more indentations |
| 8 | **Concave Points** | Number of concave portions | More = more irregular shape |
| 9 | **Symmetry** | How symmetric the shape is | Cancer cells are less symmetric |
| 10 | **Fractal Dimension** | "Coastline" complexity | Measures boundary irregularity |

### Three Statistics for Each Measurement

For each of the 10 base measurements, we calculate three statistics:

| Statistic | What It Means | Example |
|-----------|--------------|---------|
| **Mean** | Average value across all cells in the sample | `radius_mean` |
| **Standard Error (SE)** | How much the values vary | `radius_se` |
| **Worst** | Largest/most severe value (mean of 3 worst cells) | `radius_worst` |

This gives us: **10 measurements × 3 statistics = 30 features**

### Full Feature List

```
Mean features (1-10):
  radius_mean, texture_mean, perimeter_mean, area_mean,
  smoothness_mean, compactness_mean, concavity_mean,
  concave_points_mean, symmetry_mean, fractal_dimension_mean

Standard Error features (11-20):
  radius_se, texture_se, perimeter_se, area_se,
  smoothness_se, compactness_se, concavity_se,
  concave_points_se, symmetry_se, fractal_dimension_se

Worst features (21-30):
  radius_worst, texture_worst, perimeter_worst, area_worst,
  smoothness_worst, compactness_worst, concavity_worst,
  concave_points_worst, symmetry_worst, fractal_dimension_worst
```

## What Do Cancer Cells Look Like?

In general, malignant (cancerous) cells tend to have:

- **Larger size** (bigger radius, area, perimeter)
- **More irregular shape** (higher concavity, concave points)
- **Less symmetry**
- **More variation** (higher standard error values)
- **Rougher texture**

Our machine learning models learn these patterns from the data!

## Most Important Features

Based on our Random Forest analysis, these features are most useful for prediction:

1. **worst concave points** - Number of indentations in the worst cells
2. **worst perimeter** - Boundary length of the worst cells
3. **worst radius** - Size of the worst cells
4. **mean concave points** - Average number of indentations
5. **worst area** - Area of the worst cells

Notice how **"worst"** features (the most abnormal cells) are very important - this makes sense because cancer is detected by finding the most abnormal cells!

## Class Distribution

Our dataset has more benign cases than malignant:

```
Benign (0):     ████████████████████████████████████  357 (62.7%)
Malignant (1):  █████████████████████                 212 (37.3%)
                ─────────────────────────────────────
                0        100       200       300       400
```

This is called **class imbalance** - but it's not severe enough to cause problems. We use **stratified splitting** to ensure both training and testing sets maintain this ratio.

## How Is This Used in Our Code?

In `data_loader.py`:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  # 569 x 30
y = pd.Series(data.target)  # 569 labels (0 or 1)
```

That's it! The dataset is built into scikit-learn, so we don't need external files.

## Real-World Considerations

While our system achieves 98.25% accuracy, it's important to understand:

1. **Not a replacement for doctors**: This is a decision-support tool, not a diagnostic device
2. **False negatives are dangerous**: Missing a cancer case (predicting benign when it's malignant) is worse than a false alarm
3. **Requires proper validation**: Real medical AI needs clinical trials and regulatory approval

That's why we focus on **Recall** (catching all cancer cases) in addition to accuracy.

---

**Next**: Learn the [ML Concepts](03_ML_CONCEPTS.md) that power our predictions.

*See [Glossary](09_GLOSSARY.md) for any unfamiliar terms.*
