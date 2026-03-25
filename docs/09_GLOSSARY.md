# Glossary

Quick reference for technical terms used in this project.

---

## A

### Accuracy
The percentage of all predictions that were correct.
```
Accuracy = (TP + TN) / Total
```
*Our best: 98.25%*

### AUC (Area Under Curve)
The area under the ROC curve. Measures overall model discrimination ability. Range: 0 to 1, where 1 is perfect.
*Our best: 0.9954*

---

## B

### Bagging (Bootstrap Aggregating)
Training multiple models on random subsets of the data and combining their predictions. Used by Random Forest.

### Benign
A non-cancerous tumor. In our dataset, coded as `0`.

### Binary Classification
Classification with exactly two possible outcomes (e.g., benign vs malignant).

### Bootstrap Sample
A random sample drawn with replacement from the original dataset. Each sample may contain duplicates.

---

## C

### Classification
The task of predicting which category a data point belongs to.

### Classification Report
A detailed summary showing precision, recall, and F1-score for each class.

### Confusion Matrix
A table showing the counts of true positives, true negatives, false positives, and false negatives.

### Cross-Validation (CV)
A technique to evaluate model performance by splitting data into multiple train/test sets and averaging results.
*We use 5-fold CV*

---

## D

### Data Leakage
When information from the test set accidentally influences training, leading to overly optimistic results.

### Decision Boundary
The line (or surface) that separates different classes in the feature space.

### Decision Tree
A model that makes predictions by asking a series of yes/no questions about features.

---

## E

### Ensemble
A model that combines multiple base models to make predictions.

### Estimator
Scikit-learn's term for any object that can learn from data (fit) and make predictions (predict).

---

## F

### F1 Score
The harmonic mean of precision and recall. Balances both metrics.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
*Our best: 98.61%*

### False Negative (FN)
A positive case (malignant) incorrectly predicted as negative (benign). **Critical error in medical diagnosis.**
*Our ensemble: 1 FN*

### False Positive (FP)
A negative case (benign) incorrectly predicted as positive (malignant).
*Our ensemble: 2 FP*

### Feature
An input variable used to make predictions. We have 30 features (radius, texture, area, etc.).

### Feature Importance
A score indicating how useful each feature is for making predictions.

### Feature Scaling
Transforming features to a common scale. We use StandardScaler (z-score normalization).

### Fine Needle Aspiration (FNA)
A medical procedure where cells are extracted using a thin needle. Source of our dataset.

### Fit
Training a model on data. `model.fit(X_train, y_train)`

---

## G

### Generalization
A model's ability to perform well on unseen data, not just training data.

### Gini Impurity
A measure of how mixed the classes are at a decision tree node. Lower = purer.
```
Gini = 1 - p(benign)² - p(malignant)²
```

---

## H

### Hard Voting
Ensemble method where each model votes for a class, and the majority wins.

### Hyperparameter
A setting configured before training (e.g., number of trees, learning rate).

---

## I

### Imbalanced Dataset
A dataset where one class has significantly more samples than another.
*Our data: 62.7% benign, 37.3% malignant (mild imbalance)*

---

## J

### joblib
Python library for saving and loading models. Used to save `ensemble.joblib` and `scaler.joblib`.

---

## K

### K-Fold Cross-Validation
Splitting data into K parts, using each part once for testing while training on the rest.
*We use K=5*

### Kernel
A function that transforms data into a higher dimension. SVM uses the RBF kernel.

---

## L

### Label
The target variable we're trying to predict (0=benign, 1=malignant).

### Logistic Regression
A linear model that predicts probabilities using the sigmoid function.

---

## M

### Malignant
A cancerous tumor. In our dataset, coded as `1`.

### Margin
In SVM, the distance between the decision boundary and the nearest data points.

### Mean
Average value. One of three statistics computed for each base measurement.

---

## N

### Normalization
See Feature Scaling.

---

## O

### Overfitting
When a model memorizes training data but fails to generalize to new data. Signs: high training accuracy, low test accuracy.

---

## P

### Pipeline
A sequence of data processing steps. Our pipeline: Load → Preprocess → Train → Evaluate → Visualize.

### Precision
Of all positive predictions, what fraction were correct?
```
Precision = TP / (TP + FP)
```
*Our best: 98.61%*

### predict()
Method to get class predictions from a trained model.

### predict_proba()
Method to get probability predictions from a trained model.

---

## R

### Random Forest
An ensemble of decision trees trained on random data subsets with random feature selection.
*We use 100 trees*

### Random State
A seed for random number generation. Ensures reproducible results.
*We use 42*

### RBF (Radial Basis Function)
A kernel that measures similarity based on distance. Used by our SVM.

### Recall (Sensitivity)
Of all actual positive cases, what fraction did we catch?
```
Recall = TP / (TP + FN)
```
*Our best: 98.61%*

### ROC Curve
A plot of True Positive Rate vs False Positive Rate at various thresholds.

---

## S

### Sample
One data point. We have 569 samples (patients).

### scikit-learn (sklearn)
Python machine learning library used for all our models.

### Sigmoid Function
S-shaped function that maps any value to a probability between 0 and 1.
```
σ(x) = 1 / (1 + e^(-x))
```

### Soft Voting
Ensemble method that averages probability predictions from all models.
*We use soft voting*

### Specificity
Of all actual negative cases, what fraction were correctly identified?
```
Specificity = TN / (TN + FP)
```

### Standard Deviation (Std)
A measure of spread. In cross-validation, lower std means more consistent performance.

### StandardScaler
Transforms features to have mean=0 and standard deviation=1.
```
z = (x - mean) / std
```

### Standard Error (SE)
A measure of variability. One of three statistics computed for each base measurement.

### Stratified Split
Splitting data while maintaining the same class proportions in each split.

### Support Vector Machine (SVM)
A model that finds the decision boundary with maximum margin between classes.

### Support Vectors
Data points closest to the decision boundary in SVM. They define the margin.

---

## T

### Test Set
Data held out to evaluate model performance. Not used during training.
*20% of our data = 114 samples*

### Training Set
Data used to teach the model.
*80% of our data = 455 samples*

### True Negative (TN)
A negative case (benign) correctly predicted as negative.
*Our ensemble: 40 TN*

### True Positive (TP)
A positive case (malignant) correctly predicted as positive.
*Our ensemble: 71 TP*

---

## U

### Underfitting
When a model is too simple to capture patterns in the data. Signs: low accuracy on both training and test sets.

---

## V

### Validation
The process of evaluating model performance.

### VotingClassifier
Scikit-learn class that combines multiple models using voting.

---

## W

### WDBC (Wisconsin Diagnostic Breast Cancer)
The dataset we use. 569 samples, 30 features.

### Worst
The mean of the three largest values for each measurement. One of three statistics computed.

---

## Quick Reference Table

| Term | Definition | Our Value |
|------|------------|-----------|
| Samples | Total data points | 569 |
| Features | Input variables | 30 |
| Test size | Fraction for testing | 20% |
| CV folds | Cross-validation splits | 5 |
| Best accuracy | Highest accuracy achieved | 98.25% |
| Best AUC | Highest ROC-AUC | 0.9954 |
| Random state | Reproducibility seed | 42 |
| RF trees | Random Forest estimators | 100 |

---

## Common Abbreviations

| Abbrev. | Full Form |
|---------|-----------|
| ML | Machine Learning |
| CV | Cross-Validation |
| LR | Logistic Regression |
| DT | Decision Tree |
| SVM | Support Vector Machine |
| RF | Random Forest |
| ROC | Receiver Operating Characteristic |
| AUC | Area Under Curve |
| TP | True Positive |
| TN | True Negative |
| FP | False Positive |
| FN | False Negative |
| FNA | Fine Needle Aspiration |
| WDBC | Wisconsin Diagnostic Breast Cancer |
| XAI | Explainable Artificial Intelligence |

---

*Return to [Documentation Home](README.md)*
