# Breast Cancer Classification ML Pipeline

A machine learning pipeline for breast cancer classification using the Wisconsin Breast Cancer dataset. This project demonstrates a complete ML workflow including data preprocessing, model training, ensemble methods, and comprehensive evaluation.

## Features

- **Multiple ML Models**: Logistic Regression, Decision Tree, SVM, Random Forest
- **Ensemble Learning**: Soft voting classifier combining all models
- **Cross-Validation**: 5-fold CV for robust model evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Model Persistence**: Save and load trained models and scalers
- **Visualizations**: Confusion matrix, ROC curves, feature importance, model comparison
- **Configurable**: Centralized configuration for easy parameter tuning
- **Logging**: Proper logging for debugging and monitoring

## Project Structure

```
EPICS_Project/
├── main.py                 # Main pipeline orchestrator
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration and logging setup
│   ├── data_loader.py      # Dataset loading
│   ├── preprocess.py       # Data splitting and scaling
│   ├── models.py           # Model factory
│   ├── ensemble.py         # Ensemble model builder
│   ├── evaluate.py         # Evaluation metrics and CV
│   └── visualize.py        # Visualization utilities
├── results/                # Generated outputs
│   ├── metrics.csv
│   ├── metrics_pretty.txt
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   ├── cv_scores.png
│   └── dataset_distribution.png
├── models/                 # Saved models
│   ├── ensemble.joblib
│   └── scaler.joblib
├── notebooks/              # Jupyter notebooks (optional)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EPICS_Project
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This will:
1. Load the breast cancer dataset
2. Preprocess and split the data
3. Train all models with cross-validation
4. Build and evaluate the ensemble model
5. Generate visualizations
6. Save models and results

### Using Saved Models

```python
import joblib
from src.preprocess import Preprocessor

# Load saved model and scaler
ensemble = joblib.load("models/ensemble.joblib")
preprocessor = Preprocessor()
preprocessor.load_scaler()

# Make predictions on new data
X_new = ...  # Your new data
X_scaled = preprocessor.transform(X_new)
predictions = ensemble.predict(X_scaled)
```

### Configuration

Modify `src/config.py` to adjust:
- Model hyperparameters
- Train/test split ratio
- Cross-validation folds
- File paths

```python
from src.config import config

# Example: Change test size
config.model.test_size = 0.3

# Example: Adjust Random Forest parameters
config.model.rf_n_estimators = 200
```

## Results

The pipeline outputs include:

| Output | Description |
|--------|-------------|
| `metrics.csv` | Model metrics in CSV format |
| `metrics_pretty.txt` | Human-readable metrics summary |
| `classification_report.txt` | Detailed classification report |
| `confusion_matrix.png` | Confusion matrix visualization |
| `model_comparison.png` | Multi-metric model comparison |
| `roc_curves.png` | ROC curves for all models |
| `cv_scores.png` | Cross-validation scores |
| `feature_importance.png` | Top feature importances |

## Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classifier with L2 regularization |
| Decision Tree | Non-linear tree-based classifier |
| SVM | Support Vector Machine with RBF kernel |
| Random Forest | Ensemble of decision trees |
| Voting Ensemble | Soft voting combining all models |

## Dependencies

- scikit-learn
- pandas
- matplotlib
- seaborn
- joblib

## License

This project is for educational purposes.

## Author

EPICS Project Team
