# Code Walkthrough

This document explains what each file in the project does and how data flows through the system.

## Project Structure

```
EPICS_Project/
├── main.py                 # Entry point - runs the entire pipeline
├── src/                    # Source code modules
│   ├── __init__.py         # Makes src a Python package
│   ├── config.py           # Configuration and logging
│   ├── data_loader.py      # Loads the dataset
│   ├── preprocess.py       # Splits and scales data
│   ├── models.py           # Creates ML models
│   ├── ensemble.py         # Combines models
│   ├── evaluate.py         # Measures performance
│   └── visualize.py        # Generates charts
├── results/                # Output visualizations and metrics
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks
├── report/                 # LaTeX report
└── docs/                   # Documentation (you are here!)
```

## Data Flow

Here's how data moves through the system:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           main.py                                    │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ data_loader  │───>│  preprocess  │───>│       models         │  │
│  │  (Load data) │    │ (Split/Scale)│    │ (Train 4 classifiers)│  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                    │                 │
│                                                    v                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  visualize   │<───│   evaluate   │<───│      ensemble        │  │
│  │ (Make plots) │    │  (Metrics)   │    │  (Combine models)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Explanation

### 1. `main.py` - The Pipeline Orchestrator

**Purpose**: Entry point that coordinates all other modules.

**Key Class**: `Pipeline`

```python
class Pipeline:
    def __init__(self):
        # Initialize all components
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.model_factory = ModelFactory()
        self.ensemble_builder = EnsembleBuilder()
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
    
    def run(self):
        # Execute the complete workflow
        # 1. Load data
        # 2. Preprocess
        # 3. Train models
        # 4. Build ensemble
        # 5. Evaluate
        # 6. Visualize
        # 7. Save results
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `run()` | Executes the entire ML pipeline |
| `display_results()` | Prints metrics to terminal |
| `display_confusion_matrix()` | Shows ASCII confusion matrix |

**How to Run**:
```bash
python main.py
```

**Output Steps**:
```
[1/8] Loading data...
[2/8] Preprocessing data...
[3/8] Initializing models...
[4/8] Training and evaluating models...
[5/8] Building ensemble model...
[6/8] Running sample prediction...
[7/8] Saving results...
[8/8] Generating visualizations...
```

---

### 2. `src/config.py` - Configuration Management

**Purpose**: Centralizes all settings in one place.

**Key Classes**:

```python
@dataclass
class ModelConfig:
    """Model hyperparameters"""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Logistic Regression
    lr_max_iter: int = 5000
    lr_C: float = 1.0
    
    # Decision Tree
    dt_max_depth: int | None = None
    
    # SVM
    svm_C: float = 1.0
    svm_kernel: str = "rbf"
    
    # Random Forest
    rf_n_estimators: int = 100

@dataclass
class PathConfig:
    """File paths for outputs"""
    results_dir: str = "results"
    models_dir: str = "models"
    # ... more paths
```

**Global Instances**:
```python
config = Config()      # Access settings anywhere
logger = setup_logging()  # Logging instance
```

**How to Modify Settings**:
```python
from src.config import config

# Change test size
config.model.test_size = 0.3

# Change Random Forest trees
config.model.rf_n_estimators = 200
```

---

### 3. `src/data_loader.py` - Dataset Loading

**Purpose**: Loads the Wisconsin Breast Cancer dataset.

**Key Class**: `DataLoader`

```python
class DataLoader:
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the breast cancer dataset.
        
        Returns:
            X: Features (569 samples, 30 features)
            y: Labels (0=benign, 1=malignant)
        """
        data = load_breast_cancer()  # From sklearn
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return X, y
```

**Why a Separate Module?**:
- Easy to swap datasets in the future
- Clean separation of concerns
- Could add data validation or custom loading

---

### 4. `src/preprocess.py` - Data Preprocessing

**Purpose**: Splits data and applies feature scaling.

**Key Class**: `Preprocessor`

```python
class Preprocessor:
    def __init__(self):
        self.scaler = None  # Will hold StandardScaler
    
    def split_and_scale(self, X, y):
        """
        1. Split into train/test (80/20)
        2. Fit scaler on training data
        3. Transform both sets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y  # Maintain class balance
        )
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)  # Fit + transform
        X_test = self.scaler.transform(X_test)        # Only transform
        
        return X_train, X_test, y_train, y_test
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `split_and_scale()` | Main preprocessing function |
| `save_scaler()` | Save scaler to `models/scaler.joblib` |
| `load_scaler()` | Load saved scaler |
| `transform()` | Transform new data using fitted scaler |

**Why Stratified Split?**

```
Without stratify:     With stratify:
Train: 70% benign     Train: 62.7% benign  ← Same as original
Test:  50% benign     Test:  62.7% benign  ← Same as original
```

**Why Scale Only on Training Data?**

```python
# CORRECT:
scaler.fit(X_train)           # Learn mean/std from training
scaler.transform(X_train)     # Apply to training
scaler.transform(X_test)      # Apply same transform to test

# WRONG (data leakage!):
scaler.fit(X_all)             # Uses test data info during training!
```

---

### 5. `src/models.py` - Model Factory

**Purpose**: Creates and configures all ML models.

**Key Class**: `ModelFactory`

```python
class ModelFactory:
    def get_models(self) -> Dict[str, BaseEstimator]:
        """Returns dictionary of configured models."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=5000,
                C=1.0,
                random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=None,
                min_samples_split=2,
                random_state=42
            ),
            "SVM": SVC(
                C=1.0,
                kernel="rbf",
                probability=True,  # Needed for soft voting
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42
            ),
        }
```

**Why `probability=True` for SVM?**

By default, SVM only gives class predictions. We need `probability=True` to:
1. Use soft voting in the ensemble
2. Calculate ROC-AUC scores
3. Get confidence scores

---

### 6. `src/ensemble.py` - Ensemble Builder

**Purpose**: Combines multiple models into a voting classifier.

**Key Class**: `EnsembleBuilder`

```python
class EnsembleBuilder:
    def build(self, models, voting="soft"):
        """
        Create a voting classifier from models.
        
        Args:
            models: Dict of name -> model
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
        """
        estimators = [(name, model) for name, model in models.items()]
        
        return VotingClassifier(
            estimators=estimators,
            voting=voting
        )
    
    def save(self, ensemble, path=None):
        """Save trained ensemble to disk."""
        joblib.dump(ensemble, path or "models/ensemble.joblib")
    
    def load(self, path=None):
        """Load saved ensemble from disk."""
        return joblib.load(path or "models/ensemble.joblib")
```

**Soft vs Hard Voting**:

```
Hard Voting: Each model votes, majority wins
  LR: Malignant, DT: Benign, SVM: Malignant, RF: Malignant
  Result: Malignant (3-1)

Soft Voting: Average probability scores
  LR: 0.95, DT: 0.45, SVM: 0.92, RF: 0.88
  Average: 0.80 → Malignant
```

---

### 7. `src/evaluate.py` - Model Evaluation

**Purpose**: Computes all performance metrics.

**Key Class**: `Evaluator`

```python
class Evaluator:
    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Compute standard classification metrics.
        
        Returns dict with:
        - accuracy
        - precision
        - recall
        - f1_score
        - roc_auc
        """
        predictions = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
        }
        
        # Add ROC-AUC if model supports probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        
        return metrics
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `evaluate()` | Compute accuracy, precision, recall, F1, ROC-AUC |
| `cross_validate()` | Perform k-fold cross-validation |
| `get_classification_report()` | Detailed per-class metrics |
| `get_confusion_matrix()` | Returns confusion matrix array |
| `get_roc_data()` | Returns FPR, TPR for ROC plotting |
| `predict_with_confidence()` | Returns prediction + confidence score |

---

### 8. `src/visualize.py` - Visualization

**Purpose**: Generates all charts and plots.

**Key Class**: `Visualizer`

```python
class Visualizer:
    def save_confusion_matrix(self, model, X_test, y_test):
        """Create and save confusion matrix heatmap."""
        # Uses seaborn heatmap
        
    def save_model_comparison(self, metrics_dict):
        """Bar chart comparing all models across metrics."""
        
    def save_feature_importance(self, model, feature_names):
        """Horizontal bar chart of top features."""
        
    def save_dataset_distribution(self, y):
        """Bar chart of benign vs malignant counts."""
        
    def save_roc_curves(self, roc_data):
        """ROC curves for all models on one plot."""
        
    def save_cv_scores(self, cv_results):
        """Cross-validation scores with error bars."""
```

**Generated Visualizations**:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Shows TP, TN, FP, FN |
| `model_comparison.png` | All metrics for all models |
| `feature_importance.png` | Top 10 important features |
| `dataset_distribution.png` | Class balance |
| `roc_curves.png` | ROC curves comparison |
| `cv_scores.png` | Cross-validation accuracy |

---

## Complete Workflow Example

Here's what happens when you run `python main.py`:

```python
# 1. Initialize
pipeline = Pipeline()

# 2. Load data (data_loader.py)
X, y = loader.load_data()
# X: DataFrame (569, 30)
# y: Series (569,) with 0s and 1s

# 3. Preprocess (preprocess.py)
X_train, X_test, y_train, y_test = preprocessor.split_and_scale(X, y)
# X_train: (455, 30) scaled
# X_test: (114, 30) scaled

# 4. Create models (models.py)
models = model_factory.get_models()
# {"Logistic Regression": LR, "Decision Tree": DT, ...}

# 5. Train and evaluate each model (evaluate.py)
for name, model in models.items():
    model.fit(X_train, y_train)
    metrics[name] = evaluator.evaluate(model, X_test, y_test)
    cv_results[name] = evaluator.cross_validate(model, X_train, y_train)

# 6. Build ensemble (ensemble.py)
ensemble = ensemble_builder.build(models)
ensemble.fit(X_train, y_train)
metrics["Ensemble"] = evaluator.evaluate(ensemble, X_test, y_test)

# 7. Save models
ensemble_builder.save(ensemble)
preprocessor.save_scaler()

# 8. Generate visualizations (visualize.py)
visualizer.save_confusion_matrix(ensemble, X_test, y_test)
visualizer.save_model_comparison(metrics)
visualizer.save_feature_importance(models["Random Forest"], X.columns)
visualizer.save_roc_curves(roc_data)
# ...
```

---

## Using Saved Models

After training, you can load and use the models:

```python
import joblib
from src.preprocess import Preprocessor

# Load the trained ensemble
ensemble = joblib.load("models/ensemble.joblib")

# Load the scaler
preprocessor = Preprocessor()
preprocessor.load_scaler()

# Prepare new data (must have same 30 features!)
new_data = [...]  # Your new sample

# Scale the data
new_data_scaled = preprocessor.transform(new_data)

# Make prediction
prediction = ensemble.predict(new_data_scaled)
probability = ensemble.predict_proba(new_data_scaled)

print(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
print(f"Confidence: {probability[0].max():.2%}")
```

---

## Key Design Patterns Used

### 1. Separation of Concerns
Each module has one job:
- `data_loader.py` → Load data
- `preprocess.py` → Transform data
- `models.py` → Create models
- etc.

### 2. Configuration as Code
All settings in `config.py`:
- Easy to change parameters
- No magic numbers in code
- Reproducible experiments

### 3. Factory Pattern
`ModelFactory` creates models:
- Centralized model creation
- Easy to add new models
- Consistent configuration

### 4. Dependency Injection
`Pipeline` receives components:
- Easy to test with mock objects
- Flexible architecture

---

**Next**: Learn about [Evaluation Metrics](06_EVALUATION_METRICS.md) to understand how we measure success.

*See [Glossary](09_GLOSSARY.md) for any unfamiliar terms.*
