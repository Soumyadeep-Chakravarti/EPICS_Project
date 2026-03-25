"""
Additional visualizations for the report.

This script generates supplementary plots including:
- Feature correlation heatmap
- Learning curves
- Precision-Recall curves
- Feature distribution plots
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10

# Create output directory
os.makedirs("report/results", exist_ok=True)

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def plot_correlation_heatmap():
    """Generate feature correlation heatmap."""
    # Select top 15 features by variance for readability
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(15).index.tolist()

    corr_matrix = X[top_features].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    plt.title("Feature Correlation Heatmap (Top 15 Features by Variance)", fontsize=14)
    plt.tight_layout()
    plt.savefig("report/results/correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Saved: correlation_heatmap.png")


def plot_learning_curves():
    """Generate learning curves for all models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    train_sizes = np.linspace(0.1, 1.0, 10)

    for idx, (name, model) in enumerate(models.items()):
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            X_train_scaled,
            y_train,
            train_sizes=train_sizes,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax = axes[idx]
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )
        ax.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1,
            color="orange",
        )
        ax.plot(train_sizes_abs, train_mean, "o-", color="blue", label="Training Score")
        ax.plot(
            train_sizes_abs, val_mean, "o-", color="orange", label="Validation Score"
        )

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve: {name}")
        ax.legend(loc="lower right")
        ax.set_ylim(0.85, 1.02)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Learning Curves - Model Convergence Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("report/results/learning_curves.png", bbox_inches="tight")
    plt.close()
    print("Saved: learning_curves.png")


def plot_precision_recall_curves():
    """Generate precision-recall curves."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    plt.figure(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        plt.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("report/results/precision_recall_curves.png")
    plt.close()
    print("Saved: precision_recall_curves.png")


def plot_feature_distributions():
    """Plot distribution of top features by class."""
    # Get top 6 features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    importance = rf.feature_importances_
    top_indices = importance.argsort()[-6:][::-1]
    top_features = [data.feature_names[i] for i in top_indices]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        benign = X[y == 1][feature]
        malignant = X[y == 0][feature]

        ax.hist(
            benign, bins=25, alpha=0.6, label="Benign", color="steelblue", density=True
        )
        ax.hist(
            malignant,
            bins=25,
            alpha=0.6,
            label="Malignant",
            color="coral",
            density=True,
        )

        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Feature Distributions by Class (Top 6 Features)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("report/results/feature_distributions.png", bbox_inches="tight")
    plt.close()
    print("Saved: feature_distributions.png")


def plot_boxplots():
    """Plot boxplots of top features by class."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    importance = rf.feature_importances_
    top_indices = importance.argsort()[-5:][::-1]
    top_features = [data.feature_names[i] for i in top_indices]

    # Create dataframe for plotting
    plot_df = X[top_features].copy()
    plot_df["Diagnosis"] = y.map({1: "Benign", 0: "Malignant"})

    # Normalize for comparison
    plot_df_melted = pd.melt(
        plot_df,
        id_vars=["Diagnosis"],
        value_vars=top_features,
        var_name="Feature",
        value_name="Value",
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=plot_df_melted,
        x="Feature",
        y="Value",
        hue="Diagnosis",
        palette=["coral", "steelblue"],
    )
    plt.xticks(rotation=15, ha="right")
    plt.title("Feature Value Distributions by Diagnosis (Top 5 Features)")
    plt.tight_layout()
    plt.savefig("report/results/feature_boxplots.png")
    plt.close()
    print("Saved: feature_boxplots.png")


def plot_error_analysis():
    """Analyze and visualize misclassified samples."""
    from sklearn.ensemble import VotingClassifier

    models = [
        ("lr", LogisticRegression(max_iter=5000, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train_scaled, y_train)

    predictions = ensemble.predict(X_test_scaled)
    probabilities = ensemble.predict_proba(X_test_scaled)

    # Find misclassified samples
    misclassified_mask = predictions != y_test.values
    correct_mask = ~misclassified_mask

    # Confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confidence for correct vs incorrect predictions
    correct_conf = probabilities[correct_mask].max(axis=1)
    incorrect_conf = probabilities[misclassified_mask].max(axis=1)

    ax = axes[0]
    ax.hist(
        correct_conf,
        bins=20,
        alpha=0.7,
        label=f"Correct (n={len(correct_conf)})",
        color="green",
    )
    ax.hist(
        incorrect_conf,
        bins=10,
        alpha=0.7,
        label=f"Incorrect (n={len(incorrect_conf)})",
        color="red",
    )
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Incorrect")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plot of misclassified samples
    ax = axes[1]
    X_test_df = pd.DataFrame(X_test_scaled, columns=data.feature_names)

    # Use top 2 features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    top2 = rf.feature_importances_.argsort()[-2:][::-1]
    feat1, feat2 = data.feature_names[top2[0]], data.feature_names[top2[1]]

    # Plot all points
    scatter = ax.scatter(
        X_test_df.iloc[correct_mask, top2[0]],
        X_test_df.iloc[correct_mask, top2[1]],
        c=y_test.values[correct_mask],
        cmap="coolwarm",
        alpha=0.5,
        s=50,
        label="Correct",
    )
    ax.scatter(
        X_test_df.iloc[misclassified_mask, top2[0]],
        X_test_df.iloc[misclassified_mask, top2[1]],
        c="black",
        marker="x",
        s=100,
        linewidths=2,
        label="Misclassified",
    )
    ax.set_xlabel(feat1.replace("_", " ").title())
    ax.set_ylabel(feat2.replace("_", " ").title())
    ax.set_title("Misclassified Samples in Feature Space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Error Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("report/results/error_analysis.png", bbox_inches="tight")
    plt.close()
    print("Saved: error_analysis.png")


if __name__ == "__main__":
    print("Generating additional visualizations...")
    plot_correlation_heatmap()
    plot_learning_curves()
    plot_precision_recall_curves()
    plot_feature_distributions()
    plot_boxplots()
    plot_error_analysis()
    print("\nAll visualizations generated successfully!")
