
"""
Smart Student Success Prediction and Analysis System
Chart Generation Script

Run: python make_charts.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


FEATURES = [
    "Attendance",
    "Study_Hours",
    "Assignment_Score",
    "Internal_Marks",
    "Lab_Score"
]

TARGET = "Final_Result"


# ---------- Step 1: Load and preprocess ----------
def load_and_clean(path="student_data.csv"):

    df = pd.read_csv(path)
    df = df.drop_duplicates()

    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[TARGET])

    return df


# ---------- Step 2: Train Decision Tree ----------
def main():

    df = load_and_clean()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    tree = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    tree.fit(X_train, y_train)

    predictions = tree.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Decision Tree accuracy:",
          round(accuracy * 100, 1), "%")


    # ---------- Step 3: Confusion Matrix Chart ----------
    class_labels = ["Good", "Average", "Poor"]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=class_labels
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Result")
    ax.set_ylabel("Actual Result")

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for row in range(3):
        for col in range(3):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black"
            )

    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=300)
    plt.close(fig)


    # ---------- Step 4: Feature Importance Chart ----------
    importances = tree.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(FEATURES, importances)

    ax.set_xlabel("Student Performance Factors")
    ax.set_ylabel("Importance Score")

    ax.set_title("Feature Importance - Decision Tree")

    ax.tick_params(axis="x", labelrotation=25)

    fig.tight_layout()
    fig.savefig("feature_importance.png", dpi=300)
    plt.close(fig)


    # ---------- Step 5: K-Means Clustering ----------
    scaler = StandardScaler().fit(X)

    Z = scaler.transform(X)

    k = 3

    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    ).fit(Z)

    order = np.argsort([
        -Z[km.labels_ == c].mean()
        for c in range(k)
    ])

    names = {
        order[0]: "Group A (High Achiever)",
        order[1]: "Group B (Average)",
        order[2]: "Group C (At-Risk)"
    }


    # ---------- Step 6: K-Means Cluster Chart ----------
    # PCA displays the five-feature clusters in two dimensions.

    reduced = PCA(
        n_components=2,
        random_state=42
    ).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(8, 5))

    for cluster_id in range(k):

        mask = km.labels_ == cluster_id

        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=names[cluster_id],
            alpha=0.75
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    ax.set_title("K-Means Student Clusters")

    ax.legend()

    fig.tight_layout()

    fig.savefig("kmeans_clusters.png", dpi=300)
    plt.close(fig)


    # ---------- Step 7: Decision Tree Visualization ----------
    fig, ax = plt.subplots(figsize=(18, 10))

    plot_tree(
        tree,
        feature_names=FEATURES,
        class_names=[str(label) for label in tree.classes_],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )

    ax.set_title("Trained Decision Tree")

    fig.tight_layout()

    fig.savefig("decision_tree.png", dpi=300)
    plt.close(fig)


    # ---------- Step 8: Display Results ----------
    print("\nCharts generated successfully!")

    print("Saved: confusion_matrix.png")
    print("Saved: feature_importance.png")
    print("Saved: kmeans_clusters.png")
    print("Saved: decision_tree.png")

    print("\nAll charts created using student_data.csv.")


if __name__ == "__main__":
    main()
