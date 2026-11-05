"""
Smart Student Success Prediction and Analysis System
Algorithms: Decision Tree (prediction) + K-Means (grouping)
            + Recommendation Engine + Roll Number Search

Input CSV: student_data_with_ids.csv
Run:       python student_success.py
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "Attendance",
    "Study_Hours",
    "Assignment_Score",
    "Internal_Marks",
    "Lab_Score",
]
TARGET = "Final_Result"
CSV_PATH = "student_data_with_ids.csv"


# ---------- Step 1: Load and preprocess ----------
def load_and_clean(path=CSV_PATH):
    df = pd.read_csv(path, dtype={"Roll_Number": "string"})
    required = ["Roll_Number", "Student_Name"] + FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns in " + path + ": " + ", ".join(missing)
            + ". Run prepare_student_data.py first."
        )

    df = df.drop_duplicates().copy()
    df["Roll_Number"] = df["Roll_Number"].astype("string").str.strip()
    df = df.dropna(subset=["Roll_Number", TARGET])
    df = df[df["Roll_Number"] != ""].copy()

    if df["Roll_Number"].duplicated().any():
        raise ValueError("Each student must have a unique Roll_Number in the CSV.")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=FEATURES).copy()
    if df.empty:
        raise ValueError("No valid student records were found in the CSV.")

    allowed_results = {"Good", "Average", "Poor"}
    unexpected = set(df[TARGET].astype(str).unique()) - allowed_results
    if unexpected:
        raise ValueError("Unexpected Final_Result labels: " + ", ".join(sorted(unexpected)))

    return df


# ---------- Step 2: Decision Tree (supervised) ----------
def train_decision_tree(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)

    print("\n========== MODEL PERFORMANCE ==========")
    print("Decision Tree accuracy:", round(accuracy_score(y_test, pred) * 100, 1), "%")
    print("\nClassification report:")
    print(classification_report(y_test, pred))
    print(
        "Confusion matrix (rows = actual):\n",
        confusion_matrix(y_test, pred, labels=["Good", "Average", "Poor"]),
    )
    print(
        "Feature importance:",
        dict(zip(FEATURES, tree.feature_importances_.round(3))),
)
    return tree


# ---------- Step 3: K-Means (unsupervised) ----------
def train_kmeans(df, k=3):
    scaler = StandardScaler().fit(df[FEATURES])
    Z = scaler.transform(df[FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)

    print("K-Means silhouette score:", round(silhouette_score(Z, km.labels_), 3))

    # Preserve the original project's cluster naming method.
    order = np.argsort([-Z[km.labels_ == c].mean() for c in range(k)])
    names = {
        int(order[0]): "Group A (High Achiever)",
        int(order[1]): "Group B (Average)",
        int(order[2]): "Group C (At-Risk)",
    }
    return km, scaler, names


# ---------- Step 4: Recommendation engine ----------
def recommend(student, prediction, cluster_name):
    tips = []
    if student["Attendance"] < 75:
        tips.append("Improve attendance to at least 75%.")
    if student["Study_Hours"] < 3:
        tips.append("Increase daily study time to 3+ hours.")
    if student["Assignment_Score"] < 60:
        tips.append("Submit assignments regularly and revise weak topics.")
    if student["Internal_Marks"] < 60:
        tips.append("Practise previous internal papers.")
    if student["Lab_Score"] < 60:
        tips.append("Spend extra time on lab practicals.")
    if not tips:
        tips.append("Keep up the good work. Try advanced projects.")
    if "At-Risk" in cluster_name or prediction == "Poor":
        tips.append("Meet your mentor for a personalised support plan.")
    return tips


# ---------- Step 5: Predict for one student ----------
def analyse_student(student, tree, km, scaler, names):
    row = pd.DataFrame([student], columns=FEATURES)
    prediction = tree.predict(row)[0]
    cluster_id = int(km.predict(scaler.transform(row))[0])
    cluster_name = names[cluster_id]
    return prediction, cluster_name, recommend(student, prediction, cluster_name)


# ---------- Step 6: Search by roll number ----------
def search_students(df, tree, km, scaler, names):
    while True:
        print("\n========== STUDENT SEARCH ==========")
        roll = input("Enter Roll Number (or Q to quit): ").strip()

        if roll.upper() == "Q":
            print("Exiting Student Success Prediction System.")
            break

        matches = df[df["Roll_Number"].astype(str).str.strip() == roll]
        if matches.empty:
            print("Student not found. Please check the Roll Number.")
            continue
        if len(matches) > 1:
            print("More than one record has this Roll Number. Make IDs unique in the CSV.")
            continue

        selected = matches.iloc[0]
        student = {feature: float(selected[feature]) for feature in FEATURES}
        prediction, cluster, tips = analyse_student(student, tree, km, scaler, names)

        print("\n========== STUDENT DETAILS ==========")
        print("Roll Number:", selected["Roll_Number"])
        print("Student Name:", selected["Student_Name"])
        for feature, value in student.items():
            print(f"{feature}: {value:g}")

        print("\n========== MODEL OUTPUT ==========")
        print("Predicted result:", prediction)
        print("Student group:", cluster)
        print("Recorded Final_Result in dataset:", selected[TARGET])

        print("\n========== RECOMMENDATIONS ==========")
        for tip in tips:
            print("-", tip)


# ---------- Main program ----------
if __name__ == "__main__":
    try:
        data = load_and_clean()
        tree = train_decision_tree(data)
        km, scaler, names = train_kmeans(data)

        # Save trained objects for later reuse.
        joblib.dump((tree, km, scaler, names), "models.pkl")
        print("\nModels saved successfully as models.pkl")

        search_students(data, tree, km, scaler, names)

    except FileNotFoundError:
        print(
            f"ERROR: {CSV_PATH} was not found. First run: python prepare_student_data.py"
        )
    except (ValueError, KeyError) as error:
        print("ERROR:", error)


