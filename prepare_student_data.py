"""Create a student-search CSV from the existing six-column dataset.

Input : student_data.csv
Output: student_data_with_ids.csv

Existing academic values and row order are preserved. Because the source CSV has
no real student identifiers, this script creates DEMO0001, DEMO0002, ... and
Student 001, Student 002, ... as placeholders. Replace these with authorised
real roll numbers/names before using it with real student records.
"""

from pathlib import Path
import pandas as pd

SOURCE = Path("student_data.csv")
OUTPUT = Path("student_data_with_ids.csv")
FEATURES = [
    "Attendance",
    "Study_Hours",
    "Assignment_Score",
    "Internal_Marks",
    "Lab_Score",
]
TARGET = "Final_Result"
REQUIRED = FEATURES + [TARGET]


def main():
    if not SOURCE.exists():
        raise SystemExit(
            "ERROR: student_data.csv was not found. Put this script beside your original CSV."
        )

    # The original dataset should be comma-separated, as in the provided example.
    df = pd.read_csv(SOURCE)
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise SystemExit(
            "ERROR: These required columns are missing from student_data.csv: "
            + ", ".join(missing)
        )

    # Keep every source row and all original columns/values.
    if "Roll_Number" not in df.columns:
        df.insert(0, "Roll_Number", [f"DEMO{i:04d}" for i in range(1, len(df) + 1)])
    else:
        df["Roll_Number"] = df["Roll_Number"].astype("string").str.strip()

    if "Student_Name" not in df.columns:
        df.insert(1, "Student_Name", [f"Student {i:03d}" for i in range(1, len(df) + 1)])

    df["Roll_Number"] = df["Roll_Number"].astype("string").str.strip()
    if df["Roll_Number"].isna().any() or (df["Roll_Number"] == "").any():
        raise SystemExit("ERROR: Roll_Number contains an empty value. Fill it before continuing.")
    if df["Roll_Number"].duplicated().any():
        duplicates = df.loc[df["Roll_Number"].duplicated(), "Roll_Number"].tolist()
        raise SystemExit("ERROR: Roll_Number values must be unique. Duplicates: " + str(duplicates[:10]))

    # Put identity fields first, followed by the original six columns.
    output_columns = ["Roll_Number", "Student_Name"] + REQUIRED
    df = df[output_columns]
    df.to_csv(OUTPUT, index=False)

    print("Student search CSV created successfully!")
    print(f"Original records preserved: {len(df)}")
    print(f"New file: {OUTPUT.resolve()}")
    print("Note: DEMO roll numbers and Student names are placeholders.")
    print("Replace them with the correct authorised student details before the review.")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
