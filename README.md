Smart Student Success Prediction and Analysis System
A student-performance analysis project using a Decision Tree classifier, K-Means clustering, and a rule-based recommendation engine.
What the project does
- Predicts Final_Result as Good, Average, or Poor using five academic features.
- Groups students into three K-Means clusters, named High Achiever, Average, and At-Risk based on cluster-centre averages.
- Gives suggestions based on attendance, study hours, assignment score, internal marks, lab score, and prediction.
- Supports searching for a student by Roll_Number in the prepared student CSV.
- Generates four chart images using make_charts.py.
Project files
- student_success.py — trains/evaluates the models, saves models.pkl, and searches students by roll number.
- prepare_student_data.py — creates student_data_with_ids.csv from the original six-column CSV. Since the original dataset has no names or roll numbers, it creates demonstration placeholders such as DEMO0001 and Student 001.
- make_charts.py — creates confusion-matrix, feature-importance, K-Means-cluster, and Decision Tree images from student_data.csv.
- student_data.csv — original academic dataset with columns Attendance, Study_Hours, Assignment_Score, Internal_Marks, Lab_Score, Final_Result.
- student_data_with_ids.csv — prepared dataset used by student_success.py, with Roll_Number and Student_Name added before the original six columns.
Requirements
Python 3.10 or newer is recommended.

Install dependencies:
pip install -r requirements.txt
Run the project
Place the Python files and original student_data.csv in the same folder. Then run:
python prepare_student_data.py
python student_success.py
The first command creates student_data_with_ids.csv. The second trains/evaluates the models and asks for a roll number. Enter a roll number such as DEMO0001 to search, or enter Q to quit.

Generate charts separately:
python make_charts.py
The chart files are saved in the same folder: confusion_matrix.png, feature_importance.png, kmeans_clusters.png, and decision_tree.png.
Machine-learning details
Decision Tree features: Attendance, Study_Hours, Assignment_Score, Internal_Marks, and Lab_Score.

Target column: Final_Result (Good, Average, or Poor).

The Decision Tree uses a 75% training / 25% test split, random_state=42, and max_depth=4. K-Means uses three clusters, standard-scaled features, n_init=10, and random_state=42.

Roll_Number and Student_Name are only used to identify and display records. They are not model input features.
Data and privacy note
The supplied source CSV has no real student identities. The preparation script therefore adds placeholder roll numbers and names for demonstration. Do not publish real student names, roll numbers, marks, attendance, or other identifiable education records in a public repository. Use synthetic or properly anonymised data for a public GitHub repository.
Limitations
The prediction is a model estimate, not a guaranteed academic result. Accuracy depends on the quality and representativeness of the dataset. K-Means cluster names are assigned from average scaled performance; groups may overlap.

The current implementation is a command-line Python program. It does not currently include a Streamlit dashboard or SQLite databas
