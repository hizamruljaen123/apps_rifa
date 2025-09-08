# Student Performance Classification Using C4.5

This project demonstrates how to classify **student performance** (e.g., High, Medium, Low) based on features such as attendance, grades, participation, and assignments using the **C4.5 decision tree algorithm**.

---

## Steps Overview
1. **Data Collection**:  
   - Gather student data: attendance, homework scores, exam scores, extracurricular participation, etc.

2. **Data Preprocessing**:  
   - Handle missing values.  
   - Encode categorical features (if any).  
   - Split dataset into **training** and **testing** sets.

3. **C4.5 Decision Tree**:  
   - Compute **information gain ratio** for attributes.  
   - Select attribute with the highest gain ratio as root.  
   - Recursively split data until stopping criteria (pure nodes or minimum samples) are met.

4. **Evaluation**:  
   - Measure accuracy, precision, recall, or F1-score on test data.  
   - Visualize the decision tree for interpretation.

---

## Python Implementation (Simplified)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Load student dataset
data = pd.read_csv('student_performance.csv')

# Features and target
X = data[['attendance', 'homework', 'exam', 'participation']]
y = data['performance']  # e.g., High, Medium, Low

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# C4.5-like Decision Tree (using entropy criterion)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Display tree rules
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
