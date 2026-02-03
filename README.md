# task-11
AI & ML Internship – Task 11
SVM – Breast Cancer Classification

1. Introduction
Breast cancer classification is a supervised machine learning problem where the goal is to predict whether a tumor is malignant or benign based on medical features.
In this task, Support Vector Machine (SVM) is used to perform classification. Both linear and RBF kernels are explored, and hyperparameters are tuned using GridSearchCV.

2. Tools Used
•	Python
•	Scikit-learn
•	Matplotlib

3. Dataset
•	Primary Dataset: Sklearn Breast Cancer Dataset (load_breast_cancer())
•	Target values:
o	0 → Malignant
o	1 → Benign

4. Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

5. Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

6. Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Reason:
Stratification maintains the same class distribution in training and testing sets.

7. Baseline SVM – Linear Kernel
linear_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', probability=True))
])

linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

print(classification_report(y_test, y_pred_linear))

8. SVM with RBF Kernel
rbf_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])

rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

print(classification_report(y_test, y_pred_rbf))

9. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.01, 0.1, 1]
}

grid = GridSearchCV(
    Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True))
    ]),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_


10. Model Evaluation
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:\n", cm)

11. ROC Curve and AUC
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

12. Save Trained Model
joblib.dump(best_model, "svm_breast_cancer_model.pkl")

13. Final Outcome
•	Learned kernel-based classification using SVM
•	Compared linear and RBF kernels
•	Tuned hyperparameters using GridSearchCV
•	Evaluated performance using confusion matrix, ROC curve, and AUC



