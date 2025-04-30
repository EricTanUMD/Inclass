# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 2. Load the dataset
df = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/Feature Engineering/final_cleaned_dataset_fixed.csv")

# 3. Clean the data: drop any remaining missing values (safety)
df_cleaned = df.dropna()

# 4. Filter data for Binary Classification ONLY (Target == 0 or 1)
df_binary = df_cleaned[df_cleaned["Target"].isin([0, 1])]

# 5. Define features (X) and target (y)
X = df_binary.drop(columns=["Target"])  # Drop the target column from features
y = df_binary["Target"]                 # Target variable (now binary only)

# 6. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Scale features (VERY important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Build Logistic Regression Model
log_reg = LogisticRegression(
    random_state=42, 
    max_iter=1000,           # Allow enough iterations
    solver='lbfgs',          # Good for binary
    penalty='l2',            # Regularization
    C=1.0                   # Regularization strength (can tune later)
)

# 9. Cross-Validation BEFORE final fitting
print("\n🔵 Performing 5-Fold Cross-Validation...")
cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# 10. Train the model on the full training data
log_reg.fit(X_train_scaled, y_train)

# 11. Predict on the test set
y_pred = log_reg.predict(X_test_scaled)

# 12. Evaluation: Classification Report
print("\n🔵 Classification Report:")
print(classification_report(y_test, y_pred))

# 13. Evaluation: Confusion Matrix
print("\n🔵 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=["No Dropout", "Dropout"], 
            yticklabels=["No Dropout", "Dropout"])
plt.title("Confusion Matrix (Binary Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 14. ROC Curve and AUC
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class 1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve (Binary Logistic Regression)")
plt.legend(loc="lower right")
plt.show()

# 15. Feature Importance: Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n🔵 Feature Coefficients (Importance):")
print(coefficients)

plt.figure(figsize=(12,8))
sns.barplot(x="Coefficient", y="Feature", data=coefficients)
plt.title("Feature Impact (Binary Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()
