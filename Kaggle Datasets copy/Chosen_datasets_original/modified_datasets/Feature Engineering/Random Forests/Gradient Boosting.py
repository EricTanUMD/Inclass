# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

# 2. Load the dataset
df = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/Feature Engineering/final_cleaned_dataset_fixed.csv")

# 3. Clean the data
# Force numeric conversion (just in case)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any missing values
df_cleaned = df.dropna()

# 4. Binary classification only (assuming binary task like dropout 0/1)
df_binary = df_cleaned[df_cleaned["Target"].isin([0, 1])]

# 5. Define features (X) and target (y)
X = df_binary.drop(columns=["Target"])
y = df_binary["Target"]

# 6. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Build Gradient Boosting Model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# 8. Cross-Validation (on training set)
print("\n🔵 Performing 5-Fold Cross-Validation on Training Set...")
cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# 9. Train the model on the full training data
gb_model.fit(X_train, y_train)

# 10. Predictions
y_pred = gb_model.predict(X_test)

# 11. Evaluation
print("\n🔵 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔵 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=["No Dropout", "Dropout"], 
            yticklabels=["No Dropout", "Dropout"])
plt.title("Confusion Matrix (Gradient Boosting)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. ROC Curve and AUC
y_prob = gb_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Gradient Boosting)')
plt.legend(loc="lower right")
plt.show()

# 13. Feature Importance
importances = gb_model.feature_importances_
indices = np.argsort(importances)[::-1]

top_features = X.columns[indices]

plt.figure(figsize=(12,8))
sns.barplot(x=importances[indices], y=top_features)
plt.title("Feature Importances (Gradient Boosting)")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.show()

# 14. Print Top 10 Features
print("\n🔵 Top 10 Most Important Features:")
for feature, importance in zip(top_features[:10], importances[indices][:10]):
    print(f"{feature}: {importance:.4f}")
