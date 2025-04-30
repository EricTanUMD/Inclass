# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the dataset
df = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/Feature Engineering/final_cleaned_dataset_fixed.csv")

# 2. Clean the data

# Forcefully convert numeric columns
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any missing values (like row 3046)
df_cleaned = df.dropna()

# 3. Split into features and target
X = df_cleaned.drop(columns=["Target"])
y = df_cleaned["Target"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Build the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,  # 100 trees
    random_state=42,
    class_weight='balanced',  # Handle slight imbalance in dropout vs no-dropout
)

# 6. Cross-Validation BEFORE fitting on full training set
print("\n🔵 Performing 5-Fold Cross-Validation...")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# 7. Fit Random Forest on the full training set
rf_model.fit(X_train, y_train)

# 8. Predictions
y_pred = rf_model.predict(X_test)

# 9. Evaluation
print("\n🔵 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔵 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["No Dropout", "Dropout"], 
            yticklabels=["No Dropout", "Dropout"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 10. Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Top Features
top_features = X.columns[indices]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=top_features)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.show()

# 11. Print top features
print("\n🔵 Top 10 Most Important Features:")
for feature, importance in zip(top_features[:10], importances[indices][:10]):
    print(f"{feature}: {importance:.4f}")
