# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Load your dataset
df = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/Feature Engineering/final_cleaned_dataset_fixed.csv")


# 2. Clean the dataset
# Force all numeric columns to be truly numeric, convert non-numeric to NaN
df_cleaned = df.copy()

for col in df_cleaned.columns:
    if df_cleaned[col].dtype in ['float64', 'int64']:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Drop rows with any missing values
df_cleaned = df_cleaned.dropna()

# Confirm no missing values
assert df_cleaned.isnull().sum().sum() == 0, "There are still missing values!"


# 3. Visualizations

## 3.1 Correlation Matrix
plt.figure(figsize=(20, 16))
sns.heatmap(df_cleaned.corr(), cmap="coolwarm", annot=False, fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of All Features (Cleaned Data)")
plt.show()

## 3.2 Target Variable Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df_cleaned, x="Target")
plt.title("Distribution of Target Variable (0 = Not Dropout, 1 = Dropout) [Cleaned Data]")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

## 3.3 Key Feature Distributions by Target Class
key_features = ["CGPA", "SGPA", "GPA", "Absences", "StudyTimeWeekly", "Age"]

for feature in key_features:
    plt.figure()
    sns.histplot(data=df_cleaned, x=feature, kde=True, hue="Target", element="step", stat="density", common_norm=False)
    plt.title(f"Distribution of {feature} by Target Class [Cleaned Data]")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.show()

## 3.4 Boxplots for Academic Features
for feature in ["CGPA", "GPA", "SGPA"]:
    plt.figure()
    sns.boxplot(data=df_cleaned, x="Target", y=feature)
    plt.title(f"{feature} Distribution by Target (Dropout vs Non-Dropout) [Cleaned Data]")
    plt.xlabel("Target")
    plt.ylabel(feature)
    plt.show()

## 3.5 Pairplot (Safe Version)
# If the dataset is large, you need to sample a smaller subset for faster plotting
sample_df = df_cleaned.sample(n=500, random_state=42) if len(df_cleaned) > 500 else df_cleaned

sns.pairplot(sample_df, vars=["CGPA", "GPA", "Absences", "StudyTimeWeekly"], hue="Target", diag_kind="hist")
plt.suptitle("Pairwise Relationships (Sample of 500 Rows) [Cleaned Data]", y=1.02)
plt.show()
