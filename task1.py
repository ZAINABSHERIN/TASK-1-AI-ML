import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display the first 5 rows
print("First 5 Rows:")
print(df.head())

# Display information about the dataset
print("\nDataset Information:")
df.info()

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values

# Fill missing values in the Age column using the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing values in the Embarked column using the mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove the Cabin column because it contains many missing values
df.drop("Cabin", axis=1, inplace=True)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# Step 4: Encode Categorical Features
# -----------------------------

# Convert the 'Sex' column into numerical values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert the 'Embarked' column using One-Hot Encoding
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Display the first 5 rows after encoding
print("\nDataset After Encoding:")
print(df.head())

# -----------------------------
# Step 5: Standardize Numerical Features
# -----------------------------
# Select numerical columns to scale
numeric_columns = ["Age", "Fare"]

# Create StandardScaler object
scaler = StandardScaler()

# Apply standardization
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Display dataset after scaling
print("\nDataset After Standardization:")
print(df.head())

# -----------------------------
# Step 6: Visualize Outliers
# -----------------------------
# Boxplot for Age
plt.figure(figsize=(6, 4))
plt.boxplot(df["Age"])
plt.title("Boxplot of Age")
plt.ylabel("Age")
plt.show()

# Boxplot for Fare
plt.figure(figsize=(6, 4))
plt.boxplot(df["Fare"])
plt.title("Boxplot of Fare")
plt.ylabel("Fare")
plt.show()

# -----------------------------
# Step 7: Remove Outliers using IQR
# -----------------------------

# Calculate Q1, Q3, and IQR for the Fare column
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df = df[(df["Fare"] >= Q1 - 1.5 * IQR) & (df["Fare"] <= Q3 + 1.5 * IQR)]

# Display the new shape of the dataset
print("\nDataset Shape After Removing Outliers:")
print(df.shape)

# Display the first 5 rows
print("\nFirst 5 Rows After Removing Outliers:")
print(df.head())

# -----------------------------
# Step 8: Save the Cleaned Dataset
# -----------------------------

# Save the cleaned dataset as a new CSV file
df.to_csv("cleaned_titanic.csv", index=False)

print("\n✅ Cleaned dataset has been saved successfully as 'cleaned_titanic.csv'")