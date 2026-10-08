# Lab 05 - Graded Task 2
# Loan Dataset using Naive Bayes

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("loan_data.csv")

print("Dataset loaded successfully!")


# --------------------------------------------------
# 2. Data Exploration
# --------------------------------------------------

print("\n================================")
print("DATA EXPLORATION")
print("================================")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nLoan payment distribution:")
print(df["fully_paid"].value_counts())


# --------------------------------------------------
# 3. Separate Features and Target
# --------------------------------------------------

X = df.drop("fully_paid", axis=1)

y = df["fully_paid"]


print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# --------------------------------------------------
# 4. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 5. Train Naive Bayes Classifier
# --------------------------------------------------

model = GaussianNB()

model.fit(X_train, y_train)

print("\nNaive Bayes model trained successfully!")


# --------------------------------------------------
# 6. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


print("\n================================")
print("PREDICTIONS")
print("================================")

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)


# --------------------------------------------------
# 7. Calculate Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("Accuracy:", accuracy)

print("Accuracy Percentage:",
      f"{accuracy * 100:.2f}%")


# --------------------------------------------------
# 8. Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Fully Paid",
            "Fully Paid"
        ]
    )
)


# --------------------------------------------------
# 9. Identify Customers Who Have Not Fully Paid
# --------------------------------------------------

not_fully_paid = X_test[y_pred == 0]

print("\n================================")
print("CUSTOMERS NOT FULLY PAID")
print("================================")

print(not_fully_paid)


# --------------------------------------------------
# 10. Final Message
# --------------------------------------------------

print("\nTotal predicted customers who have not fully paid:",
      len(not_fully_paid))