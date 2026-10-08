import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# Separate features and target
X = data[
    [
        "pregnant",
        "glucose",
        "bp",
        "skin",
        "insulin",
        "bmi",
        "pedigree",
        "age"
    ]
]

y = data["label"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=1
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# Create Decision Tree model
model = DecisionTreeClassifier(random_state=1)


# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)


# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", f"{accuracy * 100:.2f}%")


# Display Decision Tree
plt.figure(figsize=(20, 10))

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Diabetes", "Diabetes"],
    filled=True
)

plt.title("Decision Tree - Diabetes Prediction")
plt.show()