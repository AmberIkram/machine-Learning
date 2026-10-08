import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Load Titanic dataset
data = pd.read_csv("titanic.csv")

print("First 5 rows:")
print(data.head())


# Convert Sex into numbers
sex_encoder = LabelEncoder()

data["Sex"] = sex_encoder.fit_transform(data["Sex"])


# Select features
X = data[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]
]


# Target
y = data["Survived"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=1
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# Create Decision Tree
model = DecisionTreeClassifier(random_state=1)


# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredictions:")
print(y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nModel Accuracy Percentage:")
print(f"{accuracy * 100:.2f}%")