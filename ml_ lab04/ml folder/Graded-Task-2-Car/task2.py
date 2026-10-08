import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = pd.read_csv("car.csv")

print("First 5 rows:")
print(data.head())


# --------------------------------------------------
# 2. Convert Categorical Data into Numbers
# --------------------------------------------------

encoder = LabelEncoder()

columns = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety"
]

for column in columns:
    data[column] = encoder.fit_transform(data[column])


# Encode target column
data["class"] = encoder.fit_transform(data["class"])


print("\nEncoded Dataset:")
print(data.head())


# --------------------------------------------------
# 3. Define Features and Target
# --------------------------------------------------

X = data[
    [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety"
    ]
]

y = data["class"]


# --------------------------------------------------
# 4. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=1
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 5. Create Decision Tree
# --------------------------------------------------

model = DecisionTreeClassifier(random_state=1)


# --------------------------------------------------
# 6. Train Model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# --------------------------------------------------
# 7. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)


# --------------------------------------------------
# 8. Calculate Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nModel Accuracy Percentage:")
print(f"{accuracy * 100:.2f}%")