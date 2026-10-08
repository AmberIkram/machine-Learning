import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = pd.read_csv("salaries.csv")

print("First 5 rows:")
print(data.head())


# --------------------------------------------------
# 2. Convert Text Data into Numbers
# --------------------------------------------------

label_encoder = LabelEncoder()

data["company_n"] = label_encoder.fit_transform(data["company"])
data["job_n"] = label_encoder.fit_transform(data["job"])
data["degree_n"] = label_encoder.fit_transform(data["degree"])


print("\nDataset after encoding:")
print(data)


# --------------------------------------------------
# 3. Define Features and Target
# --------------------------------------------------

X = data[["company_n", "job_n", "degree_n"]]

y = data["salary_more_then_100k"]


# --------------------------------------------------
# 4. Create Decision Tree Classifier
# --------------------------------------------------

model = DecisionTreeClassifier(random_state=1)


# --------------------------------------------------
# 5. Train the Model
# --------------------------------------------------

model.fit(X, y)

print("\nModel trained successfully!")


# --------------------------------------------------
# 6. Calculate Model Score
# --------------------------------------------------

score = model.score(X, y)

print("\nModel Score:")
print(score)

print("\nModel Score Percentage:")
print(f"{score * 100:.2f}%")


# --------------------------------------------------
# 7. Make Predictions
# --------------------------------------------------

# Google + Computer Engineer + Bachelors
prediction1 = model.predict(
    [[
        label_encoder.fit_transform(data["company"])[
            list(data["company"]).index("Google")
        ],
        label_encoder.fit_transform(data["job"])[
            list(data["job"]).index("Computer Engineer")
        ],
        label_encoder.fit_transform(data["degree"])[
            list(data["degree"]).index("Bachelors")
        ]
    ]]
)

print("\nPrediction 1:")
print(prediction1)


# Google + Computer Engineer + Masters
prediction2 = model.predict(
    [[
        label_encoder.fit_transform(data["company"])[
            list(data["company"]).index("Google")
        ],
        label_encoder.fit_transform(data["job"])[
            list(data["job"]).index("Computer Engineer")
        ],
        label_encoder.fit_transform(data["degree"])[
            list(data["degree"]).index("Masters")
        ]
    ]]
)

print("\nPrediction 2:")
print(prediction2)