# Lab 05 - Graded Task 1
# Wine Dataset using Naive Bayes

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


# --------------------------------------------------
# 1. Load Wine Dataset
# --------------------------------------------------

wine = load_wine()

X = wine.data
y = wine.target

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(X.shape)

print("\nTarget classes:")
print(wine.target_names)


# --------------------------------------------------
# 2. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=20,
    stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# --------------------------------------------------
# 3. Gaussian Naive Bayes
# --------------------------------------------------

gaussian_model = GaussianNB()

gaussian_model.fit(X_train, y_train)

gaussian_predictions = gaussian_model.predict(X_test)

gaussian_accuracy = accuracy_score(
    y_test,
    gaussian_predictions
)

print("\n--------------------------------")
print("Gaussian Naive Bayes")
print("--------------------------------")

print("Accuracy:",
      f"{gaussian_accuracy * 100:.2f}%")


# --------------------------------------------------
# 4. Gaussian Confusion Matrix
# --------------------------------------------------

gaussian_cm = confusion_matrix(
    y_test,
    gaussian_predictions
)

print("\nGaussian Confusion Matrix:")
print(gaussian_cm)


# --------------------------------------------------
# 5. Multinomial Naive Bayes
# --------------------------------------------------

multinomial_model = MultinomialNB()

multinomial_model.fit(X_train, y_train)

multinomial_predictions = multinomial_model.predict(X_test)

multinomial_accuracy = accuracy_score(
    y_test,
    multinomial_predictions
)

print("\n--------------------------------")
print("Multinomial Naive Bayes")
print("--------------------------------")

print("Accuracy:",
      f"{multinomial_accuracy * 100:.2f}%")


# --------------------------------------------------
# 6. Multinomial Confusion Matrix
# --------------------------------------------------

multinomial_cm = confusion_matrix(
    y_test,
    multinomial_predictions
)

print("\nMultinomial Confusion Matrix:")
print(multinomial_cm)


# --------------------------------------------------
# 7. Compare Both Models
# --------------------------------------------------

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(
    "Gaussian Naive Bayes:",
    f"{gaussian_accuracy * 100:.2f}%"
)

print(
    "Multinomial Naive Bayes:",
    f"{multinomial_accuracy * 100:.2f}%"
)


if gaussian_accuracy > multinomial_accuracy:

    print("\nGaussian Naive Bayes performed better.")

elif multinomial_accuracy > gaussian_accuracy:

    print("\nMultinomial Naive Bayes performed better.")

else:

    print("\nBoth models performed equally.")


# --------------------------------------------------
# 8. Predictions
# --------------------------------------------------

print("\n================================")
print("PREDICTIONS")
print("================================")

print("\nActual values:")
print(y_test)

print("\nGaussian predictions:")
print(gaussian_predictions)

print("\nMultinomial predictions:")
print(multinomial_predictions)