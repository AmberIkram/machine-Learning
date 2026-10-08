# Lab 05 – Naive Bayes Classification

Is lab mein **Naive Bayes** algorithm (scikit-learn) ke do graded tasks hain. Har task ka code aur output ka screenshot maujood hai.

## Folder Structure

```
lab 05/
├── task-1-Wine/          task1.py
├── task-2-Loan/          task2.py + loan_data.csv
├── Output_Screenshots/   Dono tasks ke output ke screenshots
└── README.md
```

## Requirements

```
pip install pandas scikit-learn
```

Run karne ka tareeqa (script ke apne folder ke andar se, taake CSV mil jaye):
```
cd task-2-Loan
python task2.py
```

---

## Task 1 – Wine Dataset (`task-1-Wine/task1.py`)
**Kaam:** Wine ko uski chemical properties se 3 classes (class_0, class_1, class_2) mein classify karna, aur do Naive Bayes models ka muqabla karna.
- Dataset: sklearn ka built-in `load_wine()` – 178 samples, 13 features (alag CSV ki zaroorat nahi).
- 70% train (124) / 30% test (54), `random_state=20`, `stratify=y`.
- **Gaussian Naive Bayes** (`GaussianNB`) aur **Multinomial Naive Bayes** (`MultinomialNB`) dono train kiye.
- Dono ki accuracy, confusion matrix, aur actual vs predicted values print hoti hain.
- **Output:**
  | Model | Accuracy |
  |---|---|
  | Gaussian NB | **96.30%** |
  | Multinomial NB | **72.22%** |
- **Nateeja:** Gaussian NB behtar raha, kyunke Wine ke features continuous (decimal) values hain jin ke liye Gaussian distribution munasib hai; Multinomial NB asal mein counts/frequencies wale data ke liye hota hai.
- Screenshot: `Output_Screenshots/task-1-Wine/`

## Task 2 – Loan Dataset (`task-2-Loan/task2.py`)
**Kaam:** Customer ki profile se predict karna ke usne loan poora wapas kiya (`fully_paid`) ya nahi.
- Dataset: `loan_data.csv` – 20 rows; features: age, income, loan_amount, credit_score, previous_loans, employment_years; target `fully_paid` (1 = paid, 0 = not paid).
- Data exploration: head, shape, columns, info, missing values (koi nahi), statistical summary, aur class distribution (11 paid / 9 not paid).
- 70% train (14) / 30% test (6), `random_state=42`, `stratify=y`.
- `GaussianNB` train kiya, predictions, accuracy aur classification report nikali.
- Un customers ki list nikali jinke baare mein model ne predict kiya ke wo loan poora wapas nahi karenge.
- **Output:** Accuracy = **83.33%**; 2 customers predicted "Not Fully Paid" (index 10 aur 8).
- Screenshots: `Output_Screenshots/task-2-Loan/` (output lamba hone ki wajah se 3 parts mein)

---

## Notes
- Loan dataset bohot chhota (20 rows) hai, test set sirf 6 rows ka hai – is liye accuracy stable nahi, sirf learning purpose ke liye hai.
- Screenshots scripts ko run karke bane asal output se generate kiye gaye hain.
