# Machine Learning Folder – Decision Tree Activities & Graded Tasks

Is folder mein 4 chhote Machine Learning projects hain. Sab mein **scikit-learn ka `DecisionTreeClassifier`** use hua hai. Har project ka code, dataset (CSV) aur output ka screenshot maujood hai.

## Folder Structure

```
ml folder/
├── Activity-1-Diabetes/        activity1.py + diabetes.csv
├── Activity-2-Salary/          activity2.py + salaries.csv
├── Graded-Task-1-Titanic,/     task1.py + titanic.csv
├── Graded-Task-2-Car/          task2.py + car.csv
├── Output_Screenshots/         Har project ke output ke screenshots
└── README.md
```
(Note: archive mein kuch folders ki duplicate copies bhi thin, wo as-is rakhi gayi hain.)

## Requirements

```
pip install pandas scikit-learn matplotlib
```

Run karne ka tareeqa (script ke apne folder ke andar se, taake CSV mil jaye):
```
cd Activity-1-Diabetes
python activity1.py
```

---

## 1. Activity-1-Diabetes (`activity1.py`)
**Kaam:** Patient ke medical data se predict karna ke diabetes hai ya nahi.
- Dataset: `diabetes.csv` – 20 rows, 8 features (pregnant, glucose, bp, skin, insulin, bmi, pedigree, age), target `label` (0/1).
- Data ko 70% train / 30% test mein split kiya (`random_state=1`).
- Decision Tree train karke test data par predictions aur accuracy nikali.
- Decision Tree ka plot bhi banta hai (`matplotlib`).
- **Output:** Accuracy = **16.67%** (dataset bohot chhota hai, sirf 6 test rows, is liye accuracy kam hai).
- Screenshots: `Output_Screenshots/Activity-1-Diabetes/` (console output + decision tree plot)

## 2. Activity-2-Salary (`activity2.py`)
**Kaam:** Predict karna ke kisi employee ki salary 100k se zyada hogi ya nahi.
- Dataset: `salaries.csv` – 23 rows; columns: company, job, degree, `salary_more_then_100k`.
- Text columns (company, job, degree) ko `LabelEncoder` se numbers mein convert kiya.
- Model poore data par train hua; score nikala.
- Do predictions: Google + Computer Engineer + **Bachelors**, aur Google + Computer Engineer + **Masters**.
- **Output:** Model score = **100%**; dono predictions = **1** (salary > 100k).
- Screenshot: `Output_Screenshots/Activity-2-Salary/`

## 3. Graded-Task-1-Titanic (`task1.py`)
**Kaam:** Titanic ke passengers ka survive karna predict karna.
- Dataset: `titanic.csv` – 20 rows; features: Pclass, Sex, Age, SibSp, Parch, Fare; target `Survived`.
- `Sex` ko `LabelEncoder` se numeric banaya, 70/30 split, Decision Tree train kiya.
- **Output:** Accuracy = **100%** (6 test rows par).
- Screenshot: `Output_Screenshots/Graded-Task-1-Titanic/`

## 4. Graded-Task-2-Car (`task2.py`)
**Kaam:** Car ki condition/class (unacc, acc, good, vgood) predict karna.
- Dataset: `car.csv` – 20 rows; features: buying, maint, doors, persons, lug_boot, safety; target `class`.
- Sab categorical columns aur target ko `LabelEncoder` se encode kiya, 70/30 split, Decision Tree train kiya.
- **Output:** Accuracy = **33.33%** (chhota dataset, 6 test rows).
- Screenshot: `Output_Screenshots/Graded-Task-2-Car/`

---

## Common Workflow (sab projects mein)
1. CSV load karna (`pandas`)
2. Text data ko numbers mein convert karna (`LabelEncoder`)
3. Features (X) aur target (y) alag karna
4. Train/Test split (70/30, `random_state=1`)
5. `DecisionTreeClassifier` train karna
6. Predictions aur accuracy nikalna

## Notes
- Datasets bohot chhote (20–23 rows) hain, is liye accuracy values stable nahi hain – ye sirf learning purpose ke liye hain.
- Screenshots scripts ko run karke bane output se generate kiye gaye hain.
