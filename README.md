# Heart Disease Prediction

A supervised ML project that predicts heart disease from patient medical and lifestyle records. Built for CPSC 4310 (Machine Learning) at Seattle University by Team 7: Henry Lai, Kimi Yokoyama, Hirna Derege, and Fatimah Abdullahi.

## The idea

Diagnosis has a long history of bias in the US. We wanted to see how far a model could get using only clinical and lifestyle records, so predictions don't depend on one person's judgment. The dataset has no race, ethnicity, or income columns, so we can't actually test for that kind of bias (more on that below). What we could do is build and compare models on the data we have.

## Dataset

`data/smart_healthcare_dataset.csv`: 5,000 synthetic patient records, no missing values.

| Type | Columns |
|---|---|
| Demographic | `age`, `gender` |
| Body and labs | `bmi`, `blood_pressure`, `cholesterol`, `glucose` |
| Lifestyle | `exercise_level` (0 to 2), `smoking`, `alcohol` |
| Symptoms and history | `fatigue`, `chest_pain`, `dizziness`, `diabetes`, `stroke` |
| Target | `heart_disease` (0/1, about 30% positive) |

`health_risk_score` is in the file but is 100 for every row, so it is left out of the models.

## What's in the repo

```
data/        dataset
notebooks/   02 EDA and data quality, 03 baseline models, 04 model comparison
reports/     Part 2 EDA report (PDF export)
src/train.py reproduces the model comparison and a feature ablation
results/     metrics.csv written by train.py
```

## Run it

```bash
pip install -r requirements.txt
python src/train.py          # writes results/metrics.csv
jupyter notebook notebooks/  # or open the notebooks directly
```

## Models

Stratified 80/20 split (`random_state=42`), `class_weight='balanced'`, scaler fit on the training set only. Logistic regression gets the scaled features, the other three use raw features.

Results from `src/train.py` on the 1,000-row test set, all features:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.991 | 0.974 | 0.997 | 0.985 | 0.9996 |
| Random Forest | 0.966 | 0.952 | 0.933 | 0.943 | 0.9961 |
| Decision Tree (depth 5) | 0.957 | 0.888 | 0.980 | 0.932 | 0.9910 |
| SVM (RBF) | 0.943 | 0.852 | 0.980 | 0.912 | 0.9934 |

We picked logistic regression as the final model. It scored best and its coefficients are easy to explain, which matters for anything medical.

## Caveats

These scores are very high, and we don't think they would hold on real patient data.

- **The data is synthetic and very clean.** Age alone gets a ROC-AUC of about 0.99 with logistic regression. In the EDA, heart disease cases are almost all older patients.
- **Some features may be shortcuts.** `diabetes` and `stroke` correlate with the target at 0.86 and 0.84. Dropping `chest_pain`, `diabetes`, and `stroke` takes logistic regression from 0.985 F1 to 0.955 F1 and ROC-AUC from 0.9996 to 0.9983, so the model still does well without them, but age is doing most of the work.
- **Limited fairness check.** Heart disease rates are close across gender (30.4% female, 29.6% male), but there are no race, ethnicity, or socioeconomic columns, so we can't test the bias question this project started with.
- **Notebook text vs. code.** A few numbers in the written analysis of notebook 04 don't match what the code outputs. `src/train.py` and `results/metrics.csv` are the source of truth.

## Next steps

- Cross-validation instead of a single split
- Permutation importance on the logistic regression and random forest
- Try the same pipeline on a real, messier heart disease dataset