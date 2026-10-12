# Conflict Intensity Prediction

A multi-class machine learning project that predicts the **intensity of violent conflict events** (Low / Medium / High) by combining armed-conflict event data with world trade data for the same country and year.

## Project Overview
- Conflict event records were merged with trade data on **country + year**.
- The target variable `intensity_level` is derived from total deaths per event:
  - **Low**: 10 deaths or fewer
  - **Medium**: 11 to 100 deaths
  - **High**: more than 100 deaths
- Dataset size after cleaning: ~125K events.

## Workflow
1. **Data cleaning**: missing values, duplicates, whitespace in text fields, outlier removal (IQR)
2. **Feature engineering**: civilian/combatant death ratios, deaths per source, `geo_magnitude`, temporal features
3. **Feature selection**: ANOVA F-test, Mutual Information, and RFE with Random Forest
4. **Class imbalance handling**: SMOTE (original split: Low ≈ 89%, Medium ≈ 10%, High ≈ 1%)
5. **Dimensionality reduction and visualization**: PCA and t-SNE
6. **Model training and comparison**: CatBoost, Logistic Regression, and Random Forest (without SMOTE), evaluated with Stratified 5-Fold CV

## Results
Evaluated on the held-out test set (24,931 rows):

| Model | Test Accuracy | Macro F1 | High-class Recall |
|---|---|---|---|
| **CatBoost** | **0.608** | 0.37 | 0.71 |
| Random Forest (No SMOTE) | 0.486 | 0.32 | 0.66 |
| Logistic Regression | 0.473 | 0.28 | 0.54 |

> The task is difficult because of the severe class imbalance and the limited predictive power of the available features, so the focus was on recall for the rare classes (Medium / High) rather than overall accuracy alone.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, CatBoost, Matplotlib, Seaborn

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook conflict_intensity_prediction.ipynb
```

## Data
The data files are not included in this repository. The notebook reads `final_cleaned_data.csv` (the result of merging the conflict events data with the trade data). Place this file in the same folder as the notebook before running it, or rebuild it using the commented-out MySQL cells at the beginning of the notebook.
