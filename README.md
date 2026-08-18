# Hotel Booking Cancellation Prediction

Predicting whether a hotel booking will be canceled, using the [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset (119,390 bookings, 32 raw features).

## Pipeline

1. **Data Cleaning** — missing values, duplicates, invalid values (negative ADR, zero-guest bookings), outlier review (kept, not removed — see notebook for reasoning)
2. **EDA** — 24+ charts covering the target distribution, cancellation drivers, and feature relationships
3. **Feature Engineering** — agent/company presence flags, arrival date decomposition, room mismatch flag, total nights, leakage columns (`reservation_status`, `reservation_status_date`) dropped
4. **Encoding** — one-hot encoding for categorical columns, frequency encoding for `country`
5. **Feature Selection** — low-variance filtering + Random Forest importance ranking (95% cumulative importance → 27 features)
6. **Modeling** — 7 models, Random Forest hyperparameter-tuned via `RandomizedSearchCV`

## Key Visualizations

### Target Distribution
The dataset is imbalanced: ~63% not canceled vs ~37% canceled — why F1/ROC-AUC are used to rank models instead of raw accuracy.

<img width="558" height="470" alt="image" src="https://github.com/user-attachments/assets/c10ed881-69d4-4264-ace9-b228c8442ed7" />

### Correlation Heatmap
`deposit_type_Non Refund`, `lead_time`, and `room_mismatch` show the strongest relationships with cancellation.

<img width="2021" height="1790" alt="image" src="https://github.com/user-attachments/assets/324a269c-5954-4546-bfee-16761a167fd3" />

### Feature Importance (Random Forest)
`lead_time`, `country_encoded`, and `total_of_special_requests` are the top predictors.

<img width="962" height="701" alt="image" src="https://github.com/user-attachments/assets/02ba1a4f-862c-46dc-92f0-00d3bf38b773" />

## Model Results

| Model | Train Acc | Test Acc | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | 85.7% | 84.6% | 0.747 | 0.665 | **0.704** | 0.908 |
| **Random Forest** (tuned) | 91.7% | 84.6% | 0.757 | 0.650 | 0.699 | 0.910 |
| Gradient Boosting | 83.6% | 83.7% | 0.735 | 0.639 | 0.684 | 0.898 |
| Decision Tree | 83.6% | 82.5% | 0.699 | 0.640 | 0.668 | 0.880 |
| KNN | 83.9% | 81.9% | 0.701 | 0.599 | 0.646 | 0.875 |
| Logistic Regression | 78.4% | 79.1% | 0.675 | 0.462 | 0.549 | 0.838 |
| Linear Regression* | 77.7% | 78.3% | 0.713 | 0.354 | 0.473 | 0.827 |

*Not a classification algorithm — included as a baseline, output thresholded at 0.5.*

**Best model: XGBoost** (highest F1 Score), with tuned Random Forest close behind on ROC-AUC.

Random Forest was tuned via `RandomizedSearchCV` (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`; 6 candidates × 2-fold CV, optimizing F1). It improved F1 from 0.679 → 0.699 over the untuned baseline, but its train/test accuracy gap widened to 7.1 percentage points — a mild overfitting signal worth keeping in mind next to XGBoost's 1.1-point gap.

### Model Comparison
<img width="1391" height="590" alt="image" src="https://github.com/user-attachments/assets/7a7b94e8-7e7e-4778-9c8a-6575e6d32298" />

### Train vs Test Accuracy (Overfitting Check)
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/256ff3a5-295e-4e72-b636-8520deb56d78" />

### Confusion Matrices
=<img width="1670" height="790" alt="image" src="https://github.com/user-attachments/assets/e9480857-2798-45b8-840f-d5a54c51e712" />

## Repo Structure

```
├── SFE_Project.ipynb      # full notebook: cleaning, EDA, feature engineering, modeling
├── hotel_bookings.csv      # dataset
├── images/                 # exported charts used in this README
└── README.md
```

## Running It

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
jupyter notebook SFE_Project.ipynb
```
