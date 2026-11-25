# Real Estate Housing Price Prediction (Ridge & Lasso Regression)

An end-to-end regression machine learning project predicting median house values (`median_house_value`) on the **California Housing** dataset (`housing.csv`), comparing **Linear Regression (OLS)**, **Ridge Regression (L2)**, and **Lasso Regression (L1)** with hyperparameter tuning via `GridSearchCV`, feature engineering, and visual error analysis.

---

## Project Overview

- **Dataset:** `housing.csv` (**20,640** records and **10** original columns: 9 numerical features and `ocean_proximity` categorical feature).
- **Missing Value Handling:** Imputed **207** missing values in `total_bedrooms` using the column median.
- **Feature Engineering:** Created ratio features (`average_room_for_person` and `rooms_per_household`) to capture housing density and household structure.
- **Preprocessing & Scaling:** Encoded `ocean_proximity` using `OneHotEncoder` inside a `ColumnTransformer`, split data into train (`16,512` rows) and test (`4,128` rows) sets (`80/20` split, `random_state=42`), and standardized features using `StandardScaler`.
- **Models Evaluated:** `LinearRegression` (OLS baseline), `Ridge` (`GridSearchCV` over `alpha` in `np.logspace(-2, 3, 20)`), and `Lasso` (`GridSearchCV` over `alpha` in `np.logspace(-1, 3, 20)` with `5`-fold cross-validation).

---

## Model Evaluation Results (Test Set)

| Model | RMSE | MAE | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (OLS)** | `67,817.17` | `49,079.79` | `0.6490` |
| **Ridge Regression** | `67,777.49` | `49,018.15` | `0.6494` |
| **Lasso Regression** | `67,786.63` | `49,044.87` | `0.6493` |

> **Key Takeaway:** Regularized models (**Ridge** and **Lasso**) slightly outperform unregularized **OLS Linear Regression** across RMSE, MAE, and R², with **Ridge** achieving the lowest test RMSE (`67,777.49`) and highest R² (`0.6494`).

---

## Visualizations

The notebook generates and saves key visualizations during execution:
- `Features Distribution.png` — Histograms of housing numerical attributes.
- `Boxplot.png` — Outlier inspection across features.
- `Correlation Heatmap - Housing Features.png` — Linear correlation matrix among numerical features.
- `Prediction vs actual_values.png` — Scatter comparison of predicted vs. actual house values against the ideal diagonal (`y = x`) for Linear Regression, Ridge, and Lasso.

---

## Tech Stack

- **Language:** Python
- **Data Manipulation & Visualization:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-Learn` (`LinearRegression`, `Ridge`, `Lasso`, `GridSearchCV`, `OneHotEncoder`, `ColumnTransformer`, `StandardScaler`, `train_test_split`, `mean_squared_error`, `mean_absolute_error`, `r2_score`)

---

## Repository Structure

```text
Housing_price_prediction/
├── housing.csv
├── housing_price_prediction.ipynb
├── Features Distribution.png
├── Boxplot.png
├── Correlation Heatmap - Housing Features.png
├── Prediction vs actual_values.png
└── README.md