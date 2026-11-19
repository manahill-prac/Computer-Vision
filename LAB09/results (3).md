# Comparative Implementation of Machine Learning Regression Algorithms

A comparative study of six regression algorithms using the **California Housing Dataset**. All models are trained and evaluated using the same train-test split to ensure a fair comparison.

## Assignment Overview

- **Dataset:** California Housing
- **Samples:** 20,640
- **Features:** 8
- **Target:** `MedHouseVal` (median house value, in units of $100,000)
- **Train-test split:** 80:20 (random state = 42)
- **Evaluation metrics:** MAE, MSE, RMSE, and R²

## Algorithms Implemented

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Neural Network Regression (MLP)
5. XGBoost Regression
6. K-Nearest Neighbors (KNN) Regression

Scaling is applied to models that require or benefit from it (Linear Regression, MLP, and KNN) using pipelines. The scaler is fitted on training data only to prevent data leakage. No hyperparameter tuning is performed.

## Results

Performance on the held-out test set (sorted by RMSE; lower is better):

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| XGBoost | 0.2993 | 0.2059 | **0.4538** | **0.8429** |
| Random Forest | 0.3268 | 0.2540 | 0.5040 | 0.8062 |
| Neural Network (MLP) | 0.3484 | 0.2680 | 0.5177 | 0.7955 |
| Decision Tree | 0.4314 | 0.4076 | 0.6385 | 0.6889 |
| KNN | 0.4462 | 0.4324 | 0.6576 | 0.6700 |
| Linear Regression | 0.5332 | 0.5559 | 0.7456 | 0.5758 |

*The target is measured in units of $100,000. For example, an MAE of 0.2993 is approximately $29,930.*

### Best-performing model

**XGBoost** achieved the best test performance, with the lowest MAE and RMSE and the highest R² among the six models.

## Notebook Contents

- Dataset loading and exploratory data analysis
- Data preprocessing and train-test split
- Implementation and training of six regression models
- Metric calculation and verification
- Actual vs. predicted plots
- Model performance comparison charts
- Residual analysis
- Tree-based feature importance
- Comparative discussion and conclusion generated from the experiment

## How to Run

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. Open `Regression_Comparison.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the notebook cells from top to bottom.

The California Housing dataset is fetched through scikit-learn and may require an internet connection on the first run. The notebook checks for XGBoost and installs it if it is missing.

## Repository File

- `Regression_Comparison.ipynb` — Complete notebook containing code, outputs, visualizations, evaluation, discussion, and conclusion.

---

**Course Assignment:** Comparative implementation and evaluation of regression algorithms.
