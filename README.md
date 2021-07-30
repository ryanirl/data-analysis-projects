# Data Analysis Projects

---

## 1. Car EDA & Price Prediction

**NOTE:** UPDATE LINKS

Link to Notebook: https://github.com/ryanirl/data-analysis-projects/blob/main/car_price_prediction/car_price_prediction.ipynb

Notebook covering the Data Gathering, Cleaning, EDA, Feature Engineering, and Model Prediction of Used Car Prices. Data was
gathered here: https://www.kaggle.com/austinreese/craigslist-carstrucks-data and inspired by my first ever Python 
project (https://github.com/ryanirl/CraigslistScraper). Random Forest Regressor (vanilla) was the best performing model and managed to achieve
an R2 Score of 0.931. 


**Models Evaluated & Performance:**

<br />

| Model | R2 Score | MAE Score |
| --- | --- | --- |
| Random Forest Regressor | 1570.125 | 0.931 |
| XGBoost Regressor | 2368.176 | 0.902 |
| LightGBM Regressor | 2629.613 | 0.886 |
| K-Nearest Regressor | 2973.658 | 0.829 |
| Linear Regression | 3989.318 | 0.794 |

<br />


**Dependencies:** Pandas, Numpy, Seaborn, Matplotlib, XGBoost, LightGBM, SKLearn, SciPy, and categorical_encoders 

