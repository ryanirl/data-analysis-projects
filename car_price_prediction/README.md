# Car Price Regression

Covers the data gathering, cleaning, EDA, feature engineering, and model prediction of automobile prices. This project was 
inspired by my first every python project (https://github.com/ryanirl/CraigslistScraper), though the actual dataset used in the notebook is 
a kaggle dataset found here: https://www.kaggle.com/austinreese/craigslist-carstrucks-data. Big shout out to Austin Reese for taking the
time to compile this dataset and make it public as it contains 400,000+ datapoints. Random Forest Regressor (vanilla) was the best performing 
model and managed to achieve an R2 Score of 0.931. 


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


