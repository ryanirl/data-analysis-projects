# Car Price Regression

Notebook covering the data gathering, cleaning, EDA, feature engineering, and
prediction of automobile prices. This project was inspired by my first every
python project (https://github.com/ryanirl/CraigslistScraper), though the
actual dataset used in the notebook is a kaggle dataset that can be found here:
https://www.kaggle.com/austinreese/craigslist-carstrucks-data. Big shout out to
Austin Reese for taking the time to compile this large dataset of 400,000+
craigslist ads and make it public. Without hyperparameter tuning, Random Forest 
Regressor (vanilla) was the best performing model and managed to achieve an R2 
Score of 0.931 on the test set. 

**Models Evaluated & Performance:**

<br />

| Model                   | MAE Score | R2 Score |
| ----------------------- | --------- | -------- |
| Random Forest Regressor | 1570.125  | 0.931    |
| XGBoost Regressor       | 2368.176  | 0.902    |
| LightGBM Regressor      | 2629.613  | 0.886    |
| K-Nearest Regressor     | 2973.658  | 0.829    |
| Linear Regression       | 3989.318  | 0.794    |

<br />


**Dependencies:** Pandas, Numpy, Seaborn, Matplotlib, XGBoost, LightGBM, SKLearn, SciPy, and categorical_encoders 

---

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<br />






