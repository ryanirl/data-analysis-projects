# Data Analysis Projects

---

## 1. Car EDA & Price Prediction

Link to Notebook: https://github.com/ryanirl/data-analysis-projects/blob/main/car_price_prediction/car_price_prediction.ipynb

Notebook covering the data gathering, cleaning, EDA, feature engineering, and
model prediction of automobile prices. This project was inspired by my first
every python project (https://github.com/ryanirl/CraigslistScraper), though the
actual dataset used in the notebook is a kaggle dataset found here:
https://www.kaggle.com/austinreese/craigslist-carstrucks-data. Big shout out to
Austin Reese for taking the time to compile this dataset and make it public as
it contains 400,000+ craigslist ads. Random Forest Regressor (vanilla) was the
best performing model and managed to achieve an R2 Score of 0.931. 


**Models Evaluated & Performance:**

| Model                   | MAE Score | R2 Score |
| ----------------------- | --------- | -------- |
| Random Forest Regressor | 1570.125  | 0.931    |
| XGBoost Regressor       | 2368.176  | 0.902    |
| LightGBM Regressor      | 2629.613  | 0.886    |
| K-Nearest Regressor     | 2973.658  | 0.829    |
| Linear Regression       | 3989.318  | 0.794    |

<br />


**Dependencies:** Pandas, Numpy, Seaborn, Matplotlib, XGBoost, LightGBM,
SKLearn, SciPy, and categorical_encoders 

<br />

---

## 2. Cell Instance Segmentation

Link to Repo: https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation


**Models Evaluated & Performance:**

We evaluate the Precision of the IoU at different thresholds in the range 
[0.5, 1) and then took the Mean Average of each Precision to get the MAP 
IoU Score. 

**Transfer Trained on original data w/ NO Augmentations:**

| Model               | AP @ IoU 0.5 | AP @ IoU 0.75 | AP @ IoU 0.9 | MAP IoU @ [0.5, 1) | 
| ------------------- | ------------ | ------------- | ------------ | ------------------ |
| Mask R-CNN R50-FPN  | NONE         | NONE          | NONE         | NONE               | 
| Mask R-CNN R101-FPN | 0.537        | 0.206         | 0.003        | 0.255              | 

<br />

**Transfer Trained on LIVECELL first, then trained on competition specific data:**

| Model               | AP @ IoU 0.5 | AP @ IoU 0.75 | AP @ IoU 0.9 | MAP IoU @ [0.5, 1) | 
| ------------------- | ------------ | ------------- | ------------ | ------------------ |
| Mask R-CNN R50-FPN  | NONE         | NONE          | NONE         | NONE               | 
| Mask R-CNN R101-FPN | NONE         | NONE          | NONE         | NONE               | 



<br />


**Sample Predictions:**

| Image Type     | SH-SY5Y                         | Astrocyte                      | Cort                          | 
| -------------- |:-------------------------------:|:------------------------------:|:-----------------------------:|
| Cell Images    | ![](../img/shsy5y_image_2.png)  | ![](../img/astro_image_1.png)  | ![](../img/cort_image_1.png)  |
| Actual Mask    | ![](../img/shsy5y_actual_2.png) | ![](../img/astro_actual_1.png) | ![](../img/cort_actual_1.png) |
| Predicted Mask | ![](../img/shsy5y_pred_2.png)   | ![](../img/astro_pred_1.png)   | ![](../img/cort_pred_1.png)   |


<br />


**Dependencies:** Pandas, Numpy, Matplotlib, SKLearn, SciPy, detectron2,
PyYAML, cv2, PyTorch, pycocotools, fastcore, joblib, and tqdm.


---


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<br />





