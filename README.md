# Data Analysis Projects

---

## 1. Car EDA & Price Prediction

Link to Notebook: https://github.com/ryanirl/data-analysis-projects/blob/main/car_price_prediction/car_price_prediction.ipynb

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


---

## 2. Cell Instance Segmentation

**WORK IN PROGRESS**

Link to Repo: https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation

Cell instance segmentation is the segmentation and detection of individual
cells (see image below) in microscopy imaging. This can be particularly useful 
for studying how particular cells may or may not react to various treatments.
Segmenting instances of cells accurately by hand is a tedious and time-consuming 
task. This project follows from the "Sartorius Cell Instance Segmentation" Kaggle
competition [[1]](#1), and aims to benchmark and analyzes two current thought to be 
SOTA instance segmentation methods against the task of accurately detecting and 
segmenting individual cells.

**Evaluation Metric**: The metric used in this competition is the MAP IoU 
score. This is calculated by evaluating the precision of the intersection over 
union (IoU) at different thresholds within the range [0.5, 0.95] at a step 
size of 0.05. Then the mean average of each precision is found to get the MAP 
IoU metric score. 

**Model Scores (LB):**

| Model                 | AP @ IoU<br>0.5 | AP @ IoU<br>0.75 | AP @ IoU<br>0.9 | MAP IoU @<br>[0.5, 0.95] | LB Public | 
| --------------------- | --------------- | ---------------- | --------------- | ------------------------ | --------- | 
| Mask R-CNN R50-FPN    | 0.5644          | 0.2650           | 0.0125          | 0.2893                   | 0.306     | 
| CellPose w/ SizeModel | 0.6187          | 0.2491           | 0.0103          | 0.2975                   | 0.312     | 


**Cell Specific Scores (CV):**

| Model                 | Cort   | SH-SY5Y | Astrocyte | 
| --------------------- | ------ | ------- | --------- |
| Mask R-CNN R50-FPN    | 0.3869 | 0.1879  | 0.1914    | 
| CellPose w/ SizeModel | 0.3924 | 0.2274  | 0.1865    |


<br />


**Dependencies:** Pandas, Numpy, Matplotlib, SKLearn, SciPy, detectron2,
PyYAML, cv2, PyTorch, pycocotools, fastcore, joblib, and tqdm.

---

## 3. Dimensionality Reduction on Genotypes

<p align="center">
    <img src="./dimensionality_reduction_on_genotypes/img/chr_21_visualization_1.svg" width="100%">
</p>

The goal of this project is to use common dimensionality reduction techniques
such as PCA, t-SNE, and UMAP to infer ancestery from an individuals genotype.
Given that the genotype of any two individuals (human) is roughly 99.9% alike, 
we would like to pick out what is different (the variants) and do dimensionality
reduction on those parts of the genome. Today, we identify these variants
through a process called variant calling. In this project, the data is from the
1000 Genomes Project [[2]](#2) and is given in the form of a VCF (Variant Call
Format) file containing variants at various loci on an individual's
chromosome. Such variants include single nucleotide polymorphisms (SNPs) or
structural variants (SVs) such as insertion/deletions (INDEL's) and more. 

<br />

**Dependencies:** Altair, Pysam, UMAP (umap-learn), Pandas, Numpy, 
SKLearn, SciPy, and tqdm.

---

<!-- References -->
## References:

<a id = "1">[1]</a>: Sartorius - Cell Instance Segmentation (Sep. 2021). *Kaggle*. https://www.kaggle.com/c/sartorius-cell-instance-segmentation

<a id = "2">[2]</a>: 1000 Genomes Project Consortium, Auton, A., Brooks, L. D., Durbin, R. M., Garrison, E. P., Kang, H. M., Korbel, J. O., Marchini, J. L., McCarthy, S., McVean, G. A., & Abecasis, G. R. (2015). A global reference for human genetic variation. Nature, 526(7571), 6874. https://doi.org/10.1038/nature15393


---


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<br />




