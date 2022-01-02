# Cell Instance Segmentation

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Goals](#goals)
* [Data](#data)
* [Performance](#evaluation-metric)
* [License](#license)

Follows from the Kaggle competition here:
https://www.kaggle.com/c/sartorius-cell-instance-segmentation

**Dependencies:** Pandas, Numpy, Matplotlib, SKLearn, SciPy, detectron2,
PyYAML, cv2, CellPose, PyTorch, pycocotools, fastcore, joblib, and tqdm.


<!-- Goals -->
## Goals:

I've always found kaggle fascinating but never tried to compete in a competition. 
And so, as an introduction to the Kaggle community I decided to join a competition 
and learn by doing. My goals for this competition are as follows: 

Goals:
1. Build a baseline SOTA model with miminal parameter optimization through prebuild 
libraries, in this case those libaries consist of Detectron2 and Cellpose. 
2. Explore data and make model optimizing decisions based on findings. 
3. Analize the strengths and weeknesses of current SOTA models / methods.
4. Analyze the top couple winning or any interesting submissions to get an baseline 
understanding of what I must do differently to be a strong competitor in a Kaggle 
competition.

---

<!-- DATA -->
## Data:

TODO:
 - List data format and keep it simple
 - Post some findings and how I transfered them to model optimizations
 - Examples of image, mask, and prediction.


Data in this competition was given in the form of Testing, Training, Train Semi
Supervised and LIVECELL Data. 

The Training, Testing, and Train Semi Supervised data consist of 3 cell types:
1. Astroctyes (astro)
2. SH-SY5Y (shsy5y)
3. Cort (cort)

Though the competition is only based on the 3 cell types listed above, the predecessor
to this dataset is the LIVECELL dataset which containing 8 Cell Types, including the 
SH-SY5Y cell line but not the Astro or Cort cell line. The SH-SY5Y data in the LIVECELL 
Data IS seperate from the data we are given in the competition. The other 7 Cell Types 
in the LIVECELL Data are:
1. A172
2. BT474
3. BV-2
4. Huh7
5. MCF7
6. SkBr3
7. SK-OV-3


The extra data in the LIVECELL dataset will almost surely be utilized in training to 
acheive high scores on the Testing set, some ideas include:
1. Combining the addition data from the SH-SY5Y cell line to the data in the Training set.
2. First Training on the larger LIVECELL dataset then Transer Learning that model to the
original training set. 


---

<!-- Performance -->
## Models Evaluated & Performance

In depth analysis of Model Performance and Postprocessing techniques can be 
found here:

*ADD LINK*

### Evaluation Metric:

We evaluate the Precision of the IoU at different thresholds in the range 
[0.5, 0.95] with a step size of 0.05, and then took the Mean Average of 
each Precision to get the MAP IoU Score. 

NOTES: To understand the low AP scores at an IoU threshold of 0.9 consider 
reading this discussion:
https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/281205

**Scores:**

| Model                 | AP @ IoU 0.5 | AP @ IoU 0.75 | AP @ IoU 0.9 | MAP IoU @ [0.5, 1] | LB Public | 
| --------------------- | ------------ | ------------- | ------------ | ------------------ | --------- | 
| Mask R-CNN R50-FPN    | 0.5644       | 0.2650        | 0.0125       | 0.2893             | 0.306     | 
| CellPose w/ SizeModel | 0.6187       | 0.2491        | 0.0103       | 0.2975             | 0.312     | 


**Cell Specific Scores (CV):**

| Model                 | Cort   | SH-SY5Y | Astrocyte | 
| --------------------- | ------ | ------- | --------- |
| Mask R-CNN R50-FPN    | 0.3869 | 0.1879  | 0.1914    | 
| CellPose w/ SizeModel | 0.3924 | 0.2274  | 0.1865    |


**Interesting Findings:** 
- Both the Mask R-CNN and the Cellpose Model (which is based off of a UNet) had positive LB correlation.
- On my local CV Cellpose performs slighly better on the SH-SY5Y and Cort cell line.
    - When I dug into this it turned out Cellpose is better than Mask R-CNN at classifying cells that are
    close together. This may be do to the larger Archor Size I set on the Mask R-CNN. 

- Mask R-CNN performs better on the Astrocyte cell line. This was do to Cellpose Astro predictions being more 
irregular and jittery shaped where the Mask R-CNN being able to classify cirular shapes better (more on this 
here: *add link*)


**Side Note:**

Though in my analysis I only show the IoU Score at an AP of [0.5, 0.75, 0.9], the public LB 
is calculated through MAP's of [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], I just
didn't feel the need to list each one in my analysis.


---

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<br />







