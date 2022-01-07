# Cell Instance Segmentation

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Data](#data)
* [Performance](#evaluation-metric)
* [Interesting Findings](#interesting-findings)
* [License](#license)

Follows from the Kaggle competition here:
https://www.kaggle.com/c/sartorius-cell-instance-segmentation

**Dependencies:** Pandas, Numpy, Matplotlib, SKLearn, SciPy, detectron2,
PyYAML, cv2, CellPose, PyTorch, pycocotools, fastcore, joblib, and tqdm.

---

<!-- DATA -->
## Data:

Data in this competition was given in the form of Testing, Training, Train Semi
Supervised and LIVECELL Data. 

The Training, Testing, and Train Semi Supervised data consist of 3 cell types:
1. Astroctyes ("astro")
2. SH-SY5Y ("shsy5y")
3. Cort ("cort")

Though in this competition we're only tested on the 3 cell types listed above, the predecessor
to competition dataset, the LIVECELL dataset, is also given and contains 8 Cell Types. Of the 8 
extra cell lines, the only one that overlaps with the training data is the SH-SY5Y cell line. 
The SH-SY5Y data in the LIVECELL Dataset IS seperate from the data we are given in the competition. 
The other 7 Cell Types in the LIVECELL Data are:
1. A172
2. BT474
3. BV-2
4. Huh7
5. MCF7
6. SkBr3
7. SK-OV-3

The extra data in the LIVECELL dataset will almost surely be utilized in training to 
acheive high scores, some ideas include:
1. Combining the addition data from the SH-SY5Y cell line in the LIVECELL dataset.
2. First Training on the larger LIVECELL dataset then Transer Learning that model to 
the original training set. 


---

<!-- Performance -->
## Models Evaluated & Performance

In depth analysis of Model Performance can be found here:

https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation/MODEL_PERFOMANCE.md

### Training Technique:

**Mask R-CNN R50 FPN:**

Each trained multiple models, here are details about my highest performing model. Each model was first trained
on the LIVECELL Dataset then transfered to the smaller Sartorius Dataset. This gave me a 2% improvement overall
from models that weren't first trained on LIVECELL.

**Training Details:**
- Epochs: 100 -> 50 (100 on LIVECELL -> 50 on Sartorius)
- Batch Size: 2
- LR: 0.0005
- Resize Max Size: 1333 (default)
- Resize Min Size: (640, 672, 704, 736, 768, 800) (default)
- Anchor Generator Size: [[32], [64], [128], [256], [512]] (default)
- N Classes: 8 -> 3 
- Detections per Image: 1000


**Inference Details:**

- Custom NMS: 
    - Astrocyte: 0.4
    - SH-SY5Y: 0.25
    - Cort: 0.7

- Custom Score Thresholding: 
    - Astrocyte: 0.4 
    - SH-SY5Y: 0.15
    - Cort: 0.55

- Custom Per Pixel Score Thresholding: 
    - Astrocyte: 0.45
    - SH-SY5Y: 0.5
    - Cort: 0.45


**See here for more details:**

Training: https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation/src/d2_train.py

Inferance: https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation/src/detectron2_src/d2_config.yaml



### Evaluation Metric:

We evaluate the Precision of the IoU at different thresholds in the range 
[0.5, 0.95] with a step size of 0.05, and then took the Mean Average of 
each Precision to get the MAP IoU Score. 

NOTES: To understand the low AP scores at an IoU threshold of 0.9 consider 
reading this discussion:
https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/281205

**Scores:**

| Model                 | AP @ IoU<br>0.5 | AP @ IoU<br>0.75 | AP @ IoU<br>0.9 | MAP IoU @<br>[0.5, 0.95] | LB Public | 
| --------------------- | --------------- | ---------------- | --------------- | ------------------------ | --------- | 
| Mask R-CNN R50-FPN    | 0.5644          | 0.2650           | 0.0125          | 0.2893                   | 0.306     | 
| CellPose w/ SizeModel | 0.6187          | 0.2491           | 0.0103          | 0.2975                   | 0.312     | 


**Cell Specific Scores (CV):**

| Model                 | Cort   | SH-SY5Y | Astrocyte | 
| --------------------- | ------ | ------- | --------- |
| Mask R-CNN R50-FPN    | 0.3869 | 0.1879  | 0.1914    | 
| CellPose w/ SizeModel | 0.3924 | 0.2274  | 0.1865    |


**Side Note:**

Though in my analysis I only show the IoU Score at an AP of [0.5, 0.75, 0.9], the public LB 
is calculated through MAP's of [0.5, 0.55, ..., 0.9, 0.95], I just didn't feel the need to 
list each one in my analysis.


**Sample Predictions:**

![](./img/astro_analysis_annotated.png)

---

<!-- Interesting Findings -->
## Interesting Findings

- A lot of the Astrocyte annotations are NOT pixel perfect and some I would even consider broken. 
- Both the Mask R-CNN and the Cellpose Model (which is based off of a UNet) had positive LB correlation. 
Probably due to a larger distribution of the Cort cell line in the public LB dataset.
- On my local CV, Cellpose performs slighly better on the SH-SY5Y and Cort cell line. Though on the private
LB the Mask R-CNN Model performed better than my Cellpose model by 0.006. 
    - When I dug into this it seems like Cellpose is better than my minimally optimized Mask R-CNN at classifying 
    cells that are close together, but worse at per-pixel predictions. My guess would be that in the private 
    LB, the Cort samples were more spread out and not as bunched together as my validation set. 

- The Mask R-CNN performs better on the Astrocyte cell line. From my inferences Cellpose Astro predictions were
larger, irregular, and jittery shaped leading to FP's where the Mask R-CNN was able to classify smoother cirular shapes 
better but had more FN's. This difference probably has something to do with the fact that Cellpose is predicting 
gradient flows and not the mask directly.
- The Mask R-CNN on the SH-SY5Y cell line had significantly more FN's than Cellpose. From my experience this is because 
of how cells are in each image and how bunched together the they are. 


<!-- Improving -->
## Ideas for Improvment:

**WORK IN PROGESS**

- Better / Smarter BBox Proposal
- Larger Resize of Images
- U-Net replacing Mask Head


---

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<br />







