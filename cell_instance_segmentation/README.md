# Cell Instance Segmentation

Cell instance segmentation is the segmentation and detection of individual
cells (see image below) in microscopy imaging. This can be particularly useful 
for studying how particular cells may or may not react to various treatments.
Segmenting instances of cells accurately by hand is a tedious and time-consuming 
task.

This project follows from the "Sartorius Cell Instance Segmentation" Kaggle
competition [[6]](#6), and aims to benchmark and analyzes current SOTA instance 
segmentation methods against the task of accurately detecting and segmenting 
individual cells.


![](./img/front_image.png)


<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Summary](#summary)
    * [General Information](#general-information)
    * [Models Performance](#models-performance)
* [Data](#data)
    * [Information](#information)
    * [EDA and Findings](#eda-and-findings)
* [Models](#models)
    * [Information and Analysis](#information-and-analysis)
* [Final Words](#final-words)
* [References](#references)
* [License](#license)

Follows from the Kaggle competition here:
https://www.kaggle.com/c/sartorius-cell-instance-segmentation

**Dependencies:** Pandas, Numpy, Matplotlib, Seaborn, SKLearn, SciPy, Detectron2,
PyYAML, cv2, Cellpose, PyTorch, pycocotools, fastcore, joblib, and tqdm.

---

<!-- Summary -->
## Summary 

<!-- General Information -->
### General Information

The Sartorius data consists of 3 cell types: Astroctyes ("astro"), SH-SY5Y
("shsy5y"), and Cort ("cort"). We must detect individual cell types in an 
image and accurately segment each cell instance.

For this competition trained and am analyzing 2 models: a ResNet50 Mask R-CNN [[7]](#7) and the 
CellPose [[8]](#8) model.


<!-- Models Performance -->
### Models Performance



<!-- Metric -->
#### Evaluation Metric:

We evaluate the precision of the intersection over union (IoU) at different 
thresholds within the range [0.5, 0.95] with a step size of 0.05. Then take 
the mean average of each precision to get the MAP IoU metric score. 

**Note #1:** To understand the low AP scores at an IoU threshold of 0.9 consider 
reading this discussion [here](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/281205).

<!-- Scores -->
#### Scores:

| Model                 | AP @ IoU<br>0.5 | AP @ IoU<br>0.75 | AP @ IoU<br>0.9 | MAP IoU @<br>[0.5, 0.95] | LB Public | 
| --------------------- | --------------- | ---------------- | --------------- | ------------------------ | --------- | 
| Mask R-CNN R50-FPN    | 0.5644          | 0.2650           | 0.0125          | 0.2893                   | 0.306     | 
| CellPose w/ SizeModel | 0.6187          | 0.2491           | 0.0103          | 0.2975                   | 0.312     | 


| Model                 | Cort   | SH-SY5Y | Astrocyte | 
| --------------------- | ------ | ------- | --------- |
| Mask R-CNN R50-FPN    | 0.3869 | 0.1879  | 0.1914    |
| CellPose w/ SizeModel | 0.3924 | 0.2274  | 0.1865    |


**Note #2:** Though in my analysis I only show the IoU Score at an AP of [0.5, 0.75, 0.9], the public LB 
is calculated through MAP's of [0.5, 0.55, ..., 0.9, 0.95], I just didn't feel the need to list each one in 
my summary. The detailed tables of my model performance can be found 
[here](https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation/MODEL_PERFORMANCE.md).


#### Sample Predictions:

![](./img/astro_analysis_annotated.png)



---

<!-- Data -->
## Data:

<!-- Data Information -->
### Information 

Data in this competition was given in the form of Testing, Training, Train Semi
Supervised, and LIVECELL Data. The data can be downloaded from the Kaggle website
linked at the top of this repo.

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

The extra data in the LIVECELL dataset will almost surely be utilized during training to 
acheive higher scores, some ideas include:
1. Combining the addition data from the SH-SY5Y cell line in the LIVECELL dataset.
2. First training on the larger LIVECELL dataset then transer learning that model to 
the original training set. 

<!-- Data EDA -->
### EDA and Findings: 

*WORK IN PROGRESS*

Expand each key finding for a detailed analysis of each. 

<details>
   <summary>Cell Size and Count:</summary>

<br />

| Cell Size    | Cort                                | SH-SY5Y                               | Astocytes                            |
| ------------ | ----------------------------------- | ------------------------------------- | ------------------------------------ |
| Distribution | ![](./img/size_count/cort_size.png) | ![](./img/size_count/shsy5y_size.png) | ![](./img/size_count/astro_size.png) |
| Count        | 10777.00                            | 52286.00                              | 10522.00                             |
| Mean         | 240.16                              | 224.50                                | 905.81                               |
| STD          | 139.17                              | 133.94                                | 855.19                               |
| Min          | 33.00                               | 30.00                                 | 37.00                                |
| Max          | 2054.00                             | 2254.00                               | 13327.00                             |


| Cell Count   | Cort                                 | SH-SY5Y                                | Astocytes                             |
| ------------ | ------------------------------------ | -------------------------------------- | ------------------------------------- |
| Distribution | ![](./img/size_count/cort_count.png) | ![](./img/size_count/shsy5y_count.png) | ![](./img/size_count/astro_count.png) |
| Count        | 320.00                               | 155.00                                 | 131.00                                |
| Mean         | 33.68                                | 337.33                                 | 80.32                                 |
| STD          | 16.50                                | 149.60                                 | 64.13                                 |
| Min          | 4.00                                 | 49.00                                  | 5.00                                  |
| Max          | 108.00                               | 790.00                                 | 594.00                                |



---

</details>


<details>
   <summary>There is an Uneven Distribution of Cell Types:</summary>

<br />

<p align="center">
    <img src="./img/cell_type_distribution.png" width="50%">
</p>

In the training set there are 320 Cort (~52.81%), 155 SH-SY5Y (~25.58%), and 131 (~21.62%) Astro cell images. 
My model performance on each cell type can be seen here: 


| Model                 | Cort   | SH-SY5Y | Astrocyte | MAP IoU @<br>[0.5, 0.95] | LB Public |
| --------------------- | ------ | ------- | --------- | ------------------------ | --------- |
| Mask R-CNN R50-FPN    | 0.3869 | 0.1879  | 0.1914    | 0.2893                   | 0.306     |
| CellPose w/ SizeModel | 0.3924 | 0.2274  | 0.1865    | 0.2975                   | 0.312     |

Both models performed much better on the Cort cell line than the SH-SY5Y and Astro cell line. Also, both my
models had a positive LB correlation (roughly about +0.015) leading me to believe there *might* (pure specalation
here) be a larger distribution of the Cort cell type in the private testing data than our training data. As seen
by comments in this post [[5]](#5) many people were also experiencing strong positive LB correlation (some people 
were even getting upwards of 0.03 gains). 


---

</details>


<details>
   <summary>Annotations are NOT Pixel Perfect:</summary>

<br />

Although mask prediction may be largely limited by
annotation quality. A few of the Astrocyte annotations are not pixel perfect and
some I would even consider potentially damaging to a models perforance. The
main recuring problem I saw with astrocyte masks was that some were hollow. 
Though in my non-professional opinion there were also a couple images that seemed
to be missing signifacant annotations (see ID: 3bcc8ba1dc17). As an example
of an image with hollow artifacts:

<p align="center">
    <img src="./img/annotation_not_pp_examples/hollow_artifact.png" width="65%">
</p>

This lead some people to try and *clean* these astro masks [[4]](#4). Though
one problem discussed is that if these problems lie in the training set then they also
probably lie in the competition testing set. That said, I never tried training
with a *cleaned* set but I do wonder what kind of perforance gains one might see 
if they spent a day meticulously going through and re-annotated the Astro masks
by hand as well as they could. 

Another noteable problem with the masks not being pixel perfect is how strict
the MAP IoU metric is at a threshold of above ~0.85. Though I will refer you 
to this Kaggle discussion that describes this problem very nicely:
https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/281205 [[3]](#3)

Some image ids with hollow artifacts or *potential* missing masks:
- 3bcc8ba1dc17
- 174793807517
- 13325f865bb0
- 182c3da676bd


---

</details>


---


<!-- Models -->
## Models

<!-- Models Information -->
### Information and Analysis

<details>
   <summary>CellPose</summary>

<br />

*TODO*

---

</details>



<details>
   <summary>Mask R-CNN R50 FPN</summary>

<br />


Each trained multiple models, here are details about my highest performing model. Each model was first trained
on the LIVECELL Dataset then transfered to the smaller Sartorius Dataset. This gave me a 2% improvement overall
from models that weren't first trained on LIVECELL.

<details>
   <summary><b>Training Details</b></summary>

<br />

- Epochs: 100 -> 50 (100 on LIVECELL -> 50 on Sartorius)
- Batch Size: 2
- LR: 0.0005
- Resize Max Size: 1333 (default)
- Resize Min Size: (640, 672, 704, 736, 768, 800) (default)
- Anchor Generator Size: [[32], [64], [128], [256], [512]] (default)
- N Classes: 8 -> 3 
- Detections per Image: 1000

</details>


<details>
   <summary><b>Inference & Tuning Details</b></summary>

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

</details>


*See here for code:*
- [Training](https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation/src/d2_train.py)
- [Inference](https://github.com/ryanirl/data-analysis-projects/blob/main/cell_instance_segmentation/src/detectron2_src/d2_config.yaml)


#### Mask R-CNN Analysis and Findings:

<details>
   <summary>Accurate BBox Proposals are KEY:</summary>

<br />

According to takuoko and tascj, the team of 2 who placed 1st: "We decided to 
build a solution using box-based instance segmentation, and focus more on the
bbox detection part. We think the mask prediction performance is mainly limited
by annotation quality so we did not pay much attention to it." [[1]](#1). For 
the task of Cell Instance Segmentation, I believe this is a key insight. When 
predicting a small amount of low density large objects, such as a person or cat 
in the center of the frame, I belive it's the mask prediction that can often lack
behind often not having pixel perfect borders. But, given the small and high density 
nature of these cell populations, a single vanilla ResNet50 based Mask R-CNN severely 
lacks in its ability to generate accurate BBox's do to its exhaustive Anchor Generating 
nature. For BBox proposals, the top 2 winning solutions [[1]](#1) [[2]](#2) both used 
multiple, non-exhaustive BBox Heads (such as YOLOX) followed by a Weighted Box Fusion 
(WBF) ensemble. 

---

</details>


---

</details>

---


<!-- Final -->
## Final Words:

For more information on the **winners** and additional key **takeaways** from this
competition check out this post:

- https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/299036


---

<!-- References -->
## References:

<a id = "1">[1]</a>: takuoko (Jan. 2022). "1st place solution". *Kaggle*. https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/298869

<a id = "2">[2]</a>: nvnn (Dec. 2021). "2nd place solution. *Kaggle*. "https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/297988

<a id = "3">[3]</a>: Theo Viel (Nov. 2021). "Annotations Are too Noisy for the Metric". *Kaggle*. https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/281205

<a id = "4">[4]</a>: hengck23 (Nov. 2021). "The clean astro mask are here!!!". *Kaggle*. https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/291371

<a id = "5">[5]</a>: Sirius (Nov. 2021). "Best Single Model". *Kaggle*. https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/289033

<a id = "6">[6]</a>: Sartorius - Cell Instance Segmentation (Sep. 2021). *Kaggle*. https://www.kaggle.com/c/sartorius-cell-instance-segmentation

<a id = "7">[7]</a>: He, Kaiming, et al. (2017). "Mask r-cnn." Proceedings of the IEEE international conference on computer vision. 

<a id = "8">[8]</a>: Stringer, C., Wang, T., Michaelos, M. et al. (2021). "Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100-106. https://doi.org/10.1038/s41592-020-01018-x

---

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<br />







