import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os

from detectron2 import model_zoo

from detectron2_src.d2_pipelines import *
from utils import rle_decode, load_yaml
from monkey_patches import *
from metrics import iou_map
from visualization import *

from cellpose import models, io, plot, utils
models.SizeModel.eval = eval

# Locations & Information
cellpose_model_dir = "../models/cellpose_models/cellpose_residual_on_style_on_concatenation_off_data_2021_12_07_12_54_10.758174"
size_model_dir     = "../models/cellpose_models/cellpose_residual_on_style_on_concatenation_off_data_2021_12_07_12_54_10.758174_size.npy" 

detectron2_models_dir = [
    "../models/detectron2_models/sartorius_main_transfer/model_final.pth",
#    "../models/detectron2_models/transfer_0.pth",
#    "../models/detectron2_models/model_0.pth",
#    "../models/detectron2_models/model_1.pth",
#    "../models/detectron2_models/model_2.pth",
#    "../models/detectron2_models/model_3.pth"
]

data_dir = "../data/"

actual_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

# ASTRO 25% | CORT 50% | SH-SY5Y 25%
#image_ids = [
# "2cab2cb161a4", "2d8810967b36", "2dc940ff1a71", "2f1b9aea78d7",
# "315b21b955c6", "31e4fa0a83f4", "34b6c5235ab4", "34e41956f993",
# "35fc12883459", "3625dabdf452", "36855e37531a", "3912a0bede5b",
# "3b3991c64f38", "3be8cce336d0", "3cb9d7266ea1", "3dd0e512b579",
# "3e0a07a7dcfc", "3f29e529f210", "3f6e72d6647d", "40d3650f4985",
# "40db2225676e", "40fddf5f9595", "411a7b067dcc", "41c57fe26957",
# "43d929bd6429", "4425efbbacfc", "44752904b4d5", "44a154410273",
# "44e1c6996c16", "46b08b7eee99", "478a2c53f075", "47c3b766d82e",
# "4810ddb4229c", "499a225c835d", "49d4a04f398c", "4b21aa9b6c84",
# "4b6ba2567ab0", "4b8dc9c901a6", "4bdf75f87261", "4c744a767648",
# "91b6e6e0d84d", "930699898b1c", "957d8951b270", "95e46e2b296f",
# "960479eea44e", "96304c6e06eb", "98fd9ed43654", "9ae06a9d5011",
# "9f1c2cfc936f", "a049e2a265cf", "a136a96476b3", "a55a105360b8",
# "a76fe4d00355", "a9cf3efd023a", "a9fc5e872671", "ab00526e7901",
# "ac877991fa24", "ad12c1357f63", "ad30ecfc1682", "adbaf2416db2",
# "11c2e4fcac6d", "129f894abe35", "13325f865bb0", "1395c3f12b7c",
# "144a7a69f67d", "15aeb12e7a83", "174793807517", "17754cb5b287",
# "182c3da676bd", "1874b96fd317", "194f7e69779b", "1d2396667910",
# "1d618b80769f", "1de9612cb6e1", "1ea4e44e5497", "24a07145b24d",
# "26d58ec4353a", "279107cc7fe4", "296926b5656b", "29dfe87f3a44",
#]

# CORT 
#image_ids = [
# "2cab2cb161a4", "2d8810967b36", "2dc940ff1a71", "2f1b9aea78d7",
# "315b21b955c6", "31e4fa0a83f4", "34b6c5235ab4", "34e41956f993",
# "35fc12883459", "3625dabdf452", "36855e37531a", "3912a0bede5b",
# "3b3991c64f38", "3be8cce336d0", "3cb9d7266ea1", "3dd0e512b579",
# "3e0a07a7dcfc", "3f29e529f210", "3f6e72d6647d", "40d3650f4985",
# "40db2225676e", "40fddf5f9595", "411a7b067dcc", "41c57fe26957",
# "43d929bd6429", "4425efbbacfc", "44752904b4d5", "44a154410273",
# "44e1c6996c16", "46b08b7eee99", "478a2c53f075", "47c3b766d82e",
# "4810ddb4229c", "499a225c835d", "49d4a04f398c", "4b21aa9b6c84",
# "4b6ba2567ab0", "4b8dc9c901a6", "4bdf75f87261", "4c744a767648"
#]

# SH-SY5Y 
#image_ids = [
# "91b6e6e0d84d", "930699898b1c", "957d8951b270", "95e46e2b296f",
# "960479eea44e", "96304c6e06eb", "98fd9ed43654", "9ae06a9d5011",
# "9f1c2cfc936f", "a049e2a265cf", "a136a96476b3", "a55a105360b8",
# "a76fe4d00355", "a9cf3efd023a", "a9fc5e872671", "ab00526e7901",
# "ac877991fa24", "ad12c1357f63", "ad30ecfc1682", "adbaf2416db2",
# "ae3baa051773", "af6ae867fe6e", "af890034c1b6", "aff8fb4fc364",
# "b03de5cbebb2", "b0a5b4340364", "b2a7f3d06a50", "b307d66eb656",
# "b89f9cca5384", "ba9dd157fb69", "bb3520da4cce", "bc0b9c1ff4dc",
# "bfb878cd992e", "c17eac09ff70", "c1f3e3b31108", "c3b32460bcba",
# "c4121689002f", "c7b6b79d6276", "cc40345857dd", "cc8526acd4fe",
#]

# ASTRO 
image_ids = [
 "11c2e4fcac6d", "129f894abe35", "13325f865bb0", "1395c3f12b7c",
 "144a7a69f67d", "15aeb12e7a83", "174793807517", "17754cb5b287",
 "182c3da676bd", "1874b96fd317", "194f7e69779b", "1d2396667910",
 "1d618b80769f", "1de9612cb6e1", "1ea4e44e5497", "24a07145b24d",
 "26d58ec4353a", "279107cc7fe4", "296926b5656b", "29dfe87f3a44",
 "2be2ec84ac11", "2c2cb870da85", "2c7b7d0a1573", "2d9fd17da790",
 "2dbfcf0fc496", "34bd8ce0c802", "37dd4dd6e76e", "393c8540c6fa",
 "3bcc8ba1dc17", "41a1f09b4f4e", "4318b7f15a71", "45a1f06614f0",
 "45b966b60d4b", "47fb5fcff2de", "48383b66ebd5", "4984db4ec8f3",
 "4cd85ba270d0", "4de92f67c5b8", "52ea449bc02d", "52f65c9194c0"
]


# Models
cellpose_model      = models.CellposeModel(gpu = False, pretrained_model = cellpose_model_dir)
cellpose_size_model = models.SizeModel(cellpose_model, pretrained_size  = size_model_dir) 

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.INPUT.MASK_FORMAT = "bitmask"

#cfg.INPUT.MAX_SIZE_TRAIN = 800
#cfg.INPUT.MIN_SIZE_TRAIN = (600, 650, 700, 750, 800)#, 850, 900, 950, 1000)

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]

cfg.MODEL.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]] 

cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 600

cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

cfg.MODEL.RETINANET.NUM_CLASSES = 3
#cfg.MODEL.RETINANET.NUM_CLASSES = 8

cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
#cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256

cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN  = 2000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000


cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
#
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST   = 2000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 1000

#cfg.MODEL.PIXEL_MEAN = [128, 128, 128]
#cfg.MODEL.PIXEL_STD = [11.578, 11.578, 11.578]

cfg.TEST.DETECTIONS_PER_IMAGE = 1000

#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.005
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

cfg.MODEL.WEIGHTS = detectron2_models_dir[0]
predictor_0 = DefaultPredictor(cfg)

#cfg.MODEL.WEIGHTS = detectron2_models_dir[1]
#predictor_1 = DefaultPredictor(cfg)
#
#cfg.MODEL.WEIGHTS = detectron2_models_dir[2]
#predictor_2 = DefaultPredictor(cfg)

#"1de9612cb6e1"

scores = []
cell_scores = {
    "cort"   : [],
    "astro"  : [],
    "shsy5y" : []
}
for image_id in tqdm(image_ids):
    image_dir = os.path.join(data_dir, f"train/{image_id}.png")

    image_grey = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    image_rgb  = cv2.imread(image_dir)

    cfg = load_yaml("./detectron2_src/d2_config.yaml")
    
    det2_pred = detectron2_pipeline_nms(predictor_0(image_rgb), cfg)

#    det2_pred = detectron2_pipeline_ensemble_nms(image_rgb, [predictor_0, predictor_1, predictor_2])

#    det2_pred     = detectron2_pipeline(image_rgb, predictor_0)

    actual_id_df = actual_df[actual_df["id"] == image_id]

    cell_type    = actual_id_df["cell_type"].tolist()[0]
    annotations  = actual_id_df["annotation"].tolist()

    actual = rle_decode(annotations).astype(np.uint16)

#    diam, diam_style = cellpose_size_model.eval(image_grey, channels = [0, 0], augment = True)
#    cellpose_pred, flows, _ = cellpose_model.eval(image_grey, batch_size = 1, diameter = diam, channels = [0, 0], 
#                                          flow_threshold = 0.6, mask_threshold = -1.0,
#                                          augment = True, resample = True, cluster = True)
#

    

    score = iou_map([det2_pred], [actual], verbose = True)

    plot_pred_actual(image_grey, det2_pred, actual, show = True)
#    plot_comparison(image_grey, cellpose_pred, det2_pred, show = False)

    cell_scores[cell_type].append(score)
    scores.append(score)


cort_score   = np.mean(cell_scores["cort"])
astro_score  = np.mean(cell_scores["astro"])
shsy5y_score = np.mean(cell_scores["shsy5y"])

print(np.mean(scores))
print(f"Cort   | Score: {cort_score}")
print(f"Astro  | Score: {astro_score}")
print(f"Shsy5y | Score: {shsy5y_score}")
























