from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.engine import BestCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
import detectron2.data.transforms as T
import detectron2
setup_logger()

from detectron2_src.d2_pipelines import *
from detectron2_src.d2_utils import *
from utils import rle_decode
from metrics import precision_at

import scipy.ndimage as ndi
from scipy import stats

import pycocotools.mask as mask_util
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import copy

CELL_TYPES = ["shsy5y", "astro", "cort"]
THRESHOLDS = [0.15, 0.35, 0.55]

from detectron2.structures import polygons_to_bitmask
def polygon_to_rle(polygon, shape=(520, 704)):
    mask = polygons_to_bitmask([np.asarray(polygon) + 0.25], shape[0], shape[1])

    rle = mask_util.encode(np.asfortranarray(mask))

    return rle

def d2_pipeline(output, actual):
    pred_masks, mask_scores, pred_classes, pred_bboxes = d2_parse_output(output)

    pred_class = stats.mode(pred_classes)[0][0]

    pred_masks = pred_masks > THRESHOLDS[pred_class]

    encoded_preds   = [mask_util.encode(np.asarray(mask, order = 'F')) for mask in pred_masks]
    encoded_actuals = [x["segmentation"] for x in actual]

    # Only for Transfer Learning on LIVECELL Dataset
#    encoded_actuals = [polygon_to_rle(actual[0]) for actual in encoded_actuals]

    ious = mask_util.iou(
        encoded_preds, 
        encoded_actuals, 
        [0] * len(encoded_actuals)
    )

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)

        prec.append(
            tp / (tp + fp + fn)
        )

    return np.mean(prec), CELL_TYPES[pred_class]
#    return np.mean(prec)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset = DatasetCatalog.get(dataset_name)

        self.annotations_cache = {item["image_id"]: item["annotations"] for item in dataset}
            
    def reset(self):
        self.scores = []
        self.class_scores = {
            "cort"   : [],
            "shsy5y" : [],
            "astro"  : []
        }

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            image_id = inp["image_id"]

            actual = self.annotations_cache[image_id]

#            score = d2_pipeline(out, actual)
#            self.scores.append(score)
            score, pred_type = d2_pipeline(out, actual)
            self.class_scores[pred_type].append(score)

    def evaluate(self):
        scores = {
            "MAP IoU | Overall" : np.mean(self.scores),
            "MAP IoU | Astro"   : np.mean(self.class_scores["astro"]),
            "MAP IoU | SH-Y5Y"  : np.mean(self.class_scores["shsy5y"]),
            "MAP IoU | Cort"    : np.mean(self.class_scores["cort"])
        }

#        scores = {
#            "MAP IoU | Overall" : np.mean(self.scores)
#        }

        return scores


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder = None):
        return MAPIOUEvaluator(dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()

        hooks = super().build_hooks()

        best_checkpoint = BestCheckpointer(
            cfg.TEST.EVAL_PERIOD, 
            DetectionCheckpointer(self.model, cfg.OUTPUT_DIR), 
            "MAP IoU | Overall", 
            "max"
        )

        hooks.insert(-1, best_checkpoint)

        return hooks




