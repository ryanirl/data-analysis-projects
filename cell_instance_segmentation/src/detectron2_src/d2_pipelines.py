from pathlib import Path
import pandas as pd
import numpy as np
import torch
import time 
import yaml
import cv2
import sys

sys.path.append("../")

from scipy import stats

from detectron2.layers.mask_ops import _do_paste_mask
from detectron2.structures.masks import ROIMasks
import detectron2

from detectron2_src.d2_utils import *
from monkey_patches import *
from utils import *

detectron2.layers.mask_ops.paste_masks_in_image.__code__ = paste_masks_in_image.__code__
ROIMasks.to_bitmasks = to_bitmasks

THRESHOLDS = [.15, .35, .55]
MIN_PIXELS = [75, 150, 75]

def detectron2_pipeline(output, cfg):
    pred_masks, mask_scores, pred_classes, pred_bboxes = d2_parse_output(output)

    pred_class = stats.mode(pred_classes)[0][0]

    cfg = cfg[pred_class]

    score_idx  = mask_scores >= THRESHOLDS[pred_class]

    pred_masks = pred_masks[score_idx]

    pred_masks = pred_masks > cfg["pixel_score"]

    prediction = np.zeros((520, 704))
    for n, mask in enumerate(pred_masks, 1):
        mask = mask * (1 - prediction)
        mask = mask > 0

        mask = clean_binary_mask(mask)

        prediction[mask] = n

    return prediction.astype(np.uint16)


def detectron2_pipeline_nms(output, cfg):
    return _detectron2_pipeline_nms(
        *d2_parse_output(output), cfg
    )

def detectron2_pipeline_ensemble_nms(outputs, cfg):
    return _detectron2_pipeline_nms(
        *d2_ensemble_outputs(outputs), cfg
    )

def _detectron2_pipeline_nms(pred_masks, mask_scores, pred_classes, pred_boxes, cfg):
    pred_class = stats.mode(pred_classes)[0][0]

    cfg = cfg[pred_class]

    nms_idx = nms(pred_boxes, mask_scores, cfg["nms"])

    pred_classes = pred_classes[nms_idx]
    pred_masks   = pred_masks[nms_idx]
    mask_scores  = mask_scores[nms_idx]
    pred_boxes   = pred_boxes[nms_idx]

    mask_areas = np.sum(pred_masks, axis = (1, 2))

    # Removes masks with too low of a score or with an outlier area.
    score_idx    = mask_scores >= cfg["score_threshold"]
    min_area_idx = mask_areas  >= cfg["min_area"]
    max_area_idx = mask_areas  <= cfg["max_area"]

    idx = np.logical_and.reduce((
        score_idx,
        min_area_idx,
        max_area_idx
    ))

    pred_masks  = pred_masks[idx]
    mask_scores = mask_scores[idx]
    pred_boxes  = pred_boxes[idx]

    pred_masks = pred_masks > cfg["pixel_score"]

    binary_mask = np.zeros((520, 704))
    for i, mask in enumerate(pred_masks, 1):
        mask = mask * (1 - binary_mask)
        mask = mask > 0

        mask = clean_binary_mask(mask)

        binary_mask[mask] = i

    return binary_mask


