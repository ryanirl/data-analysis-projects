#############################################################
# REFERENCE: 
# 
# These metric functions are from: 
# https://www.kaggle.com/theoviel/competition-metric-map-iou
#############################################################

import numpy as np 

def compute_iou(pred, actual):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        actual (np.ndarray): Actual mask.
        pred (np.ndarray): Predicted mask.

    Returns:
        np.ndarray: IoU matrix, of size true_objects x pred_objects.

    """
    pred_objects = len(np.unique(pred))
    true_objects = len(np.unique(actual))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        pred.flatten(), actual.flatten(), 
        bins = (pred_objects, true_objects)

    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_pred = np.histogram(pred,   bins = pred_objects)[0]
    area_true = np.histogram(actual, bins = true_objects)[0]

    area_pred = np.expand_dims(area_pred, -1)
    area_true = np.expand_dims(area_true, 0)

    # Compute union
    union = area_pred + area_true - intersection

    # exclude background
    intersection = intersection[1:, 1:] 
    union        = union[1:, 1:]

    union[union == 0] = 1e-9

    iou = intersection / union
    
    return iou     


def precision_at(threshold, iou):
    matches = iou > threshold

    true_positives  = np.sum(matches, axis = 1) >= 1  # Correct
    false_positives = np.sum(matches, axis = 1) == 0  # Extra 
    false_negatives = np.sum(matches, axis = 0) == 0  # Missed

    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives)
    )

    return tp, fp, fn


def iou_map(preds, actuals, verbose = 0):
    """
    Computes the metric for the competition.

    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        preds (list of masks): Predictions.
        actuals (list of masks): Ground truths.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.

    """
    ious = [compute_iou(pred, actual) for pred, actual in zip(preds, actuals)]
    
    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)





