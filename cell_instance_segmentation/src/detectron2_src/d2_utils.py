import numpy as np

def d2_ensemble_outputs(outputs):
    masks   = []
    scores  = []
    bboxes  = []
    classes = []
    for output in outputs:
        pred_masks, mask_scores, pred_classes, pred_bboxes = d2_parse_output(output)

        masks.extend(pred_masks)
        scores.extend(mask_scores)
        bboxes.extend(pred_bboxes)
        classes.extend(pred_classes)

    return np.array(masks), np.array(scores), np.array(classes), np.array(bboxes)


def d2_parse_output(predictor_output):
    """
    Returns valuable information from Detectron2 predictor output.

    Args:
        coco_output (dictionary) : Output from Detectron2 COCO Predictor.

    Returns:
        pred_masks  (np.array) : Each predicted binary mask. 
        mask_scores (np.array) : Score of each mask.
        pred_class  (int)      : Integer representing predicted class.
        pred_bboxed (np.array) : Predicted Boxes. 

    """
    instances = predictor_output["instances"]

    pred_bboxes  = instances.pred_boxes.tensor.cpu().numpy()
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks   = instances.pred_masks.cpu().numpy()  
    mask_scores  = instances.scores.cpu().numpy()

    return pred_masks, mask_scores, pred_classes, pred_bboxes

def nms(dets, scores, thresh):
    """
    Custom NMS function.

    Taken from Fast-RCNN GitHub Page:
    https://github.com/rbgirshick/fast-rcnn/blob/b612190f279da3c11dd8b1396dd5e72779f8e463/lib/utils/nms.py

    Args:
        dets   (np.array) : boxes
        scores (np.array) : scores
        thresh (float)    : nms thresh

    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[:: -1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box

        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]

    return keep

