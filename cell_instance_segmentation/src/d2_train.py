from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
import detectron2
setup_logger()

from pathlib import Path
import os
import time

from detectron2_src.d2_trainer import Trainer

###########################################
#####        Prepare Dataset          #####
###########################################

data_dir = Path("../data/")
data_dir_livecell = Path("../data/livecell_transfer/LIVECell_dataset_2021/images")

fold = 2

coco_train_dir = f"../notebooks/data_preparation/coco_formated/fold_{fold}_train.json"
coco_val_dir   = f"../notebooks/data_preparation/coco_formated/fold_{fold}_test.json"

train_reg = "sartorius_train"
val_reg   = "sartorius_val"

register_coco_instances(train_reg, {}, coco_train_dir, data_dir)
register_coco_instances(val_reg,   {}, coco_val_dir,   data_dir)

#register_coco_instances("livecell_train", {}, "../data/livecell_transfer/livecell_annotations_train.json", data_dir_livecell)
#register_coco_instances("livecell_test", {}, "../data/livecell_transfer/livecell_annotations_test.json", data_dir_livecell)
#register_coco_instances("livecell_val", {}, "../data/livecell_transfer/livecell_annotations_val.json", data_dir_livecell)

metadata = MetadataCatalog.get(train_reg)
train_ds = DatasetCatalog.get(train_reg)

#metadata = MetadataCatalog.get("livecell_train")
#train_ds = DatasetCatalog.get("livecell_train")

###########################################
#####         Setup Config            #####
###########################################

EPOCHS = 100
BATCH_SIZE = 2
PER_EPOCH_ITER = len(train_ds) // BATCH_SIZE
TOTAL_ITERS = EPOCHS * PER_EPOCH_ITER

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.OUTPUT_DIR    = "./sartorius_main_transfer"
cfg.MODEL.WEIGHTS = "./livecell_transfer/model_final.pth"

# FINE TUNING w/ TINY LR 
cfg.MODEL.BACKBONE.FREEZE_AT = 0

cfg.INPUT.MASK_FORMAT = "bitmask"

cfg.SOLVER.CHECKPOINT_PERIOD = PER_EPOCH_ITER
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.SOLVER.MAX_ITER = TOTAL_ITERS
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.STEPS = []

cfg.DATALOADER.NUM_WORKERS = 4

cfg.TEST.EVAL_PERIOD = PER_EPOCH_ITER * 4

cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TRAIN = (600, 650, 700, 750, 800)#, 850, 900, 950, 1000)

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]

cfg.MODEL.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]] 

#cfg.INPUT.MAX_SIZE_TEST = 1024
#cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 700

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

cfg.MODEL.RETINANET.NUM_CLASSES = 3
#cfg.MODEL.RETINANET.NUM_CLASSES = 8

cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN  = 2000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000

cfg.MODEL.RPN.PRE_NMS_TOPK_TEST   = 2000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 1000

cfg.DATASETS.TRAIN = ("sartorius_train",)
cfg.DATASETS.TEST  = ("sartorius_val"  ,)

#cfg.DATASETS.TRAIN = ("livecell_train",)
#cfg.DATASETS.TEST  = ("livecell_test"  ,)

cfg.TEST.DETECTIONS_PER_IMAGE = 1000

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

predictor = DefaultPredictor(cfg)

print(f"NUM ITERS: {cfg.SOLVER.MAX_ITER}")
print(f"NUM EPOCHS: {cfg.SOLVER.MAX_ITER}")

os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
if __name__ == "__main__":
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume = False)
    trainer.train()





