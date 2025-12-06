#!/usr/bin/env python3
"""
train_detectron2_optimized.py
Optimized Detectron2 Mask R-CNN training for text detection on RTX A5000.
Uses mixed precision, fast dataloading, and CuDNN tuning for maximum throughput.
"""
import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

# ---------------------------------------------------------------------
#  GLOBAL SETTINGS
# ---------------------------------------------------------------------
torch.backends.cudnn.benchmark = True  # Autotune fastest CuDNN kernels

COCO_ANN_DIR = "dataset/coco_annotations_fixed"
IMAGES_DIR = "dataset/images"
OUTPUT_DIR = "output_detectron2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
#  REGISTER DATASETS
# ---------------------------------------------------------------------
register_coco_instances("text_train", {}, os.path.join(COCO_ANN_DIR, "train_coco.json"), IMAGES_DIR)
register_coco_instances("text_val", {}, os.path.join(COCO_ANN_DIR, "val_coco.json"), IMAGES_DIR)
register_coco_instances("text_test", {}, os.path.join(COCO_ANN_DIR, "test_coco.json"), IMAGES_DIR)

for split in ["text_train", "text_val", "text_test"]:
    MetadataCatalog.get(split).set(thing_classes=["text"])

# ---------------------------------------------------------------------
#  MAIN TRAINING
# ---------------------------------------------------------------------
def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Datasets
    cfg.DATASETS.TRAIN = ("text_train",)
    cfg.DATASETS.TEST = ("text_val",)

    # -----------------------------------------------------------------
    #  OPTIMIZATION SECTION
    # -----------------------------------------------------------------
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f" Detected GPU VRAM: {total_vram_gb:.1f} GB ({torch.cuda.get_device_name(0)})")

    # Adjust batch size based on available VRAM
    if total_vram_gb >= 22:
        cfg.SOLVER.IMS_PER_BATCH = 6  # optimal for RTX A5000 24GB
    elif total_vram_gb >= 12:
        cfg.SOLVER.IMS_PER_BATCH = 4
    else:
        cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.BASE_LR = 0.00025 * (cfg.SOLVER.IMS_PER_BATCH / 6)
    cfg.SOLVER.AMP.ENABLED = True  # Mixed precision

    # DataLoader tuning
    cfg.DATALOADER.PIN_MEMORY = True
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = min(os.cpu_count(), 16)

    # Resize only to one scale (faster)
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Training schedule
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 30000        
    cfg.SOLVER.STEPS = (25000,)      
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.TEST.EVAL_PERIOD = 0

    # Model parameters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only "text"

    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    #  LOGGING
    # -----------------------------------------------------------------
    print(f" CUDA available: {torch.cuda.is_available()}")
    print(f" Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f" DataLoader workers: {cfg.DATALOADER.NUM_WORKERS}")
    print(f" Mixed Precision: {cfg.SOLVER.AMP.ENABLED}")
    print(f" Base LR: {cfg.SOLVER.BASE_LR}")
    print(f" Output: {cfg.OUTPUT_DIR}\n")

    # -----------------------------------------------------------------
    #  TRAIN
    # -----------------------------------------------------------------
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


if __name__ == "__main__":
    main()
