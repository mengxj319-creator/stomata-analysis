"""
Train a Mask R-CNN model (one model per structure, e.g. stomatal complex or aperture) on COCO-format
polygon annotations.

    python train.py train_config.json

Settings (train_config_example.json) default to those used for the published models: ResNet-101 + FPN
initialised with COCO weights, 832 x 832 input, batch size 2, SGD (learning rate 0.001, momentum 0.9,
weight decay 0.0001), 30 epochs of 1000 steps with 50 validation steps, random 90 / 10 train / validation
split, and augmentation of each image with probability 5/6 by one of: horizontal flip, vertical flip,
rotation within +-45 deg, rotation within +-90 deg, scaling 0.5-1.5x (imgaug).

Weights are saved after every epoch in <output_folder>/<run name>/mask_rcnn_<name>_<epoch>.h5.
"""

import json
import os
import random
import sys

import mrcnn.model as modellib
from mrcnn.config import Config

from stomata import config as cfgmod
from stomata.coco import CocoDataset, image_keys

DEFAULTS = {
    "annotations": [],
    "images_folder": "train_images",
    "output_folder": "trained_models",
    "model_name": "stomata",
    "init_weights": "coco",
    "coco_weights": "dnn_model/mask_rcnn_coco.h5",
    "validation_fraction": 0.1,
    "seed": 42,
    "image_size": 832,
    "images_per_gpu": 2,
    "epochs": 30,
    "steps_per_epoch": 1000,
    "validation_steps": 50,
    "learning_rate": 0.001,
    "layers": "heads",
    "augmentation": True,
}


class _TensorBoard(modellib.keras.callbacks.TensorBoard):
    """TensorBoard logging without the profiler (it fails on Windows with non-UTF-8 system locales)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("profile_batch", 0)
        super().__init__(*args, **kwargs)


modellib.keras.callbacks.TensorBoard = _TensorBoard


def augmenter():
    import imgaug.augmenters as iaa
    return iaa.Sometimes(5 / 6, iaa.OneOf([
        iaa.Fliplr(1),
        iaa.Flipud(1),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Affine(rotate=(-90, 90)),
        iaa.Affine(scale=(0.5, 1.5)),
    ]))


def main(config_path):
    with open(config_path, encoding="utf-8") as fh:
        cfg = cfgmod._merge(dict(DEFAULTS), json.load(fh))
    root = os.path.dirname(os.path.abspath(config_path))
    resolve = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
    ann = cfg["annotations"] if isinstance(cfg["annotations"], list) else [cfg["annotations"]]
    ann = [resolve(a) for a in ann]
    if not ann:
        raise ValueError("Set 'annotations' (COCO JSON file or list of files) in the configuration file")

    keys = image_keys(ann)
    random.Random(cfg["seed"]).shuffle(keys)
    n_val = max(1, round(len(keys) * cfg["validation_fraction"]))
    train_keys, val_keys = set(keys[n_val:]), set(keys[:n_val])
    train, val = CocoDataset(), CocoDataset()
    train.load(ann, resolve(cfg["images_folder"]), train_keys)
    val.load(ann, resolve(cfg["images_folder"]), val_keys)
    train.prepare()
    val.prepare()
    print(f"{len(train.image_ids)} training and {len(val.image_ids)} validation images; "
          f"classes: {', '.join(train.class_names[1:])}")

    class TrainConfig(Config):
        NAME = cfg["model_name"]
        NUM_CLASSES = 1 + len(train.class_names[1:])
        GPU_COUNT = 1
        IMAGES_PER_GPU = cfg["images_per_gpu"]
        IMAGE_MIN_DIM = cfg["image_size"]
        IMAGE_MAX_DIM = cfg["image_size"]
        STEPS_PER_EPOCH = cfg["steps_per_epoch"]
        VALIDATION_STEPS = cfg["validation_steps"]
        LEARNING_RATE = cfg["learning_rate"]

    config = TrainConfig()
    config.display()
    out = resolve(cfg["output_folder"])
    os.makedirs(out, exist_ok=True)
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=out)
    init = cfg["init_weights"]
    heads = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    if init == "coco":
        model.load_weights(resolve(cfg["coco_weights"]), by_name=True, exclude=heads)
    elif init == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init == "last":
        model.load_weights(model.find_last(), by_name=True)
    else:  # a weights file, e.g. a previously trained stomata model
        model.load_weights(resolve(init), by_name=True)
    model.train(train, val, learning_rate=config.LEARNING_RATE, epochs=cfg["epochs"], layers=cfg["layers"],
                augmentation=augmenter() if cfg["augmentation"] else None)
    print(f"Weights saved in {model.log_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python train.py train_config.json")
    main(sys.argv[1])
