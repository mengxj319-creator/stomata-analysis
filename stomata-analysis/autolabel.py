"""
Auto-labelling: detect instances with a trained model and save them as COCO-format polygon annotations,
to be corrected in an annotation tool (e.g. Makesense.ai) and used for training (semi-automated annotation).

    python autolabel.py train_config.json

Uses from the configuration file: images_folder, autolabel_weights, autolabel_output, autolabel_category,
image_size and detection_min_confidence.
"""

import json
import os
import sys

import cv2

from stomata import config as cfgmod
from stomata.coco import detections_to_coco
from stomata.detection import detect, load_model


def main(config_path):
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    root = os.path.dirname(os.path.abspath(config_path))
    resolve = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
    folder = resolve(cfg.get("images_folder", "train_images"))
    out = resolve(cfg.get("autolabel_output", "annotations/autolabel.json"))
    model = load_model(resolve(cfg["autolabel_weights"]), cfg.get("image_size", 832),
                       cfg.get("detection_min_confidence", 0.7))
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(cfgmod.IMAGE_EXT)]
    results = []
    for k, f in enumerate(files, 1):
        img = cv2.imread(os.path.join(folder, f))
        _, contours = detect(model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results.append((f, img.shape[1], img.shape[0], contours))
        print(f"  [{k}/{len(files)}] {f}: {len(contours)} instances")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(detections_to_coco(results, cfg.get("autolabel_category", "stomata")), fh)
    print(f"Saved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python autolabel.py train_config.json")
    main(sys.argv[1])
