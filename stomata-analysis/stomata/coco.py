"""
COCO-format polygon annotations: reading them for training, and writing detections as annotations
(auto-labelling) so that they can be corrected in an annotation tool.
"""

import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mrcnn import utils


class CocoDataset(utils.Dataset):
    """Instances from one or more COCO JSON files (polygon segmentations). Image files are looked up by
    'file_name' in images_folder."""

    def load(self, annotation_files, images_folder, image_ids=None):
        """image_ids: optional subset of (file index, image id) keys, e.g. for a train / validation split."""
        categories = {}
        for k, path in enumerate(annotation_files):
            with open(path, encoding="utf-8") as fh:
                coco = json.load(fh)
            for c in coco["categories"]:
                categories.setdefault(c["name"], len(categories) + 1)
            cat_name = {c["id"]: c["name"] for c in coco["categories"]}
            anns = {}
            for a in coco["annotations"]:
                anns.setdefault(a["image_id"], []).append(dict(a, class_id=categories[cat_name[a["category_id"]]]))
            for im in coco["images"]:
                key = (k, im["id"])
                if image_ids is not None and key not in image_ids:
                    continue
                if not anns.get(im["id"]):
                    continue  # images without annotated instances are skipped
                self.add_image("stomata", image_id=f"{k}_{im['id']}",
                               path=os.path.join(images_folder, im["file_name"]),
                               width=im["width"], height=im["height"], annotations=anns[im["id"]])
        for name, cid in sorted(categories.items(), key=lambda x: x[1]):
            self.add_class("stomata", cid, name)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        masks, class_ids = [], []
        for a in info["annotations"]:
            m = Image.new("1", (info["width"], info["height"]))
            draw = ImageDraw.ImageDraw(m, "1")
            for poly in a["segmentation"]:
                draw.polygon(poly, fill=1)
            masks.append(np.array(m) > 0)
            class_ids.append(a["class_id"])
        return np.dstack(masks), np.array(class_ids, dtype=np.int32)


def image_keys(annotation_files):
    """All (file index, image id) keys that have at least one annotation."""
    keys = []
    for k, path in enumerate(annotation_files):
        with open(path, encoding="utf-8") as fh:
            coco = json.load(fh)
        annotated = {a["image_id"] for a in coco["annotations"]}
        keys += [(k, im["id"]) for im in coco["images"] if im["id"] in annotated]
    return keys


def detections_to_coco(results, category_name):
    """results: list of (file_name, width, height, list of contour lists). One polygon per contour."""
    coco = dict(images=[], annotations=[], categories=[dict(id=1, name=category_name)])
    ann_id = 1
    for img_id, (file_name, width, height, instances) in enumerate(results, 1):
        coco["images"].append(dict(id=img_id, file_name=file_name, width=int(width), height=int(height)))
        for contours in instances:
            polys = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
            if not polys:
                continue
            pts = np.vstack(polys)
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            coco["annotations"].append(dict(
                id=ann_id, image_id=img_id, category_id=1, iscrowd=0,
                segmentation=[p.astype(float).ravel().tolist() for p in polys],
                area=float(sum(cv2.contourArea(p.astype(np.float32)) for p in polys)),
                bbox=[int(x), int(y), int(w), int(h)]))
            ann_id += 1
    return coco
