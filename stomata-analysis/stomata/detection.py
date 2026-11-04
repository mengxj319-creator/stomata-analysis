"""
Mask R-CNN inference and per-stoma measurements.

For every detected instance:
  - centroid (image moments of the mask)
  - area (sum of the areas of the mask contours)
  - length and width (long and short side of the minimum-area rectangle of the largest mask contour)
  - detection score and bounding box

Aperture (APA model) and stomatal complex (CPA model) detections are paired one-to-one by the nearest
mask centroid, which gives guard cell area (GCA = CPA - APA) and junction length
(complex length - aperture length) for each stoma.
"""

import cv2
import numpy as np

import mrcnn.model as modellib
from mrcnn.visualize import InferenceConfig, get_mask_contours


def load_model(weights, image_size=832, min_confidence=0.7):
    """weights: .h5 file; image_size: network input size (px); min_confidence: detection score threshold."""
    config = InferenceConfig(num_classes=1, image_size=image_size)
    config.DETECTION_MIN_CONFIDENCE = min_confidence
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
    model.load_weights(filepath=weights, by_name=True)
    return model


def detect(model, image_rgb):
    """Return (records, contours): one record per detected instance (pixel units) and its mask contours."""
    result = model.detect([image_rgb], verbose=0)[0]
    records, contours_all = [], []
    for i in range(len(result["class_ids"])):
        mask = result["masks"][:, :, i].astype(np.uint8)
        mom = cv2.moments(mask)
        if mom["m00"] == 0:
            continue
        contours = get_mask_contours(mask)
        if not contours:
            continue
        main = max(contours, key=cv2.contourArea)
        side_a, side_b = cv2.minAreaRect(main)[1]
        y1, x1, y2, x2 = result["rois"][i]
        records.append(dict(cx_px=mom["m10"] / mom["m00"], cy_px=mom["m01"] / mom["m00"],
                            area_px=sum(cv2.contourArea(c) for c in contours),
                            length_px=max(side_a, side_b), width_px=min(side_a, side_b),
                            score=float(result["scores"][i]),
                            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)))
        contours_all.append(contours)
    return records, contours_all


def pair_apa_cpa(apa, cpa):
    """Pair each aperture with the stomatal complex whose mask centroid is nearest (Euclidean distance;
    one-to-one: an aperture whose nearest complex is already taken is left unpaired).
    Returns (apa_index, cpa_index) pairs."""
    centre = lambda r: (r["cx_px"], r["cy_px"])
    if not apa or not cpa:
        return []
    cpa_centres = np.array([centre(r) for r in cpa], dtype=float)
    pairs, taken = [], set()
    for i, r in enumerate(apa):
        j = int(np.argmin(np.linalg.norm(cpa_centres - np.array(centre(r), dtype=float), axis=1)))
        if j in taken:
            continue
        taken.add(j)
        pairs.append((i, j))
    return pairs


def draw_overlay(img_bgr, cpa_contours, apa_contours):
    """Detections drawn on the image: CPA outlines in green, APA outlines in red."""
    out = img_bgr.copy()
    for contours in cpa_contours:
        cv2.polylines(out, contours, True, (0, 255, 0), 3)
    for contours in apa_contours:
        cv2.polylines(out, contours, True, (0, 0, 255), 3)
    return out
