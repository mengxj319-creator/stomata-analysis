"""
Step 1: detect stomata with the Mask R-CNN models and measure stomatal traits.

    python detect.py config.json

Input: images in <images_folder>/<group>/<unit>/ (see README). Output, in <output_folder>:
  stomata.csv      one row per detection: model (CPA or APA), centroid, area, length, width, score, box
  images.csv       one row per image: size, number of detections, stomatal density
  paired.csv       one row per stoma with both an aperture and a complex detection: CPA, APA, GCA = CPA - APA,
                   lengths and widths, junction length = complex length - aperture length, APA/CPA area ratio
  traits_units.csv unit means (mean of the image means) of density and of the paired traits
  overlays/        images with the detections drawn (CPA green, APA red), if save_overlays is true
"""

import os
import sys
import time

import cv2
import pandas as pd

from stomata import config as cfgmod
from stomata.detection import detect, draw_overlay, load_model, pair_apa_cpa

TRAITS = ["CPA_um2", "APA_um2", "GCA_um2", "complex_length_um", "complex_width_um", "aperture_length_um",
          "aperture_width_um", "junction_length_um", "APA_CPA_ratio"]


def main(config_path):
    cfg = cfgmod.load(config_path)
    out = cfg["output_folder"]
    os.makedirs(out, exist_ok=True)
    um = 1 / cfg["pixels_per_um"]
    images = cfgmod.list_images(cfg["images_folder"])
    groups = sorted({r["group"] for r in images})
    print(f"{len(images)} images in {len(groups)} group(s): {', '.join(groups)}")

    def weights_for(group, model):
        return cfg["group_weights"].get(group, {}).get(model) or cfg["weights"].get(model)

    models = [m for m in ("CPA", "APA") if any(weights_for(g, m) for g in groups)]
    if not models:
        raise ValueError("No model weights given: set 'weights' in the configuration file")
    jobs = {}  # (model, weights file) -> images
    for m in models:
        for info in images:
            w = weights_for(info["group"], m)
            if w:
                jobs.setdefault((m, w), []).append(info)

    stomata, detections, contours = [], {}, {}
    for (m, w), job in jobs.items():
        if not os.path.isfile(w):
            raise FileNotFoundError(f"Weights not found: {w}")
        print(f"\n{m} model: {w} ({len(job)} images)")
        model = load_model(w, cfg["network_input_size"], cfg["detection_min_confidence"])
        t0 = time.time()
        for k, info in enumerate(job, 1):
            img = cv2.imread(info["path"])
            recs, cnts = detect(model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            detections[(info["path"], m)] = recs
            contours[(info["path"], m)] = cnts
            info["width_px"], info["height_px"] = img.shape[1], img.shape[0]
            for j, r in enumerate(recs):
                stomata.append(dict(group=info["group"], unit=info["unit"], image=info["image"], model=m,
                                    weights=os.path.basename(w), stoma_id=j,
                                    cx_px=r["cx_px"], cy_px=r["cy_px"], cx_um=r["cx_px"] * um, cy_um=r["cy_px"] * um,
                                    area_um2=r["area_px"] * um ** 2, length_um=r["length_px"] * um,
                                    width_um=r["width_px"] * um, score=r["score"],
                                    x1=r["x1"], y1=r["y1"], x2=r["x2"], y2=r["y2"]))
            print(f"  [{k}/{len(job)}] {info['group']}/{info['unit']}/{info['image']}: {len(recs)} detections")
        print(f"  {(time.time() - t0) / len(job):.1f} s per image")

    image_rows, paired = [], []
    for info in images:
        w_mm, h_mm = info["width_px"] * um / 1000, info["height_px"] * um / 1000
        row = dict(group=info["group"], unit=info["unit"], image=info["image"], path=info["path"],
                   width_px=info["width_px"], height_px=info["height_px"],
                   width_um=info["width_px"] * um, height_um=info["height_px"] * um)
        for m in models:
            row[f"n_{m}"] = len(detections.get((info["path"], m), []))
        if "CPA" in models:
            row["density_mm2"] = row["n_CPA"] / (w_mm * h_mm)
        image_rows.append(row)
        apa = detections.get((info["path"], "APA"), [])
        cpa = detections.get((info["path"], "CPA"), [])
        for i, j in pair_apa_cpa(apa, cpa):
            a, c = apa[i], cpa[j]
            paired.append(dict(group=info["group"], unit=info["unit"], image=info["image"], apa_id=i, cpa_id=j,
                               CPA_um2=c["area_px"] * um ** 2, APA_um2=a["area_px"] * um ** 2,
                               GCA_um2=(c["area_px"] - a["area_px"]) * um ** 2,
                               complex_length_um=c["length_px"] * um, complex_width_um=c["width_px"] * um,
                               aperture_length_um=a["length_px"] * um, aperture_width_um=a["width_px"] * um,
                               junction_length_um=(c["length_px"] - a["length_px"]) * um,
                               APA_CPA_ratio=a["area_px"] / c["area_px"]))
        if cfg["save_overlays"]:
            odir = os.path.join(out, "overlays", info["group"], info["unit"])
            os.makedirs(odir, exist_ok=True)
            cv2.imwrite(os.path.join(odir, info["image"]),
                        draw_overlay(cv2.imread(info["path"]), contours.get((info["path"], "CPA"), []),
                                     contours.get((info["path"], "APA"), [])))

    stomata, image_df, paired = pd.DataFrame(stomata), pd.DataFrame(image_rows), pd.DataFrame(paired)
    stomata.to_csv(os.path.join(out, "stomata.csv"), index=False)
    image_df.to_csv(os.path.join(out, "images.csv"), index=False)
    paired.to_csv(os.path.join(out, "paired.csv"), index=False)

    # unit means of the image means
    per_image = image_df[["group", "unit", "image"] + (["density_mm2"] if "CPA" in models else [])]
    if len(paired):
        per_image = per_image.merge(paired.groupby(["group", "unit", "image"])[TRAITS].mean().reset_index(),
                                    on=["group", "unit", "image"], how="left")
    units = per_image.groupby(["group", "unit"]).mean(numeric_only=True)
    units.insert(0, "n_images", per_image.groupby(["group", "unit"]).size())
    units.reset_index().to_csv(os.path.join(out, "traits_units.csv"), index=False)
    print(f"\nSaved stomata.csv, images.csv, paired.csv and traits_units.csv to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python detect.py config.json")
    main(sys.argv[1])
