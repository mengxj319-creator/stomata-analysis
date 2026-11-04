"""Reading the configuration file (JSON) and finding the images."""

import copy
import json
import os

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

DEFAULTS = {
    "images_folder": "images",
    "output_folder": "results",
    "pixels_per_um": None,
    "weights": {"CPA": None, "APA": None},
    "group_weights": {},
    "network_input_size": 832,
    "detection_min_confidence": 0.7,
    "save_overlays": True,
    "spatial": {
        "seed": 42,
        "min_stomata_per_image": 5,
        "compare_groups": [],
        "ripley": {"run": True, "max_distance_um": 200, "step_um": 1, "n_simulations": 1999,
                   "n_permutations": 9999, "size_null_model": False, "n_size_null_simulations": 999},
        "moran": {"run": True, "neighbourhoods": ["knn5"], "n_permutations": 999},
        "gistar": {"run": True, "neighbourhoods": ["knn5"], "z_threshold": 1.65, "n_null_permutations": 99},
        "kde": {"run": True, "bandwidth": 0.2, "grid_size": 200, "threshold": 0.7, "save_maps": True},
    },
}


def _merge(base, new):
    for k, v in new.items():
        if k.startswith("_"):
            continue  # "_comment" entries
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in ("group_weights",):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load(path):
    """Configuration with defaults filled in; relative paths are taken relative to the configuration file."""
    with open(path, encoding="utf-8") as fh:
        cfg = _merge(copy.deepcopy(DEFAULTS), json.load(fh))
    root = os.path.dirname(os.path.abspath(path))
    resolve = lambda p: p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
    cfg["images_folder"] = resolve(cfg["images_folder"])
    cfg["output_folder"] = resolve(cfg["output_folder"])
    cfg["weights"] = {m: resolve(w) for m, w in cfg["weights"].items()}
    cfg["group_weights"] = {g: {m: resolve(w) for m, w in ws.items()} for g, ws in cfg["group_weights"].items()}
    if not cfg["pixels_per_um"]:
        raise ValueError("Set 'pixels_per_um' (pixel scale of your images) in the configuration file")
    return cfg


def list_images(folder):
    """images/<group>/<unit>/<image> or images/<group>/<image> (then the group is a single unit, 'all').
    Returns a list of dicts with group, unit, image, path."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")
    rows = []
    for group in sorted(os.listdir(folder)):
        gdir = os.path.join(folder, group)
        if not os.path.isdir(gdir):
            continue
        for entry in sorted(os.listdir(gdir)):
            path = os.path.join(gdir, entry)
            if os.path.isdir(path):
                rows += [dict(group=group, unit=entry, image=f, path=os.path.join(path, f))
                         for f in sorted(os.listdir(path)) if f.lower().endswith(IMAGE_EXT)]
            elif entry.lower().endswith(IMAGE_EXT):
                rows.append(dict(group=group, unit="all", image=entry, path=path))
    if not rows:
        raise FileNotFoundError(f"No images found in {folder} (expected {folder}/<group>/<unit>/<image>)")
    return rows
