"""
Step 2: spatial pattern analysis of the detected stomata.

    python spatial.py config.json

Reads stomata.csv and images.csv written by detect.py (or files in the same format, see README) from
<output_folder> and writes the results to <output_folder>/spatial/. Analyses (each can be switched off in the
configuration file):

  Ripley's K / L    point pattern of the stomatal centroids (CPA detections) per unit: global CSR envelope test,
                    RPSD, density-scaled RPSD* and L*(r*); optional size-aware (hard-core) null model;
                    groups compared by Welch / Mann-Whitney tests and studentized permutation tests of the curves
  Moran's I         spatial autocorrelation of CPA and APA per image, for each neighbourhood definition
  Getis-Ord Gi*     fraction of hot/cold spots per image, compared with a within-image permutation null
  KDE               density maps per image, high-density extent, mean and std maps per group

Units (e.g. plants) are the experimental units: image values are averaged per unit before groups are compared.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from stomata import autocorrelation as ac  # noqa: E402
from stomata import config as cfgmod  # noqa: E402
from stomata import kde, ripley  # noqa: E402
from stomata.compare import compare_units, group_pairs  # noqa: E402

KEYS = ["group", "unit", "image"]


def point_model(stomata):
    """Model whose centroids define the point pattern: CPA (whole stomatal complex) if available."""
    return "CPA" if (stomata.model == "CPA").any() else "APA"


def unit_patterns(stomata, images, model):
    """{(group, unit): [(points_um, bounds_um), ...]} with one pattern per image."""
    s = stomata[stomata.model == model]
    out = {}
    for _, im in images.iterrows():
        pts = s.loc[(s.group == im.group) & (s.unit == im.unit) & (s.image == im.image), ["cx_um", "cy_um"]]
        out.setdefault((im.group, im.unit), []).append((pts.to_numpy(), (0, im.width_um, 0, im.height_um)))
    return out


def run_ripley(stomata, images, cfg, pairs, out, rng_seed):
    rc = cfg["spatial"]["ripley"]
    max_d, step = float(rc["max_distance_um"]), float(rc["step_um"])
    r = np.arange(0, max_d + step / 2, step)
    model = point_model(stomata)
    patterns = unit_patterns(stomata, images, model)
    rng = np.random.default_rng(rng_seed)
    usable = {k: [(p, b) for p, b in v if len(p) >= ripley.MIN_POINTS] for k, v in patterns.items()}
    lam_min = min(len(p) / ripley.window_area(b) for v in usable.values() for p, b in v)
    rstar = np.linspace(0, max_d * np.sqrt(lam_min), 101)  # r = r*/sqrt(lambda) stays <= max_d for every unit
    rows, curves = [], {}
    for (g, u), pats in sorted(usable.items()):
        if not pats:
            continue
        res = ripley.analyze_unit(pats, r, max_d, rc["n_simulations"], rng)
        curves[f"L|{g}|{u}"] = res.pop("L_mean")
        curves[f"lower|{g}|{u}"], curves[f"upper|{g}|{u}"] = res.pop("lower"), res.pop("upper")
        curves[f"Lstar|{g}|{u}"] = ripley.scaled_curve(pats, rstar, max_d)
        if rc["size_null_model"]:
            s = stomata[(stomata.model == model) & (stomata.group == g) & (stomata.unit == u)]
            lx, ly = ((s.x2 - s.x1) / cfg["pixels_per_um"]).median(), ((s.y2 - s.y1) / cfg["pixels_per_um"]).median()
            hc = ripley.size_null_model(pats, lx, ly, r, max_d, rc["n_size_null_simulations"], rng,
                                        curves[f"lower|{g}|{u}"])
            curves[f"Lsize|{g}|{u}"] = hc.pop("L_size_null")
            res.update(stoma_length_um=lx, stoma_width_um=ly, **hc)
        rows.append(dict(group=g, unit=u, points_from=model, **res))
        print(f"  Ripley {g}/{u}: P = {res['global_p']:.4f}, RPSD = {res['rpsd_um']}")
    units = pd.DataFrame(rows)
    units.to_csv(os.path.join(out, "ripley_units.csv"), index=False)
    np.savez(os.path.join(out, "ripley_curves.npz"), r=r, rstar=rstar, **curves)

    cols = ["rpsd_um", "rpsd_star", "density_mm2", "mnnd_um"]
    if rc["size_null_model"]:
        cols += ["rpsd_expected_from_size_um", "mean_deviation_from_size_null"]
    tests = compare_units(units, cols, pairs)
    curve_tests = []
    for a, b in pairs:
        ka = [f"{a}|{u}" for u in units[units.group == a].unit]
        kb = [f"{b}|{u}" for u in units[units.group == b].unit]
        if len(ka) < 2 or len(kb) < 2:
            continue
        labels = np.array([True] * len(ka) + [False] * len(kb))
        for name, prefix in (("L(r)", "L"), ("L*(r*)", "Lstar")):
            C = np.array([curves[f"{prefix}|{k}"] for k in ka + kb])
            t0, p = ripley.studentized_perm_test(C, labels, rc["n_permutations"], np.random.default_rng(rng_seed))
            curve_tests.append(dict(variable=f"curve {name}", group_a=a, group_b=b, n_a=len(ka), n_b=len(kb),
                                    statistic=t0, permutation_p=p))
    pd.concat([tests, pd.DataFrame(curve_tests)], ignore_index=True).to_csv(
        os.path.join(out, "ripley_tests.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for g in sorted(units.group.unique()):
        L = np.array([curves[f"L|{g}|{u}"] for u in units[units.group == g].unit])
        m = L.mean(axis=0)
        ax.plot(r, m, label=f"{g} (n = {len(L)})")
        if len(L) > 1:
            se = L.std(axis=0, ddof=1) / np.sqrt(len(L))
            ax.fill_between(r, m - se, m + se, alpha=0.25)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Distance r (µm)")
    ax.set_ylabel("L(r) (µm), mean ± SE over units")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ripley_L.png"), dpi=200)
    plt.close(fig)


def run_moran(stomata, cfg, pairs, out, rng_seed):
    mc = cfg["spatial"]["moran"]
    rng = np.random.default_rng(rng_seed)
    rows = []
    for (g, u, im, model), d in stomata.groupby(KEYS + ["model"]):
        if len(d) < cfg["spatial"]["min_stomata_per_image"]:
            continue
        pts, vals = d[["cx_um", "cy_um"]].to_numpy(), d.area_um2.to_numpy()
        for nb in mc["neighbourhoods"]:
            I, EI, p, isl = ac.morans_i(vals, ac.weights(pts, nb), mc["n_permutations"], rng)
            rows.append(dict(group=g, unit=u, image=im, trait=model, neighbourhood=nb, n=len(d), I=I, EI=EI, p=p,
                             share_without_neighbours=isl, significant_positive=(p < 0.05) and (I > EI)))
    img = pd.DataFrame(rows)
    img.to_csv(os.path.join(out, "moran_images.csv"), index=False)
    units = (img.groupby(["trait", "neighbourhood", "group", "unit"])
             .agg(I_mean=("I", "mean"), share_significant_positive=("significant_positive", "mean"),
                  n_images=("I", "size")).reset_index())
    units.to_csv(os.path.join(out, "moran_units.csv"), index=False)
    compare_units(units, ["I_mean"], pairs, by=("trait", "neighbourhood")).to_csv(
        os.path.join(out, "moran_tests.csv"), index=False)
    print(f"  Moran's I: {len(img)} image x trait x neighbourhood results")


def run_gistar(stomata, cfg, pairs, out, rng_seed):
    gc = cfg["spatial"]["gistar"]
    t = gc["z_threshold"]
    rng = np.random.default_rng(rng_seed)
    rows = []
    for (g, u, im, model), d in stomata.groupby(KEYS + ["model"]):
        if len(d) < cfg["spatial"]["min_stomata_per_image"]:
            continue
        pts, x = d[["cx_um", "cy_um"]].to_numpy(), d.area_um2.to_numpy()
        for nb in gc["neighbourhoods"]:
            W = ac.weights(pts, nb)
            z = ac.gistar_z(x, W)
            null = np.array([ac.gistar_z(rng.permutation(x), W) for _ in range(gc["n_null_permutations"])])
            rows.append(dict(group=g, unit=u, image=im, trait=model, neighbourhood=nb, n=len(x),
                             hot=float((z > t).mean()), cold=float((z < -t).mean()),
                             flagged=float((np.abs(z) > t).mean()), flagged_null=float((np.abs(null) > t).mean()),
                             fdr_hits=ac.bh_count(z)))
    img = pd.DataFrame(rows)
    img["excess"] = img.flagged - img.flagged_null
    img["any_fdr_hit"] = img.fdr_hits > 0
    img.to_csv(os.path.join(out, "gistar_images.csv"), index=False)
    by = ["trait", "neighbourhood", "group", "unit"]
    units = img.groupby(by)[["hot", "cold", "flagged", "flagged_null", "excess", "any_fdr_hit"]].mean().reset_index()
    units.to_csv(os.path.join(out, "gistar_units.csv"), index=False)
    # excess of flagged points over the permutation null, per group (one-sample t-test against 0)
    summary = []
    for (trait, nb, g), d in units.groupby(["trait", "neighbourhood", "group"]):
        summary.append(dict(trait=trait, neighbourhood=nb, group=g, n_units=len(d), flagged=d.flagged.mean(),
                            flagged_null=d.flagged_null.mean(), excess=d.excess.mean(),
                            p_excess_vs_0=stats.ttest_1samp(d.excess, 0).pvalue if len(d) >= 2 else np.nan))
    pd.DataFrame(summary).to_csv(os.path.join(out, "gistar_groups.csv"), index=False)
    compare_units(units, ["excess"], pairs, by=("trait", "neighbourhood")).to_csv(
        os.path.join(out, "gistar_tests.csv"), index=False)
    print(f"  Gi*: {len(img)} image x trait x neighbourhood results")


def run_kde(stomata, images, cfg, pairs, out):
    kc = cfg["spatial"]["kde"]
    model = point_model(stomata)
    s = stomata[stomata.model == model]
    rows, maps = [], {}
    for _, im in images.iterrows():
        d = s[(s.group == im.group) & (s.unit == im.unit) & (s.image == im.image)]
        if len(d) < 3:
            continue
        xy = np.vstack([d.cx_px.to_numpy() / im.width_px, d.cy_px.to_numpy() / im.height_px])
        z = kde.kde_map(xy, kc["bandwidth"], kc["grid_size"])
        maps.setdefault(im.group, []).append(z)
        rows.append(dict(group=im.group, unit=im.unit, image=im.image, n=len(d),
                         high_density_extent=kde.extent(z, kc["threshold"])))
    img = pd.DataFrame(rows)
    img.to_csv(os.path.join(out, "kde_images.csv"), index=False)
    units = img.groupby(["group", "unit"]).high_density_extent.mean().reset_index()
    units.to_csv(os.path.join(out, "kde_units.csv"), index=False)
    compare_units(units, ["high_density_extent"], pairs).to_csv(os.path.join(out, "kde_tests.csv"), index=False)

    mean_maps = {g: np.mean(z, axis=0) for g, z in maps.items()}
    std_maps = {g: np.std(z, axis=0) for g, z in maps.items()}
    smin = min(m.min() for m in std_maps.values())
    smax = max(m.max() for m in std_maps.values())
    std_maps = {g: (m - smin) / (smax - smin) for g, m in std_maps.items()}
    pd.DataFrame([dict(group=g, n_images=len(maps[g]),
                       high_variability_extent=kde.extent(std_maps[g], kc["threshold"])) for g in sorted(maps)]
                 ).to_csv(os.path.join(out, "kde_groups.csv"), index=False)
    if kc["save_maps"]:
        mdir = os.path.join(out, "kde_maps")
        os.makedirs(mdir, exist_ok=True)
        for g in sorted(maps):
            aspect = images[images.group == g].height_px.iloc[0] / images[images.group == g].width_px.iloc[0]
            for name, m, cmap in (("mean", mean_maps[g], "viridis"), ("std", std_maps[g], "magma")):
                fig, ax = plt.subplots(figsize=(5, 5 * aspect + 0.6))
                h = ax.imshow(m.T, origin="upper", extent=(0, 1, 1, 0), cmap=cmap, vmin=0, vmax=1, aspect=aspect)
                ax.set_title(f"{g}: {name} KDE (n = {len(maps[g])} images)")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(os.path.join(mdir, f"{g}_{name}.png"), dpi=200)
                plt.close(fig)
                np.save(os.path.join(mdir, f"{g}_{name}.npy"), m)
    print(f"  KDE: {len(img)} images")


def main(config_path):
    cfg = cfgmod.load(config_path)
    src = cfg["output_folder"]
    out = os.path.join(src, "spatial")
    os.makedirs(out, exist_ok=True)
    stomata = pd.read_csv(os.path.join(src, "stomata.csv"), dtype={"group": str, "unit": str, "image": str})
    images = pd.read_csv(os.path.join(src, "images.csv"), dtype={"group": str, "unit": str, "image": str})
    ripley.MIN_POINTS = ac.MIN_POINTS = cfg["spatial"]["min_stomata_per_image"]
    pairs = group_pairs(sorted(images.group.unique()), cfg["spatial"]["compare_groups"])
    seed = cfg["spatial"]["seed"]
    sc = cfg["spatial"]
    if sc["ripley"]["run"]:
        print("Ripley's K / L ...")
        run_ripley(stomata, images, cfg, pairs, out, seed)
    if sc["moran"]["run"]:
        print("Moran's I ...")
        run_moran(stomata, cfg, pairs, out, seed)
    if sc["gistar"]["run"]:
        print("Getis-Ord Gi* ...")
        run_gistar(stomata, cfg, pairs, out, seed)
    if sc["kde"]["run"]:
        print("KDE ...")
        run_kde(stomata, images, cfg, pairs, out)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python spatial.py config.json")
    main(sys.argv[1])
