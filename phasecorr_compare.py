"""
phasecorr_compare.py
--------------------

Validation / success metric for phasecorr_pipeline.py.

Success (agreed with the user) = the corrected AVHRR overlays MODIS better
than the original, measured quantitatively over independent checkpoint
windows, especially where the large error lives. This script:

  1. NCC + SSIM over a grid of independent checkpoint windows on land,
     for baseline (original AVHRR) and phase-corr corrected, vs MODIS.
  2. Per-row-band breakdown (did the top/bottom edges improve?).
  3. A tie-point quality table (ours vs the fixed AROSICS run) - the
     SSIM-improved / NCC-improved fraction that was 0 in the user's old run.
  4. Visual checkerboard + full-scene, baseline vs corrected.

Run with the `geo` conda env, AFTER phasecorr_pipeline.py:
    conda run -n geo python phasecorr_compare.py
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

gdal.UseExceptions()

OUTPUT_DIR = "phasecorr_output"
AVHRR_COMMON = os.path.join(OUTPUT_DIR, "avhrr_common.tif")
MODIS_COMMON = os.path.join(OUTPUT_DIR, "modis_common.tif")
CORRECTED = os.path.join(OUTPUT_DIR, "avhrr_phasecorr_corrected.tif")
VALID_MASK = os.path.join(OUTPUT_DIR, "valid_mask.tif")

CHECK_WINDOW = 128     # checkpoint window size (px)
CHECK_STRIDE = 128     # checkpoint spacing (px)
MIN_VALID = 0.6        # require this much valid data in a checkpoint window


def read(path):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None
    return arr


def ncc(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < a.size * 0.3:
        return np.nan
    av = a[v] - a[v].mean()
    bv = b[v] - b[v].mean()
    d = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av * bv).sum() / d) if d else np.nan


def ssim_safe(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.mean() < MIN_VALID:
        return np.nan
    a2 = np.where(np.isfinite(a), a, 0.0)
    b2 = np.where(np.isfinite(b), b, 0.0)
    rng = float(np.nanmax([a2.max(), b2.max()]) - np.nanmin([a2.min(), b2.min()]))
    if rng <= 0:
        return np.nan
    try:
        return float(structural_similarity(a2, b2, data_range=rng))
    except Exception:
        return np.nan


def checkpoint_metrics(modis, avhrr, corrected, valid):
    ysize, xsize = modis.shape
    half = CHECK_WINDOW // 2
    rows = []
    for cy in range(half, ysize - half, CHECK_STRIDE):
        for cx in range(half, xsize - half, CHECK_STRIDE):
            r0, r1 = cy - half, cy + half
            c0, c1 = cx - half, cx + half
            vw = valid[r0:r1, c0:c1]
            if vw.mean() < MIN_VALID:
                continue
            m = modis[r0:r1, c0:c1]
            a = avhrr[r0:r1, c0:c1]
            c = corrected[r0:r1, c0:c1]
            rows.append({
                "Y_IM": cy, "X_IM": cx,
                "ncc_base": ncc(a, m), "ncc_corr": ncc(c, m),
                "ssim_base": ssim_safe(a, m), "ssim_corr": ssim_safe(c, m),
            })
    return pd.DataFrame(rows)


def normalize(arr):
    lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def checkerboard(a, b, tile=40):
    h, w = a.shape
    out = a.copy()
    ys, xs = np.indices((h, w))
    mask = ((ys // tile) + (xs // tile)) % 2 == 0
    out[mask] = b[mask]
    return out


def main():
    modis = read(MODIS_COMMON)
    avhrr = read(AVHRR_COMMON)
    corrected = read(CORRECTED)
    valid = read(VALID_MASK).astype(bool)

    print("=" * 60)
    print("CHECKPOINT-WINDOW METRICS (independent of tie points)")
    print("=" * 60)
    df = checkpoint_metrics(modis, avhrr, corrected, valid)
    print(f"Checkpoint windows evaluated: {len(df)}")

    for metric in ["ncc", "ssim"]:
        base = df[f"{metric}_base"].mean()
        corr = df[f"{metric}_corr"].mean()
        improved = (df[f"{metric}_corr"] > df[f"{metric}_base"]).mean()
        print(f"\n{metric.upper()}:  baseline={base:.4f}  corrected={corr:.4f}  "
              f"delta={corr - base:+.4f}  ({improved*100:.0f}% of windows improved)")

    print("\nPer-row-band NCC (top -> bottom):")
    df["band"] = pd.cut(df["Y_IM"], bins=5)
    band = df.groupby("band", observed=True).agg(
        n=("ncc_base", "size"),
        ncc_base=("ncc_base", "mean"),
        ncc_corr=("ncc_corr", "mean"),
    )
    band["delta"] = band["ncc_corr"] - band["ncc_base"]
    print(band.round(4).to_string())

    df.to_csv(os.path.join(OUTPUT_DIR, "checkpoint_metrics.csv"), index=False)

    print("\n" + "=" * 60)
    print("TIE-POINT QUALITY: phase-corr vs fixed AROSICS")
    print("=" * 60)
    ours = pd.read_csv(os.path.join(OUTPUT_DIR, "tie_points_verified.csv"))
    real = ours[ours["shift_px"] >= 1.0]   # exclude near-zero anchors from the NCC summary
    print(f"Phase-corr: {len(ours)} NCC-verified tie points "
          f"({len(real)} with real shift >=1px), "
          f"max shift {ours['shift_px'].max():.1f}px "
          f"(~{ours['shift_px'].max()*1.11:.0f}km)")
    print(f"  Real-shift points: mean NCC before={np.nanmean(real['ncc_before']):.3f} "
          f"after={np.nanmean(real['ncc_after']):.3f} "
          f"(median delta={np.nanmedian(real['ncc_after']-real['ncc_before']):+.3f})")
    aros_csv = os.path.join(OUTPUT_DIR, "arosics_tie_points.csv")
    if os.path.exists(aros_csv):
        at = pd.read_csv(aros_csv)
        matched = at[at["ABS_SHIFT"] != -9999]
        ssim_impr = (matched["SSIM_IMPROVED"] == True).sum() if "SSIM_IMPROVED" in matched else 0  # noqa: E712
        print(f"AROSICS (same inputs): {len(matched)} matched, "
              f"SSIM_IMPROVED for {ssim_impr} of them "
              f"(the user's original run: 0)")

    print("\n" + "=" * 60)
    print("VISUAL CHECKERBOARD")
    print("=" * 60)
    m_n = normalize(modis)
    a_n = normalize(avhrr)
    c_n = normalize(corrected)

    fig, axes = plt.subplots(1, 2, figsize=(18, 13))
    axes[0].imshow(checkerboard(m_n, a_n), cmap="gray")
    axes[0].set_title("MODIS vs ORIGINAL AVHRR")
    axes[1].imshow(checkerboard(m_n, c_n), cmap="gray")
    axes[1].set_title("MODIS vs CORRECTED AVHRR")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Checkerboard overlay (aligned = continuous edges across tiles)")
    out = os.path.join(OUTPUT_DIR, "compare_checkerboard.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 12))
    for ax, img, title in zip(
        axes, [m_n, a_n, c_n],
        ["MODIS (reference)", "AVHRR original", "AVHRR corrected"]):
        ax.imshow(img, cmap="gray"); ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(OUTPUT_DIR, "compare_full_scene.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Zoomed crop at the highest-improvement checkpoint window, so a ~110px
    # shift is actually visible (the whole-scene view is too small to show it).
    best = df.loc[(df["ncc_corr"] - df["ncc_base"]).idxmax()]
    cy, cx = int(best.Y_IM), int(best.X_IM)
    h = 200
    r0, r1, c0, c1 = max(cy - h, 0), cy + h, max(cx - h, 0), cx + h
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, img, title in zip(
        axes, [m_n, a_n, c_n],
        [f"MODIS @({cx},{cy})", "AVHRR original", "AVHRR corrected"]):
        ax.imshow(img[r0:r1, c0:c1], cmap="gray"); ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Zoomed crop at best-improvement window "
                 f"(NCC {best.ncc_base:.2f} -> {best.ncc_corr:.2f})")
    out = os.path.join(OUTPUT_DIR, "compare_zoomed.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
