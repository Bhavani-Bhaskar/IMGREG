"""
visualize_overlay.py
---------------------

Visually verify a shifted/corrected AVHRR by overlapping it on the MODIS
reference - the same style as the "expected outcome" image (the AVHRR swath
sitting on top of the MODIS basemap, narrowing at top & bottom).

Produces, for both the ORIGINAL and the CORRECTED AVHRR:
  1. composite  - MODIS basemap everywhere, AVHRR shown opaque inside its swath
                  footprint (matches the expected-outcome look).
  2. blend      - MODIS basemap, AVHRR alpha-blended (see through to MODIS to
                  judge alignment).
  3. redcyan    - AVHRR in red, MODIS in cyan; aligned -> gray, misaligned ->
                  colour fringes.
and a side-by-side original-vs-corrected comparison so the improvement is
obvious.

Inputs default to the bow-tie pipeline result on the common grid, but every
path is a top-level constant / CLI arg so you can point it at any corrected
GeoTIFF (e.g. the manual expected output) and any MODIS/AVHRR pair.

Run with the `geo` conda env:
    conda run -n geo python visualize_overlay.py
    conda run -n geo python visualize_overlay.py --corrected some_other.tif
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal

gdal.UseExceptions()

# ---- inputs (override via CLI) ----
_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
MANUAL_DIR = os.path.join(_BASE, "inputs")   # inputs built by preprocessing.py
ORIGINAL_AVHRR_NPY = os.path.join(MANUAL_DIR, "a_arr.npy")     # AVHRR visible, common grid
MODIS_NPY = os.path.join(MANUAL_DIR, "m_arr.npy")             # MODIS, same grid
CORRECTED_TIF = os.path.join(_BASE, "output", "avhrr_bowtie_corrected.tif")  # shifted AVHRR

OUT_DIR = os.path.join(_BASE, "output", "report")
BLEND_ALPHA = 0.6            # AVHRR opacity in the blend view
DOWNSAMPLE = 2              # display downsample for speed (1 = full res)


def load_raster(path):
    """Load a GeoTIFF band or a .npy array as float32."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None
    return arr


def norm(a, valid=None):
    """Percentile stretch to [0,1] over valid pixels."""
    a = a.astype(np.float32)
    v = np.isfinite(a) & (a != 0) if valid is None else valid
    out = np.zeros_like(a)
    if v.sum() < 10:
        return out
    lo, hi = np.nanpercentile(a[v], 2), np.nanpercentile(a[v], 98)
    out = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    out[~np.isfinite(out)] = 0
    return out


def composite(modis_n, avhrr_n, avhrr_valid):
    """MODIS basemap, AVHRR opaque inside its swath footprint."""
    out = modis_n.copy()
    out[avhrr_valid] = avhrr_n[avhrr_valid]
    return out


def blend(modis_n, avhrr_n, avhrr_valid, alpha):
    out = modis_n.copy()
    out[avhrr_valid] = alpha * avhrr_n[avhrr_valid] + (1 - alpha) * modis_n[avhrr_valid]
    return out


def redcyan(avhrr_n, modis_n):
    """AVHRR -> red, MODIS -> cyan. Aligned = gray, misaligned = colour fringe."""
    return np.dstack([avhrr_n, modis_n, modis_n])


def save_gray(img, title, path):
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def save_rgb(img, title, path):
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(img, origin="upper")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", default=ORIGINAL_AVHRR_NPY)
    p.add_argument("--modis", default=MODIS_NPY)
    p.add_argument("--corrected", default=CORRECTED_TIF)
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args()

    if not os.path.exists(args.corrected):
        raise SystemExit(f"Corrected AVHRR not found: {args.corrected}\n"
                         f"Run bowtie_pipeline.py first, or pass --corrected <file>.")

    os.makedirs(args.out, exist_ok=True)

    modis = load_raster(args.modis)
    original = load_raster(args.original)
    corrected = load_raster(args.corrected)

    d = DOWNSAMPLE
    modis, original, corrected = modis[::d, ::d], original[::d, ::d], corrected[::d, ::d]

    modis_n = norm(modis)
    orig_n = norm(original)
    corr_n = norm(corrected)
    orig_valid = np.isfinite(original) & (original != 0)
    corr_valid = np.isfinite(corrected) & (corrected != 0)

    # --- composite (expected-outcome style) ---
    comp_orig = composite(modis_n, orig_n, orig_valid)
    comp_corr = composite(modis_n, corr_n, corr_valid)
    save_gray(comp_orig, "ORIGINAL AVHRR on MODIS", os.path.join(args.out, "overlay_original_on_modis.png"))
    save_gray(comp_corr, "SHIFTED (corrected) AVHRR on MODIS", os.path.join(args.out, "overlay_shifted_on_modis.png"))

    # --- side-by-side comparison ---
    h, w = comp_orig.shape
    fig, axes = plt.subplots(1, 2, figsize=(20, 20 * h / w / 2))
    axes[0].imshow(comp_orig, cmap="gray"); axes[0].set_title("BEFORE: original AVHRR on MODIS")
    axes[1].imshow(comp_corr, cmap="gray"); axes[1].set_title("AFTER: shifted AVHRR on MODIS")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(os.path.join(args.out, "overlay_compare.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved", os.path.join(args.out, "overlay_compare.png"))

    # --- alpha blend (see-through, best for judging alignment) ---
    save_gray(blend(modis_n, corr_n, corr_valid, BLEND_ALPHA),
              f"SHIFTED AVHRR blended on MODIS (alpha={BLEND_ALPHA})",
              os.path.join(args.out, "overlay_shifted_blend.png"))

    # --- red/cyan ---
    save_rgb(redcyan(orig_n, modis_n), "BEFORE red/cyan (AVHRR red, MODIS cyan)",
             os.path.join(args.out, "overlay_redcyan_before.png"))
    save_rgb(redcyan(corr_n, modis_n), "AFTER red/cyan (gray = aligned)",
             os.path.join(args.out, "overlay_redcyan_after.png"))

    print(f"\nDone. Overlays in {args.out}/")


if __name__ == "__main__":
    main()
