"""
phasecorr_refine.py  --  STAGE 2 local phase-correlation refinement
-------------------------------------------------------------------

Runs AFTER bowtie_pipeline.py. Stage 1 removes the bulk bow-tie / panoramic
distortion with tie-point + TPS warping, producing `avhrr_bowtie_corrected.tif`.
Whatever residual misregistration remains is now SMALL and locally close to a
pure translation -- exactly the regime where phase correlation works (on the RAW
image the bow-tie was too large/stretched and phase corr locked onto spurious
peaks; confirmed in earlier sessions).

So this stage:
  1. Loads the Stage-1 corrected AVHRR (NOT the raw) + MODIS + masks.
  2. Warps the AVHRR cloud/valid masks into corrected space (using the Stage-1
     shift field) so gating is aligned with the corrected content.
  3. Builds cross-sensor GRADIENT-STRUCTURE images (raw intensity is unreliable
     across AVHRR<->MODIS) of the corrected AVHRR and MODIS.
  4. Sweeps a dense grid of windows; per window runs MASKED
     skimage.registration.phase_cross_correlation (sub-pixel) to estimate the
     residual translation. Hard gates: enough clear+valid overlap, enough
     texture, and the residual magnitude is CAPPED (Stage 1 already removed the
     bulk -- a large residual is a wrong lock, so reject it).
  5. Passes every surviving residual through the pipeline's real-image NCC gate
     (`ncc_valid_mask`) so only residuals that actually improve alignment on the
     visible band survive.
  6. Fits a residual TPS field x trust-mask and warps the Stage-1 output again
     -> `avhrr_bowtie_corrected_stage2.tif` (Stage 1 is preserved for compare).
  7. Writes a Stage-1-vs-Stage-2 cloud-masked NCC report (overall + 3x3 tiles).

Where it helps most: TEXTURED interior regions that had no coastline for Stage 1
(e.g. Gangetic farmland/rivers), plus sub-pixel cleanup on already-corrected
coasts. Where it can't help: cloud / featureless regions (no signal) -- those
stay at their Stage-1 (protected/original) geolocation via the trust mask.

Run with the `geo` conda env:
    conda run -n geo python bowtie_coreg/phasecorr_refine.py \
        --inputs bowtie_coreg/inputs --output bowtie_coreg/output
"""

import os
import argparse
import numpy as np
import cv2
from osgeo import gdal
from skimage.registration import phase_cross_correlation

# Stage 1 lives in a sibling folder; make its shared functions importable.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "stage1_ncc"))
import bowtie_pipeline as bp

gdal.UseExceptions()

_BASE = os.path.dirname(os.path.abspath(__file__))

# ---- Stage-2 window / gating parameters -----------------------------------
WIN = 112               # phase-corr window size (px). Small enough that the
                        # in-window residual is ~a pure translation.
STEP = 64               # dense sweep step (px)
MAX_RESIDUAL = 15.0     # cap |residual| (px). Stage 1 removed the bulk; a bigger
                        # residual is a spurious phase-corr lock -> reject.
MIN_VALID_FRAC = 0.70   # window must be this on-swath (both sensors)
MIN_CLEAR_FRAC = 0.70   # window must be this clear of AVHRR cloud
MIN_STRUCT_STD = 3.0    # min AVHRR gradient-structure texture (else no signal)
UPSAMPLE = 10           # phase-corr sub-pixel upsampling factor
OVERLAP_RATIO = 0.3     # masked phase-corr min overlap

MEDIAN_K = 6            # local-median consistency neighbours
MEDIAN_THRESH = 8.0     # residual must agree with neighbour median within this px


def _read_tif(path):
    d = gdal.Open(path)
    a = d.GetRasterBand(1).ReadAsArray().astype(np.float64)
    d = None
    return a


def warp_mask(mask, dxf, dyf):
    """Move a boolean AVHRR-space mask into Stage-1 corrected space using the
    Stage-1 shift field (same remap the imagery underwent)."""
    w = bp.warp_with_field(mask.astype(np.float32), dxf, dyf)
    return w > 0.5


def phasecorr_residuals(a_st, m_st, valid_c, clear_c, m_valid, verbose=True):
    """Dense masked phase-correlation sweep on the gradient-structure images.
    Returns tie points (window centres) and their residual (dx, dy)."""
    H, W = a_st.shape
    half = WIN // 2
    ta, off = [], []
    n_try = n_keep = 0
    for r in range(half, H - half, STEP):
        for c in range(half, W - half, STEP):
            r0, c0 = r - half, c + 0 - half
            av = valid_c[r0:r0 + WIN, c0:c0 + WIN]
            mv = m_valid[r0:r0 + WIN, c0:c0 + WIN]
            cl = clear_c[r0:r0 + WIN, c0:c0 + WIN]
            if av.mean() < MIN_VALID_FRAC or mv.mean() < MIN_VALID_FRAC:
                continue
            if cl.mean() < MIN_CLEAR_FRAC:
                continue
            a_win = a_st[r0:r0 + WIN, c0:c0 + WIN]
            if a_win.std() < MIN_STRUCT_STD:
                continue                       # too little texture: no signal
            m_win = m_st[r0:r0 + WIN, c0:c0 + WIN]
            ref_mask = mv & (m_win != 0)
            mov_mask = av & cl
            if ref_mask.mean() < MIN_VALID_FRAC or mov_mask.mean() < MIN_VALID_FRAC:
                continue
            n_try += 1
            try:
                out = phase_cross_correlation(
                    m_win, a_win,
                    reference_mask=ref_mask, moving_mask=mov_mask,
                    upsample_factor=UPSAMPLE, overlap_ratio=OVERLAP_RATIO)
            except Exception:
                continue
            shift = np.asarray(out[0], dtype=float)   # [drow, dcol], moving->reference
            drow, dcol = float(shift[0]), float(shift[1])
            if not (np.isfinite(drow) and np.isfinite(dcol)):
                continue
            if np.hypot(drow, dcol) > MAX_RESIDUAL:
                continue                       # large residual = wrong lock
            ta.append((c, r))
            off.append((dcol, drow))           # off = (dx, dy)
            n_keep += 1
    if verbose:
        print(f"Phase-corr sweep: {n_keep} residuals kept of {n_try} textured windows")
    return np.array(ta, dtype=float), np.array(off, dtype=float)


def gradient_ncc_report(cor, stage2, m_arr, clear_c, valid_c, m_valid, path=None):
    """Cloud-masked gradient NCC vs MODIS, Stage-1 vs Stage-2, overall + 3x3."""
    from scipy.ndimage import sobel

    def grad(a):
        a = np.nan_to_num(a.astype(np.float64))
        return np.hypot(sobel(a, 0), sobel(a, 1))

    def ncc(x, y, v):
        x, y = x[v], y[v]
        if x.size < 50:
            return np.nan
        x = x - x.mean(); y = y - y.mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / d) if d > 0 else np.nan

    gm = grad(m_arr)
    base_valid = m_valid & clear_c
    H, W = m_arr.shape
    lines = []

    def block(a, label):
        g = grad(a)
        onsw = a > 0
        rr = np.linspace(0, H, 4).astype(int)
        cc = np.linspace(0, W, 4).astype(int)
        lines.append(f"--- {label} ---")
        vals = []
        for i in range(3):
            row = []
            for j in range(3):
                v = base_valid[rr[i]:rr[i+1], cc[j]:cc[j+1]] & onsw[rr[i]:rr[i+1], cc[j]:cc[j+1]]
                n = ncc(gm[rr[i]:rr[i+1], cc[j]:cc[j+1]], g[rr[i]:rr[i+1], cc[j]:cc[j+1]], v)
                row.append(n)
                if np.isfinite(n):
                    vals.append(n)
            lines.append("  " + "  ".join(f"{x:+.3f}" if np.isfinite(x) else " nan " for x in row))
        ov = ncc(gm, g, base_valid & onsw)
        lines.append(f"  overall {ov:+.4f}   mean-tile {np.mean(vals):+.4f}")
        return ov, np.mean(vals)

    o1, t1 = block(cor, "STAGE 1 (avhrr_bowtie_corrected.tif)")
    o2, t2 = block(stage2, "STAGE 2 (avhrr_bowtie_corrected_stage2.tif)")
    lines.append("")
    lines.append(f"DELTA  overall {o2 - o1:+.4f}   mean-tile {t2 - t1:+.4f}")
    text = "\n".join(lines)
    print(text)
    if path:
        with open(path, "w") as f:
            f.write(text + "\n")
    return o1, o2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default=os.path.join(_BASE, "inputs"))
    p.add_argument("--output", default=os.path.join(_BASE, "output"))
    args = p.parse_args()

    bp.INPUTS_DIR = args.inputs
    bp.OUTPUT_DIR = args.output
    arrays, geotransform, projection = bp.load_arrays()
    m_arr = arrays["m_arr"]
    H, W = m_arr.shape

    cor_path = os.path.join(args.output, "avhrr_bowtie_corrected.tif")
    if not os.path.exists(cor_path):
        raise SystemExit(f"Stage-1 output not found: {cor_path}\n"
                         f"Run bowtie_pipeline.py first.")
    cor = _read_tif(cor_path)

    # Stage-1 shift field -> move AVHRR-space masks into corrected space
    s1_dxf = _read_tif(os.path.join(args.output, "shift_field_dx.tif")).astype(np.float32)
    s1_dyf = _read_tif(os.path.join(args.output, "shift_field_dy.tif")).astype(np.float32)
    clear = ~arrays["cloud_mask"]
    s_valid = arrays["s_arr"] > 0
    clear_c = warp_mask(clear, s1_dxf, s1_dyf) & (cor > 0)
    valid_c = (cor > 0)
    cloud_c = warp_mask(arrays["cloud_mask"], s1_dxf, s1_dyf)
    m_valid = np.isfinite(m_arr) & (m_arr > 0)

    print("\n" + "=" * 60)
    print("STAGE 2: LOCAL PHASE-CORRELATION RESIDUAL REFINEMENT")
    print("=" * 60)

    # cross-sensor gradient-structure images (corrected AVHRR vs MODIS)
    a_st = bp.gradient_structure(cor, valid_c)
    m_st = bp.gradient_structure(m_arr, m_valid)

    ta, off = phasecorr_residuals(a_st, m_st, valid_c, clear_c, m_valid)

    if len(ta) >= MEDIAN_K + 1:
        keep = bp.median_consistency_mask(ta, off, k=MEDIAN_K, thresh=MEDIAN_THRESH)
        print(f"Local-median consistency: {int(keep.sum())}/{len(ta)} residuals kept")
        ta, off = ta[keep], off[keep]

    # real-image NCC gate on the visible band (corrected AVHRR vs MODIS)
    if len(ta):
        cor32, m32 = cor.astype(np.float32), m_arr.astype(np.float32)
        keep = bp.ncc_valid_mask(ta, off, cor32, m32, clear=clear_c)
        print(f"NCC validity filter: {int(keep.sum())}/{len(ta)} residuals improve alignment")
        ta, off = ta[keep], off[keep]

    print(f"Surviving residuals: {len(ta)}")
    if len(ta):
        print(f"  |residual| mean={np.hypot(off[:,0],off[:,1]).mean():.2f}px "
              f"max={np.hypot(off[:,0],off[:,1]).max():.2f}px  "
              f"dx[{off[:,0].min():+.1f},{off[:,0].max():+.1f}] "
              f"dy[{off[:,1].min():+.1f},{off[:,1].max():+.1f}]")

    if len(ta) < 6:
        print("Too few residuals to fit a Stage-2 field; Stage 1 kept as final.")
        # still emit a stage-2 file == stage-1 so downstream is uniform
        bp.save_geotiff(cor, geotransform, projection,
                        os.path.join(args.output, "avhrr_bowtie_corrected_stage2.tif"))
        return

    # residual TPS field x corrected-space trust mask
    dxf, dyf = bp.fit_tps_field(ta, off, W, H)
    if bp.AUTO_PROTECT:
        trust = bp.build_trust_mask(ta, {"s_arr": cor, "cloud_mask": cloud_c})
        dxf, dyf = dxf * trust, dyf * trust
    stage2 = bp.warp_with_field(cor, dxf, dyf)

    bp.save_geotiff(dxf, geotransform, projection,
                    os.path.join(args.output, "stage2_residual_dx.tif"))
    bp.save_geotiff(dyf, geotransform, projection,
                    os.path.join(args.output, "stage2_residual_dy.tif"))
    bp.save_geotiff(stage2, geotransform, projection,
                    os.path.join(args.output, "avhrr_bowtie_corrected_stage2.tif"))
    np.save(os.path.join(args.output, "stage2_ta.npy"), ta)
    np.save(os.path.join(args.output, "stage2_off.npy"), off)
    print(f"Wrote {args.output}/avhrr_bowtie_corrected_stage2.tif")

    print("\n" + "=" * 60)
    print("VALIDATION: STAGE 1 vs STAGE 2 (cloud-masked gradient NCC vs MODIS)")
    print("=" * 60)
    report_dir = os.path.join(args.output, "report")
    os.makedirs(report_dir, exist_ok=True)
    gradient_ncc_report(cor, stage2, m_arr, clear_c, valid_c, m_valid,
                        path=os.path.join(report_dir, "ncc_validation_stage2.txt"))


if __name__ == "__main__":
    main()
