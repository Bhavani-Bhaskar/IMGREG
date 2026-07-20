"""
panoramic_model.py  --  PROTOTYPE parametric panoramic distortion model
------------------------------------------------------------------------

Motivation. The Stage-1 tie-point + TPS pipeline (bowtie_pipeline.py) and the
Stage-2 phase-corr refinement (phasecorr_refine.py) only correct where there are
matchable features. Feature-less regions (heavy cloud e.g. Gujarat, textureless
Gangetic plain) get NO correction -- the trust mask pins them to their original,
up-to-150px-wrong geolocation. This module fills that gap with a PHYSICAL model.

Idea. The geolocation error is a smooth PANORAMIC / bow-tie distortion: a function
of the AVHRR SCAN GEOMETRY (across-track scan angle + along-track scan line), not
of arbitrary map position. We recover per-pixel scan coordinates from the raw
granule's embedded GCP navigation (a regular 51x119 sample/line grid), express the
tie-point shifts in that physical (s = nadir-relative across-track, t = along-track)
frame, and fit a smooth robust polynomial dx,dy = F(s,t). Because s,t are the true
distortion axes, the model EXTRAPOLATES a physically-plausible correction into
feature-less regions along scan lines -- something a free TPS cannot do.

Empirically established limits (see the session notes / memory):
  * The physical structure is real: dx crosses ~0 at nadir (sample~1024) and grows
    with opposite sign toward each swath edge.
  * BUT a smooth global model has a ~20-30px representational floor even when fit
    directly to the manual ground truth -- the true correction has local structure
    a parametric model can't capture. So it does NOT beat TPS where features exist.
  * Therefore this is used as a HYBRID, not a replacement:
        final = parametric_bulk(everywhere) + TPS_residual * trust_mask
    Where trusted (features present) the residual completes the fit exactly, so
    covered regions match the Stage-1 result. Where untrusted (cloud/featureless)
    only the physical bulk applies -- a bounded ~20-30px-residual estimate instead
    of leaving 100-150px uncorrected. The feature-less correction is a physical
    EXTRAPOLATION and cannot be directly validated (no ground truth there).

Run with the `geo` conda env (needs the raw granule for the GCP nav):
    conda run -n geo python bowtie_coreg/panoramic_model.py \
        --granule hrpt_M03_20250506_0420_33701 \
        --inputs bowtie_coreg/inputs --output bowtie_coreg/output
"""

import os
import argparse
import numpy as np
import cv2
from osgeo import gdal
from scipy.interpolate import griddata, RBFInterpolator

# Stage 1 lives in a sibling folder; make its shared functions importable.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "stage1_ncc"))
import bowtie_pipeline as bp

gdal.UseExceptions()

_BASE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """Walk up to the project root (dir holding Data/psdd_metop), nesting-agnostic."""
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "Data", "psdd_metop")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(_BASE)


_ROOT = _find_root(_BASE)
RAW = os.path.join(_ROOT, "Data", "psdd_metop", "metop")

MODEL_ORDER = 3        # polynomial order of the scan-geometry model (LOO-chosen)
GEOM_STEP = 8          # coarse-grid step for the scan-geometry interpolation (speed)
CLIP_MARGIN = 25       # clip model output this far beyond the observed shift range


# ------------------------------------------------------------------
# Scan-geometry coordinate maps from the raw granule's GCP navigation
# ------------------------------------------------------------------

def build_scan_geometry(granule, geotransform, shape):
    """Per-grid-pixel across-track sample & along-track line, from the raw
    granule GCPs (a regular sample/line node grid). Returns (SAMP, LINE, n_samp)."""
    H, W = shape
    left, res, _, top, _, nres = geotransform
    src = gdal.Open(os.path.join(RAW, granule + "_b2.tif"))
    gcps = src.GetGCPs()
    src = None
    if not gcps:
        raise SystemExit(f"No GCPs on {granule}_b2.tif - cannot build scan geometry.")
    gsamp = np.array([g.GCPPixel for g in gcps])
    gline = np.array([g.GCPLine for g in gcps])
    glon = np.array([g.GCPX for g in gcps])
    glat = np.array([g.GCPY for g in gcps])
    gcol = (glon - left) / res
    grow = (top - glat) / (-nres)
    pts = np.column_stack([gcol, grow])

    YY, XX = np.mgrid[0:H:GEOM_STEP, 0:W:GEOM_STEP]
    q = np.column_stack([XX.ravel(), YY.ravel()])
    samp = griddata(pts, gsamp, q, method="linear")
    line = griddata(pts, gline, q, method="linear")
    nanmask = np.isnan(samp)
    if nanmask.any():                      # nearest-fill outside the GCP hull
        samp[nanmask] = griddata(pts, gsamp, q[nanmask], method="nearest")
        line[nanmask] = griddata(pts, gline, q[nanmask], method="nearest")
    SAMP = cv2.resize(samp.reshape(XX.shape).astype(np.float32), (W, H), cv2.INTER_LINEAR)
    LINE = cv2.resize(line.reshape(XX.shape).astype(np.float32), (W, H), cv2.INTER_LINEAR)
    return SAMP, LINE, int(gsamp.max()) + 1


def _norm(SAMP, LINE, n_samp):
    nadir = (n_samp - 1) / 2.0
    lmin, lmax = float(LINE.min()), float(LINE.max())
    sn = (SAMP - nadir) / nadir                             # ~[-1,+1], 0 at nadir
    tn = (LINE - lmin) / max(lmax - lmin, 1) * 2.0 - 1.0    # ~[-1,+1] along-track
    return sn, tn


# ------------------------------------------------------------------
# Robust parametric fit  dx,dy = poly(sn, tn)
# ------------------------------------------------------------------

def _design(sn, tn, order):
    cols = [(sn ** i) * (tn ** j)
            for i in range(order + 1) for j in range(order + 1 - i)]
    return np.column_stack(cols)


def _fit_irls(A, y, iters=8):
    """Huber iteratively-reweighted least squares (robust to bad tie points)."""
    w = np.ones(len(y))
    c = np.zeros(A.shape[1])
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        r = y - A @ c
        sca = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        d = 6.0 * sca
        w = np.where(np.abs(r) <= d, 1.0, d / np.maximum(np.abs(r), 1e-6))
    return c


def fit_panoramic(ta, off, SAMP, LINE, n_samp, order=MODEL_ORDER):
    """Fit the scan-geometry polynomial; return a predictor over grid coords."""
    sn, tn = _norm(SAMP, LINE, n_samp)
    ti = ta.astype(int)
    s = sn[ti[:, 1], ti[:, 0]]
    t = tn[ti[:, 1], ti[:, 0]]
    A = _design(s, t, order)
    cx = _fit_irls(A, off[:, 0])
    cy = _fit_irls(A, off[:, 1])
    dxlo, dxhi = off[:, 0].min() - CLIP_MARGIN, off[:, 0].max() + CLIP_MARGIN
    dylo, dyhi = off[:, 1].min() - CLIP_MARGIN, off[:, 1].max() + CLIP_MARGIN

    def predict_grid():
        Ag = _design(sn.ravel(), tn.ravel(), order)
        dx = np.clip(Ag @ cx, dxlo, dxhi).reshape(sn.shape).astype(np.float32)
        dy = np.clip(Ag @ cy, dylo, dyhi).reshape(sn.shape).astype(np.float32)
        return dx, dy

    def predict_pts(pos):
        pi = pos.astype(int)
        Ap = _design(sn[pi[:, 1], pi[:, 0]], tn[pi[:, 1], pi[:, 0]], order)
        return np.clip(Ap @ cx, dxlo, dxhi), np.clip(Ap @ cy, dylo, dyhi)

    return predict_grid, predict_pts


# ------------------------------------------------------------------
# Hybrid field: parametric bulk everywhere + TPS residual where trusted
# ------------------------------------------------------------------

MODEL_DILATE = 220     # px; how far beyond the tie-point hull the model is trusted
MODEL_FEATHER = 60     # px; feather of the model-reliability mask


def build_model_reliability(ta, arrays):
    """Where the parametric model may be applied: on-swath AND within a DILATED
    convex hull of the tie points (mild extrapolation only), feathered to 0
    beyond. Unlike the TPS trust mask this does NOT gate on cloud -- the physical
    bulk is a function of scan geometry, so it is valid THROUGH cloud (the whole
    point). Far corners with no nearby tie points are excluded (their extrapolation
    is unreliable, e.g. the SE corner that over-corrected in the naive hybrid)."""
    from scipy.ndimage import gaussian_filter
    on_sw = arrays["a_arr"] > 0
    H, W = on_sw.shape
    inside = np.zeros((H, W), dtype=np.uint8)
    if len(ta) >= 4:
        from scipy.spatial import Delaunay
        try:
            hull = Delaunay(ta)
            step = 20
            ys, xs = np.mgrid[0:H:step, 0:W:step]
            ins = (hull.find_simplex(np.column_stack([xs.ravel(), ys.ravel()])) >= 0)
            inside = (cv2.resize(ins.reshape(xs.shape).astype(np.float32), (W, H),
                                 interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
        except Exception:
            pass
    k = 2 * MODEL_DILATE + 1
    inside = cv2.dilate(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    rel = gaussian_filter((inside > 0).astype(np.float32), MODEL_FEATHER)
    return rel * on_sw


def build_hybrid_field(tps_ta, tps_off, model_ta, model_off, arrays,
                       SAMP, LINE, n_samp, order=MODEL_ORDER):
    """3-zone blend:
      * trusted (features present)        -> pure Stage-1 TPS (covered quality kept)
      * not trusted but model-reliable    -> pure parametric model (fills the gap,
                                             incl. cloud/featureless near coverage)
      * beyond model reach (far corners)  -> 0 (original geolocation)

    The parametric MODEL is fit to `model_ta/off` (the curated ground truth when
    available, which reaches the true +/-145px edge shifts); the trusted-zone TPS
    is fit to `tps_ta/off` (the auto tie points, matching the validated Stage-1).
    """
    H, W = arrays["a_arr"].shape
    predict_grid, _ = fit_panoramic(model_ta, model_off, SAMP, LINE, n_samp, order)
    model_dx, model_dy = predict_grid()

    tps_dx, tps_dy = bp.fit_tps_field(tps_ta, tps_off, W, H)  # full Stage-1-style TPS
    trust = bp.build_trust_mask(tps_ta, arrays)               # TPS-reliable (covered, clear)
    mrel = build_model_reliability(model_ta, arrays)          # model-reliable (near coverage)

    w_model = np.clip(1.0 - trust, 0.0, 1.0) * mrel
    dxf = tps_dx * trust + model_dx * w_model
    dyf = tps_dy * trust + model_dy * w_model
    return dxf, dyf, model_dx, model_dy, trust


# ------------------------------------------------------------------
# Validation: cloud-masked gradient NCC vs MODIS
# ------------------------------------------------------------------

def ncc_report(fields, m_arr, base_valid, path=None):
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
    H, W = m_arr.shape
    rr = np.linspace(0, H, 4).astype(int)
    cc = np.linspace(0, W, 4).astype(int)
    lines = []
    for a, label in fields:
        g = grad(a); onsw = a > 0
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
        lines.append(f"  overall {ncc(gm, g, base_valid & onsw):+.4f}   mean-tile {np.mean(vals):+.4f}")
    text = "\n".join(lines)
    print(text)
    if path:
        with open(path, "w") as f:
            f.write(text + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--granule", default="hrpt_M03_20250506_0420_33701")
    p.add_argument("--inputs", default=os.path.join(_BASE, "inputs"))
    p.add_argument("--output", default=os.path.join(_BASE, "output"))
    p.add_argument("--order", type=int, default=MODEL_ORDER)
    args = p.parse_args()

    bp.INPUTS_DIR, bp.OUTPUT_DIR = args.inputs, args.output
    arrays, geotransform, projection = bp.load_arrays()
    m_arr = arrays["m_arr"]
    a_arr = arrays["a_arr"]

    ta = np.load(os.path.join(args.output, "auto_ta.npy"))
    off = np.load(os.path.join(args.output, "auto_off.npy"))
    print(f"Loaded {len(ta)} Stage-1 auto tie points (trusted-zone TPS)")

    # Anchor the parametric MODEL to the curated ground truth when it is present
    # (it reaches the true +/-145px edge shifts the auto points under-capture);
    # otherwise fall back to the auto points so the module still runs on any granule.
    gt_ta = os.path.join(args.inputs, "curated_ta.npy")
    gt_off = os.path.join(args.inputs, "curated_off.npy")
    if os.path.exists(gt_ta) and os.path.exists(gt_off):
        model_ta = np.load(gt_ta).astype(float)
        model_off = np.load(gt_off)
        model_src = f"curated ground truth ({len(model_ta)} pts)"
    else:
        model_ta, model_off = ta.astype(float), off
        model_src = f"auto tie points ({len(model_ta)} pts) - no curated GT for this granule"

    print("\n" + "=" * 60)
    print("PARAMETRIC PANORAMIC MODEL (scan-geometry, hybrid)")
    print("=" * 60)
    print(f"Model anchored to: {model_src}")
    print(f"  model |shift| range in source: dx[{model_off[:,0].min():+.0f},{model_off[:,0].max():+.0f}] "
          f"dy[{model_off[:,1].min():+.0f},{model_off[:,1].max():+.0f}]")
    SAMP, LINE, n_samp = build_scan_geometry(args.granule, geotransform, m_arr.shape)
    print(f"Scan geometry: {n_samp} samples/scan (nadir ~{(n_samp-1)/2:.0f}); "
          f"grid sample range {SAMP.min():.0f}..{SAMP.max():.0f}")

    dxf, dyf, model_dx, model_dy, trust = build_hybrid_field(
        ta.astype(float), off, model_ta, model_off, arrays,
        SAMP, LINE, n_samp, order=args.order)

    warped = bp.warp_with_field(a_arr, dxf, dyf)
    model_only = bp.warp_with_field(a_arr, model_dx, model_dy)

    bp.save_geotiff(warped, geotransform, projection,
                    os.path.join(args.output, "avhrr_bowtie_panoramic.tif"))
    bp.save_geotiff(model_dx, geotransform, projection,
                    os.path.join(args.output, "panoramic_model_dx.tif"))
    bp.save_geotiff(model_dy, geotransform, projection,
                    os.path.join(args.output, "panoramic_model_dy.tif"))
    print(f"Wrote {args.output}/avhrr_bowtie_panoramic.tif")
    print(f"  model bulk |shift|: mean={np.hypot(model_dx, model_dy)[a_arr>0].mean():.1f}px "
          f"max={np.hypot(model_dx, model_dy)[a_arr>0].max():.1f}px")

    # fraction of the on-swath scene the trust mask currently leaves UNCORRECTED
    onsw = a_arr > 0
    uncorrected = onsw & (trust < 0.1)
    print(f"  on-swath area the model NEWLY corrects (trust<0.1): "
          f"{100*uncorrected.sum()/onsw.sum():.1f}%")

    print("\n" + "=" * 60)
    print("VALIDATION: cloud-masked gradient NCC vs MODIS (measurable regions only)")
    print("=" * 60)
    print("NOTE: NCC can only be measured on TEXTURED regions; the model's benefit")
    print("is in FEATURE-LESS regions which cannot be scored. On measurable regions")
    print("the hybrid should MATCH Stage-1 (no regression); model-only will be lower.")
    clear = ~arrays["cloud_mask"]
    m_valid = np.isfinite(m_arr) & (m_arr > 0)
    base_valid = m_valid & clear
    cor_path = os.path.join(args.output, "avhrr_bowtie_corrected.tif")
    fields = [(a_arr, "ORIGINAL (uncorrected)")]
    if os.path.exists(cor_path):
        d = gdal.Open(cor_path); s1 = d.GetRasterBand(1).ReadAsArray().astype(np.float64); d = None
        fields.append((s1, "STAGE-1 TPS (features only)"))
    fields += [(model_only, "PARAMETRIC MODEL ONLY (bulk everywhere)"),
               (warped, "HYBRID (model bulk + TPS residual)")]
    report_dir = os.path.join(args.output, "report")
    os.makedirs(report_dir, exist_ok=True)
    ncc_report(fields, m_arr, base_valid,
               path=os.path.join(report_dir, "ncc_validation_panoramic.txt"))


if __name__ == "__main__":
    main()
