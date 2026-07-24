"""
AVHRR -> MODIS registration pipeline
=====================================

Full method used to produce avhrr_TRUE_FINAL_all_bands.tif:

  Stage 1  Extract the embedded GCP navigation grid from the RAW (un-navigated)
           AVHRR granule file and interpolate it to a full-resolution per-pixel
           lon/lat "geolocation array".
  Stage 2  Use that geolocation array to properly orthorectify AVHRR channel 2
           (the output band) and channel 4 (used only internally for cloud
           masking) onto a common 0.01 deg/px grid that's phase-aligned with
           the MODIS reference (GDAL's native GEOLOC_ARRAY warp -- the same
           class of method AAPP / PyGAC use, just built from the GCPs we have
           instead of raw orbital elements).
  Stage 3  Automatically fine-tune the residual sub-pixel misalignment: dense
           gradient-structure tile matching (NCC) between the orthorectified
           AVHRR and MODIS, an automatic spatial-corroboration filter (a match
           is only trusted if nearby matches agree with it), and iterative
           leave-one-out pruning of the surviving points.
  Stage 4  Fit a thin-plate-spline correction field from the final tie points
           and apply it (with the correct inverse-mapping sign) to AVHRR
           channel 2, producing the final single-band registered output.

A `build_confidence_band()` helper is included further down if you ever want
a QA layer again, but it is NOT called by default -- the output file is a
single band (just the registered AVHRR channel 2 imagery) so there's no
band-numbering ambiguity when comparing it against MODIS.

Requires: gdal (osgeo), numpy, opencv-python (cv2), scipy, scikit-learn is NOT
required. Tested with GDAL 3.8, numpy<2.

Fill in the three paths in CONFIG below and run:
    python avhrr_modis_registration.py
"""

import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel as ndi_sobel
from scipy.interpolate import RectBivariateSpline, RBFInterpolator
from scipy.spatial import cKDTree
from osgeo import gdal

gdal.UseExceptions()

# ============================== CONFIG ======================================
# The RAW granule file -- must be the version WITHOUT "_geo" in the name and
# WITHOUT a "_b2"/"_b3a"/etc band suffix (all 5 bands in one file, native
# sensor resolution, e.g. 2048 x 4780). This is the one that has embedded
# GCPs -- check with gdalinfo or GetGCPCount() before relying on it.
RAW_GRANULE_TIF = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/hrpt_M03_20250506_0420_33701.tif"

# The MODIS reference file (single band, e.g. band 2 NIR reflectance).
MODIS_TIF = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/modis_1km.tif"

# Where to write everything (intermediate files + final output).
OUTPUT_DIR = "/home/bhaskar/Documents/ImageReg/arosics_orig/trail_op/"

# Final registered multi-band product.
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "avhrr_registered_final.tif")

# Cloud threshold on AVHRR channel 4 (thermal), DN. Pixels above this are
# treated as cloud and excluded from matching.
CLOUD_DN_THRESHOLD = 600

# Common output grid resolution (degrees/pixel). 0.01 matches MODIS's own
# native pixel size for the "modis_1km" product used in this project.
GRID_RES = 0.01
# =============================================================================


# ---------------------------------------------------------------------------
# Stage 1: build a full-resolution geolocation array from the embedded GCPs
# ---------------------------------------------------------------------------
def build_geolocation_arrays(raw_tif, out_dir):
    """Read the embedded GCP grid from the raw granule and interpolate it to a
    full-resolution per-pixel (lon, lat) array. Returns paths to the two
    single-band GeoTIFFs GDAL needs for geolocation-array warping, plus the
    native (width, height) of the sensor array."""
    ds = gdal.Open(raw_tif)
    gcps = ds.GetGCPs()
    if not gcps:
        raise RuntimeError(
            f"{raw_tif} has no embedded GCPs -- this must be the RAW granule "
            f"file (before any '_geo' processing), not a re-navigated copy."
        )

    px = np.array([g.GCPPixel for g in gcps])
    ln = np.array([g.GCPLine for g in gcps])
    lon = np.array([g.GCPX for g in gcps])
    lat = np.array([g.GCPY for g in gcps])

    uniq_px = np.unique(px)
    uniq_ln = np.unique(ln)
    print(f"GCP grid: {len(uniq_ln)} x {len(uniq_px)} = {len(gcps)} points")

    # reshape the (assumed-regular) GCP grid into 2D lon/lat arrays
    lon_grid = np.full((len(uniq_ln), len(uniq_px)), np.nan)
    lat_grid = np.full((len(uniq_ln), len(uniq_px)), np.nan)
    px_idx = {v: i for i, v in enumerate(uniq_px)}
    ln_idx = {v: i for i, v in enumerate(uniq_ln)}
    for i in range(len(px)):
        r, c = ln_idx[ln[i]], px_idx[px[i]]
        lon_grid[r, c] = lon[i]
        lat_grid[r, c] = lat[i]
    if np.isnan(lon_grid).any():
        raise RuntimeError("GCP grid has gaps -- it isn't a complete regular grid; "
                            "adjust build_geolocation_arrays to handle missing nodes.")

    W_native, H_native = ds.RasterXSize, ds.RasterYSize

    spl_lon = RectBivariateSpline(uniq_ln, uniq_px, lon_grid, kx=3, ky=3)
    spl_lat = RectBivariateSpline(uniq_ln, uniq_px, lat_grid, kx=3, ky=3)
    lines_full = np.clip(np.arange(H_native), uniq_ln.min(), uniq_ln.max())
    pixels_full = np.clip(np.arange(W_native), uniq_px.min(), uniq_px.max())
    lon_full = spl_lon(lines_full, pixels_full)
    lat_full = spl_lat(lines_full, pixels_full)

    lon_path = os.path.join(out_dir, "geoloc_lon.tif")
    lat_path = os.path.join(out_dir, "geoloc_lat.tif")
    drv = gdal.GetDriverByName("GTiff")
    for path, arr in [(lon_path, lon_full), (lat_path, lat_full)]:
        out = drv.Create(path, W_native, H_native, 1, gdal.GDT_Float64)
        out.GetRasterBand(1).WriteArray(arr.astype("float64"))
        out.FlushCache()
        out = None

    return lon_path, lat_path, (W_native, H_native)


# ---------------------------------------------------------------------------
# Stage 2: orthorectify one band using the geolocation array
# ---------------------------------------------------------------------------
_GEOLOC_SRS = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
               '298.257223563]],PRIMEM["Greenwich",0],'
               'UNIT["degree",0.0174532925199433]]')


def orthorectify_band(raw_tif, band_index, lon_path, lat_path, out_dir,
                       grid_bounds, grid_res):
    """Warp a single band from the raw granule onto the common grid using
    GDAL's GEOLOC_ARRAY method (true physical geolocation, not image matching)."""
    left, top, W, H = grid_bounds
    ds = gdal.Open(raw_tif)
    band_ds = gdal.Translate(
        os.path.join(out_dir, f"_raw_band{band_index}.tif"), ds,
        format="GTiff", bandList=[band_index],
    )
    band_ds = None

    vrt_path = os.path.join(out_dir, f"_band{band_index}_geoloc.vrt")
    gdal.Translate(vrt_path, os.path.join(out_dir, f"_raw_band{band_index}.tif"), format="VRT")
    with open(vrt_path) as f:
        content = f.read()
    geoloc_block = (
        '  <Metadata domain="GEOLOCATION">\n'
        f'    <MDI key="X_DATASET">{lon_path}</MDI>\n'
        '    <MDI key="X_BAND">1</MDI>\n'
        f'    <MDI key="Y_DATASET">{lat_path}</MDI>\n'
        '    <MDI key="Y_BAND">1</MDI>\n'
        '    <MDI key="PIXEL_OFFSET">0</MDI><MDI key="LINE_OFFSET">0</MDI>\n'
        '    <MDI key="PIXEL_STEP">1</MDI><MDI key="LINE_STEP">1</MDI>\n'
        f'    <MDI key="SRS">{_GEOLOC_SRS}</MDI>\n'
        "  </Metadata>\n"
    )
    content = content.replace("</VRTDataset>", geoloc_block + "</VRTDataset>")
    with open(vrt_path, "w") as f:
        f.write(content)

    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputBounds=(left, top - H * grid_res, left + W * grid_res, top),
        xRes=grid_res, yRes=grid_res, dstSRS="EPSG:4326", resampleAlg="bilinear",
        transformerOptions=["SRC_METHOD=GEOLOC_ARRAY"], dstNodata=0,
    )
    warped_ds = gdal.Warp("", vrt_path, options=warp_opts)
    return warped_ds.GetRasterBand(1).ReadAsArray().astype("float32")


def compute_common_grid(modis_tif, raw_bounds_lonlat, grid_res):
    """Snap the AVHRR footprint onto the MODIS pixel grid's phase, matching
    the convention used throughout this project (origin on MODIS's own grid,
    not an arbitrary round number)."""
    modis_ds = gdal.Open(modis_tif)
    mgt = modis_ds.GetGeoTransform()
    lon_min, lon_max, lat_min, lat_max = raw_bounds_lonlat

    def snap(v, origin, res, mode):
        n = (v - origin) / res
        return origin + (np.floor(n) if mode == "floor" else np.ceil(n)) * res

    left = snap(lon_min, mgt[0], grid_res, "floor")
    top = snap(lat_max, mgt[3], grid_res, "ceil")
    right = snap(lon_max, mgt[0], grid_res, "ceil")
    bottom = snap(lat_min, mgt[3], grid_res, "floor")
    W = int(round((right - left) / grid_res))
    H = int(round((top - bottom) / grid_res))
    return left, top, W, H


# ---------------------------------------------------------------------------
# Stage 3: automatic fine-tuning (gradient-structure NCC tile matching)
# ---------------------------------------------------------------------------
def gradient_structure(arr, valid, gauss=1.5):
    f = arr.astype(np.float32).copy()
    f[~valid] = np.nanmedian(arr[valid]) if valid.any() else 0
    f = gaussian_filter(f, gauss)
    return np.hypot(ndi_sobel(f, 0), ndi_sobel(f, 1))


def _dist_ok(res, maxloc, maxval, peak_ratio, suppress=20):
    """Peak distinctiveness: reject if a second, well-separated peak is nearly as
    strong (an ambiguous/repeated-texture match -- the main false-match mode)."""
    r2 = res.copy()
    y0, x0 = max(maxloc[1] - suppress, 0), max(maxloc[0] - suppress, 0)
    r2[y0:maxloc[1] + suppress, x0:maxloc[0] + suppress] = -1
    _, second, _, _ = cv2.minMaxLoc(r2)
    return second <= peak_ratio * maxval


def dense_finetune_match(a_st, m_st, a_valid, m_valid,
                          tile=120, step=60, search=100, score_thresh=0.40,
                          valid_frac_min=0.85, texture_std_min=3.0, peak_ratio=0.9):
    """SEED pass: NCC tile matching with a moderately large search so it can find
    the bulk residual (incl. tens of px of bow-tie), scene-wide. Distinctiveness +
    downstream corroboration guard against false matches. Takes pre-computed
    gradient-structure images so the grow pass can reuse them."""
    H, W = a_st.shape
    half = tile // 2
    results = []
    for r in range(search + tile, H - search - tile, step):
        for c in range(search + tile, W - search - tile, step):
            av = a_valid[r:r + tile, c:c + tile]
            if av.mean() < valid_frac_min:
                continue
            t = a_st[r:r + tile, c:c + tile]
            if t.std() < texture_std_min:
                continue
            sr0, sr1 = r - search, r + tile + search
            sc0, sc1 = c - search, c + tile + search
            if m_valid[sr0:sr1, sc0:sc1].mean() < valid_frac_min:
                continue
            res = cv2.matchTemplate(m_st[sr0:sr1, sc0:sc1].astype("float32"),
                                     t.astype("float32"), cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < score_thresh:
                continue
            if abs(maxloc[0] - search) >= search - 4 or abs(maxloc[1] - search) >= search - 4:
                continue                                  # match at search boundary -> unreliable
            if not _dist_ok(res, maxloc, maxval, peak_ratio):
                continue
            dx, dy = maxloc[0] - search, maxloc[1] - search
            results.append((c + half, r + half, dx, dy, maxval))
    return np.array(results)


def _fit_shift_model(ta, off, order=2):
    """Low-order polynomial dx,dy = f(col,row). Smoothly EXTRAPOLATES the residual
    toward the swath edges (unlike TPS), so a small search locked around the
    prediction can reach large edge shifts that a blind search can't."""
    def design(col, row):
        return np.column_stack([(col ** i) * (row ** j)
                                for i in range(order + 1) for j in range(order + 1 - i)])
    A = design(ta[:, 0].astype(float), ta[:, 1].astype(float))
    cx = np.linalg.lstsq(A, off[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A, off[:, 1], rcond=None)[0]

    def predict(col, row):
        Ap = design(np.asarray(col, float), np.asarray(row, float))
        return Ap @ cx, Ap @ cy
    return predict


def grow_match(seed_ta, seed_off, a_st, m_st, a_valid, m_valid,
               tile=120, step=48, pred_tol=32, score_thresh=0.32,
               valid_frac_min=0.80, texture_std_min=3.0, accept_px=26, iters=4):
    """Seed-and-grow to REACH THE SWATH EDGES. Each iteration: fit a smooth model
    from the current points, sweep a dense grid (right up to `half` px of the edge,
    NOT skipping a big border), predict the shift per tile, and matchTemplate in a
    SMALL window locked around the prediction. This finds ~150 px edge shifts with
    a small, robust search, and only accepts matches that land near the prediction."""
    H, W = a_st.shape
    half = tile // 2
    ta = list(map(tuple, seed_ta)); off = list(map(tuple, seed_off))
    for it in range(iters):
        predict = _fit_shift_model(np.array(ta), np.array(off), order=2)
        existing = np.array(ta)
        added = 0
        for r in range(half, H - half, step):
            for c in range(half, W - half, step):
                if len(existing) and np.min(np.hypot(existing[:, 0] - c, existing[:, 1] - r)) < step * 0.7:
                    continue
                r0, c0 = r - half, c - half
                if a_valid[r0:r0 + tile, c0:c0 + tile].mean() < valid_frac_min:
                    continue
                t = a_st[r0:r0 + tile, c0:c0 + tile]
                if t.std() < texture_std_min:
                    continue
                pdx, pdy = predict([c], [r]); pdx, pdy = float(pdx[0]), float(pdy[0])
                sr0 = r0 + int(round(pdy)) - pred_tol
                sc0 = c0 + int(round(pdx)) - pred_tol
                sr1, sc1 = sr0 + tile + 2 * pred_tol, sc0 + tile + 2 * pred_tol
                if sr0 < 0 or sc0 < 0 or sr1 > H or sc1 > W:
                    continue
                if m_valid[sr0:sr1, sc0:sc1].mean() < valid_frac_min:
                    continue
                res = cv2.matchTemplate(m_st[sr0:sr1, sc0:sc1].astype("float32"),
                                         t.astype("float32"), cv2.TM_CCOEFF_NORMED)
                _, maxval, _, maxloc = cv2.minMaxLoc(res)
                if maxval < score_thresh:
                    continue
                dx = (sc0 + maxloc[0]) - c0
                dy = (sr0 + maxloc[1]) - r0
                if np.hypot(dx - pdx, dy - pdy) > accept_px:   # must land near the prediction
                    continue
                ta.append((c, r)); off.append((dx, dy)); existing = np.array(ta); added += 1
        print(f"    grow iter {it + 1}: +{added} points (total {len(ta)})")
        if added == 0:
            break
    return np.array(ta), np.array(off)


def corroboration_filter(ta, off, radius=200, tol=10, min_agree=2):
    """Keep a candidate tie point only if at least `min_agree` other nearby
    candidates (within `radius` px) agree with its offset within `tol` px.
    This replaces manual 'does this look right' checking."""
    n = len(ta)
    tree = cKDTree(ta)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        idx = [j for j in tree.query_ball_point(ta[i], radius) if j != i]
        if not idx:
            continue
        nb_off = off[idx]
        agree = np.hypot(nb_off[:, 0] - off[i, 0], nb_off[:, 1] - off[i, 1]) < tol
        if agree.sum() >= min_agree:
            keep[i] = True
    return ta[keep], off[keep]


def loo_errors(ta, off, smoothing=10.0):
    n = len(ta)
    errs = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rx = RBFInterpolator(ta[mask], off[mask, 0], kernel="thin_plate_spline", smoothing=smoothing)
        ry = RBFInterpolator(ta[mask], off[mask, 1], kernel="thin_plate_spline", smoothing=smoothing)
        px, py = rx(ta[i:i + 1])[0], ry(ta[i:i + 1])[0]
        errs[i] = np.hypot(px - off[i, 0], py - off[i, 1])
    return errs


def iterative_loo_prune(ta, off, cutoff_px=12.0, max_iter=200, min_points=30):
    """Repeatedly drop the single worst-LOO-error point until the worst
    remaining error is under `cutoff_px`. Replaces manual outlier removal."""
    ta, off = ta.copy(), off.copy()
    for _ in range(max_iter):
        if len(ta) < min_points:
            break
        errs = loo_errors(ta, off)
        worst = np.argmax(errs)
        if errs[worst] <= cutoff_px:
            break
        ta = np.delete(ta, worst, axis=0)
        off = np.delete(off, worst, axis=0)
    return ta, off


# ---------------------------------------------------------------------------
# Stage 4: fit + apply the final correction field
# ---------------------------------------------------------------------------
def fit_tps_field(ta, off, W, H, smoothing=10.0, grid_step=40, clip_margin=15):
    rbf_dx = RBFInterpolator(ta, off[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
    rbf_dy = RBFInterpolator(ta, off[:, 1], kernel="thin_plate_spline", smoothing=smoothing)
    gx = np.linspace(0, W - 1, W // grid_step + 2)
    gy = np.linspace(0, H - 1, H // grid_step + 2)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    dx_grid = rbf_dx(grid_pts).reshape(GX.shape).astype(np.float32)
    dy_grid = rbf_dy(grid_pts).reshape(GX.shape).astype(np.float32)
    dx_grid = np.clip(dx_grid, off[:, 0].min() - clip_margin, off[:, 0].max() + clip_margin)
    dy_grid = np.clip(dy_grid, off[:, 1].min() - clip_margin, off[:, 1].max() + clip_margin)
    dxf = cv2.resize(dx_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    dyf = cv2.resize(dy_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    return dxf, dyf


def apply_field(arr, dxf, dyf):
    """cv2.remap performs INVERSE mapping (samples input at src_x/src_y for
    each output pixel) -- the correct sign here is `xs - dxf`, NOT `xs + dxf`.
    (A real bug in an earlier version of this pipeline had this backwards;
    confirmed by direct ground-truth pixel-value comparison.)"""
    H, W = arr.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    src_x = (xs - dxf).astype(np.float32)
    src_y = (ys - dyf).astype(np.float32)
    return cv2.remap(arr, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderValue=0)


def build_confidence_band(ta, W, H, well_supported_px=250, flagged_boxes=None,
                           grid_bounds=None, grid_res=None):
    """3 = within `well_supported_px` of a fine-tune point (sub-pixel accurate)
       2 = physically geolocated baseline (good, but not locally fine-tuned)
       1 = explicitly flagged as unreliable (pass flagged_boxes as a list of
           (lon0, lon1, lat0, lat1) tuples for any region you've separately
           determined has too little/contradictory signal to trust -- see
           Stage 3 note below on how to decide this)"""
    tree = cKDTree(ta)
    yy, xx = np.mgrid[0:H:20, 0:W:20]
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dist, _ = tree.query(pts)
    dist_grid = dist.reshape(xx.shape)
    dist_full = cv2.resize(dist_grid.astype("float32"), (W, H), interpolation=cv2.INTER_LINEAR)

    conf = np.full((H, W), 2, dtype="float32")
    conf[dist_full <= well_supported_px] = 3

    if flagged_boxes and grid_bounds and grid_res:
        left, top = grid_bounds
        for lon0, lon1, lat0, lat1 in flagged_boxes:
            r0 = int((top - lat1) / grid_res); r1 = int((top - lat0) / grid_res)
            c0 = int((lon0 - left) / grid_res); c1 = int((lon1 - left) / grid_res)
            conf[max(r0, 0):r1, max(c0, 0):c1] = 1
    return conf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Stage 1: geolocation arrays from embedded GCPs -------------------
    print("Stage 1: building geolocation arrays from embedded GCPs...")
    lon_path, lat_path, (W_native, H_native) = build_geolocation_arrays(RAW_GRANULE_TIF, OUTPUT_DIR)

    ds = gdal.Open(RAW_GRANULE_TIF)
    n_bands = ds.RasterCount
    lon_ds = gdal.Open(lon_path)
    lat_ds = gdal.Open(lat_path)
    lon_full = lon_ds.GetRasterBand(1).ReadAsArray()
    lat_full = lat_ds.GetRasterBand(1).ReadAsArray()
    raw_bounds = (lon_full.min(), lon_full.max(), lat_full.min(), lat_full.max())

    left, top, W, H = compute_common_grid(MODIS_TIF, raw_bounds, GRID_RES)
    print(f"  common grid: {W} x {H}, origin ({left}, {top})")

    # ---- Stage 2: orthorectify only what's needed --------------------------
    # We only need the OUTPUT band (channel 2, NIR) and the CLOUD-MASK band
    # (channel 4, thermal) -- no need to warp all 5 bands if only band 2 is
    # wanted in the final output.
    print("Stage 2: orthorectifying band 2 (output) and band 4 (cloud mask) via GEOLOC_ARRAY warp...")
    OUTPUT_BAND = 2
    CLOUD_MASK_BAND = 4 if n_bands >= 4 else None

    avhrr_b2 = orthorectify_band(RAW_GRANULE_TIF, OUTPUT_BAND, lon_path, lat_path, OUTPUT_DIR,
                                  (left, top, W, H), GRID_RES)
    print(f"  band {OUTPUT_BAND}: valid frac = {(avhrr_b2 > 0).mean():.3f}")

    cloud_arr = None
    if CLOUD_MASK_BAND is not None:
        cloud_arr = orthorectify_band(RAW_GRANULE_TIF, CLOUD_MASK_BAND, lon_path, lat_path, OUTPUT_DIR,
                                       (left, top, W, H), GRID_RES)

    modis_warp_ds = gdal.Warp(
        "", MODIS_TIF, options=gdal.WarpOptions(
            format="MEM", outputBounds=(left, top - H * GRID_RES, left + W * GRID_RES, top),
            xRes=GRID_RES, yRes=GRID_RES, resampleAlg="bilinear")
    )
    modis_arr = modis_warp_ds.GetRasterBand(1).ReadAsArray().astype("float64")

    # ---- Stage 3: automatic fine-tuning (seed -> grow to the edges) --------
    print("Stage 3: automatic fine-tune tile matching...")
    cloud_mask = (cloud_arr > CLOUD_DN_THRESHOLD) if cloud_arr is not None else None
    a_valid = (avhrr_b2 > 0) & (~cloud_mask if cloud_mask is not None else True)
    m_valid = np.isfinite(modis_arr) & (modis_arr > 0)
    a_st = gradient_structure(avhrr_b2, a_valid)
    m_st = gradient_structure(modis_arr, m_valid)

    # SEED: moderately-large search finds the bulk residual scene-wide
    matches = dense_finetune_match(a_st, m_st, a_valid, m_valid)
    print(f"  seed: {len(matches)} raw candidates")
    ta, off = matches[:, :2], matches[:, 2:4]
    ta, off = corroboration_filter(ta, off)
    print(f"  seed: {len(ta)} survive corroboration")

    # GROW: model-predicted small-window search reaches the SWATH EDGES + big shifts
    if len(ta) >= 6:
        ta, off = grow_match(ta, off, a_st, m_st, a_valid, m_valid)
        ta, off = corroboration_filter(ta, off)
        print(f"  after grow + corroboration: {len(ta)} points")

    ta, off = iterative_loo_prune(ta, off)
    errs = loo_errors(ta, off)
    print(f"  FINAL: {len(ta)} tie points, LOO RMSE={np.sqrt(np.mean(errs**2)):.2f}px "
          f"median={np.median(errs):.2f}px  (NOTE: measured on surviving points only)")
    print(f"  tie-point shift range: dx[{off[:,0].min():+.0f},{off[:,0].max():+.0f}] "
          f"dy[{off[:,1].min():+.0f},{off[:,1].max():+.0f}]  "
          f"edge coverage rows[{ta[:,1].min():.0f}-{ta[:,1].max():.0f}]/{H} "
          f"cols[{ta[:,0].min():.0f}-{ta[:,0].max():.0f}]/{W}")

    # ---- Stage 4: fit + apply the correction field -------------------------
    print("Stage 4: fitting and applying final correction field...")
    dxf, dyf = fit_tps_field(ta, off, W, H)
    final_b2 = apply_field(avhrr_b2, dxf, dyf)

    # ---- HONEST quality check: per-3x3-tile gradient-NCC vs MODIS, geoloc
    #      baseline vs final. Reveals whether the EDGES actually improved,
    #      unlike the surviving-point RMSE above.
    def _gncc(a):
        ga, gm = gradient_structure(a, a > 0), m_st
        v = (a > 0) & m_valid
        rr = np.linspace(0, H, 4).astype(int); cc = np.linspace(0, W, 4).astype(int)
        out = []
        for i in range(3):
            row = []
            for j in range(3):
                vv = v[rr[i]:rr[i+1], cc[j]:cc[j+1]]
                x = ga[rr[i]:rr[i+1], cc[j]:cc[j+1]][vv]; y = gm[rr[i]:rr[i+1], cc[j]:cc[j+1]][vv]
                if x.size < 50:
                    row.append(float("nan")); continue
                x = x - x.mean(); y = y - y.mean(); d = np.sqrt((x*x).sum()*(y*y).sum())
                row.append(float((x*y).sum()/d) if d else float("nan"))
            out.append(row)
        return np.array(out)
    base_t, fin_t = _gncc(avhrr_b2), _gncc(final_b2)
    print("\n  per-3x3-tile gradient-NCC vs MODIS  (geoloc baseline -> final):")
    for i in range(3):
        print("    " + "   ".join(f"{base_t[i,j]:+.3f}->{fin_t[i,j]:+.3f}" for j in range(3)))

    # ---- write final output (band 2 ONLY -- single band, no ambiguity) ----
    print(f"Writing final output to {FINAL_OUTPUT} ...")
    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(FINAL_OUTPUT, W, H, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out.SetGeoTransform((left, GRID_RES, 0, top, 0, -GRID_RES))
    modis_proj_ds = gdal.Open(MODIS_TIF)
    out.SetProjection(modis_proj_ds.GetProjection())
    b2_band = out.GetRasterBand(1)
    b2_band.WriteArray(final_b2)
    b2_band.SetNoDataValue(0)
    b2_band.SetDescription("AVHRR channel 2, registered")
    out.FlushCache()
    print("Done. Open this in grayscale (single band, percentile stretch) to compare against MODIS.")


if __name__ == "__main__":
    main()



"""
AVHRR -> MODIS registration pipeline
=====================================

Full method used to produce avhrr_TRUE_FINAL_all_bands.tif:

  Stage 1  Extract the embedded GCP navigation grid from the RAW (un-navigated)
           AVHRR granule file and interpolate it to a full-resolution per-pixel
           lon/lat "geolocation array".
  Stage 2  Use that geolocation array to properly orthorectify AVHRR channel 2
           (the output band) and channel 4 (used only internally for cloud
           masking) onto a common 0.01 deg/px grid that's phase-aligned with
           the MODIS reference (GDAL's native GEOLOC_ARRAY warp -- the same
           class of method AAPP / PyGAC use, just built from the GCPs we have
           instead of raw orbital elements).
  Stage 3  Automatically fine-tune the residual sub-pixel misalignment: dense
           gradient-structure tile matching (NCC) between the orthorectified
           AVHRR and MODIS, an automatic spatial-corroboration filter (a match
           is only trusted if nearby matches agree with it), and iterative
           leave-one-out pruning of the surviving points.
  Stage 4  Fit a thin-plate-spline correction field from the final tie points
           and apply it (with the correct inverse-mapping sign) to AVHRR
           channel 2, producing the final single-band registered output.

A `build_confidence_band()` helper is included further down if you ever want
a QA layer again, but it is NOT called by default -- the output file is a
single band (just the registered AVHRR channel 2 imagery) so there's no
band-numbering ambiguity when comparing it against MODIS.

Requires: gdal (osgeo), numpy, opencv-python (cv2), scipy, scikit-learn is NOT
required. Tested with GDAL 3.8, numpy<2.

Fill in the three paths in CONFIG below and run:
    python avhrr_modis_registration.py
"""

import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel as ndi_sobel
from scipy.interpolate import RectBivariateSpline, RBFInterpolator
from scipy.spatial import cKDTree
from osgeo import gdal

gdal.UseExceptions()

# ============================== CONFIG ======================================
# The RAW granule file -- must be the version WITHOUT "_geo" in the name and
# WITHOUT a "_b2"/"_b3a"/etc band suffix (all 5 bands in one file, native
# sensor resolution, e.g. 2048 x 4780). This is the one that has embedded
# GCPs -- check with gdalinfo or GetGCPCount() before relying on it.
RAW_GRANULE_TIF = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\hrpt_M03_20250509_0320_33743.tif"

# The MODIS reference file (single band, e.g. band 2 NIR reflectance).
MODIS_TIF = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\modis_1km.tif"

# Where to write everything (intermediate files + final output).
OUTPUT_DIR = r"C:\Users\laksh\Downloads\pipeline\output2"

# Final registered multi-band product.
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "avhrr_registered_final.tif")

# Cloud threshold on AVHRR channel 4 (thermal), DN. Pixels above this are
# treated as cloud and excluded from matching.
CLOUD_DN_THRESHOLD = 600

# Common output grid resolution (degrees/pixel). 0.01 matches MODIS's own
# native pixel size for the "modis_1km" product used in this project.
GRID_RES = 0.01
# =============================================================================


# ---------------------------------------------------------------------------
# Stage 1: build a full-resolution geolocation array from the embedded GCPs
# ---------------------------------------------------------------------------
def build_geolocation_arrays(raw_tif, out_dir):
    """Read the embedded GCP grid from the raw granule and interpolate it to a
    full-resolution per-pixel (lon, lat) array. Returns paths to the two
    single-band GeoTIFFs GDAL needs for geolocation-array warping, plus the
    native (width, height) of the sensor array."""
    ds = gdal.Open(raw_tif)
    gcps = ds.GetGCPs()
    if not gcps:
        raise RuntimeError(
            f"{raw_tif} has no embedded GCPs -- this must be the RAW granule "
            f"file (before any '_geo' processing), not a re-navigated copy."
        )

    px = np.array([g.GCPPixel for g in gcps])
    ln = np.array([g.GCPLine for g in gcps])
    lon = np.array([g.GCPX for g in gcps])
    lat = np.array([g.GCPY for g in gcps])

    uniq_px = np.unique(px)
    uniq_ln = np.unique(ln)
    print(f"GCP grid: {len(uniq_ln)} x {len(uniq_px)} = {len(gcps)} points")

    # reshape the (assumed-regular) GCP grid into 2D lon/lat arrays
    lon_grid = np.full((len(uniq_ln), len(uniq_px)), np.nan)
    lat_grid = np.full((len(uniq_ln), len(uniq_px)), np.nan)
    px_idx = {v: i for i, v in enumerate(uniq_px)}
    ln_idx = {v: i for i, v in enumerate(uniq_ln)}
    for i in range(len(px)):
        r, c = ln_idx[ln[i]], px_idx[px[i]]
        lon_grid[r, c] = lon[i]
        lat_grid[r, c] = lat[i]
    if np.isnan(lon_grid).any():
        raise RuntimeError("GCP grid has gaps -- it isn't a complete regular grid; "
                            "adjust build_geolocation_arrays to handle missing nodes.")

    W_native, H_native = ds.RasterXSize, ds.RasterYSize

    spl_lon = RectBivariateSpline(uniq_ln, uniq_px, lon_grid, kx=3, ky=3)
    spl_lat = RectBivariateSpline(uniq_ln, uniq_px, lat_grid, kx=3, ky=3)
    lines_full = np.clip(np.arange(H_native), uniq_ln.min(), uniq_ln.max())
    pixels_full = np.clip(np.arange(W_native), uniq_px.min(), uniq_px.max())
    lon_full = spl_lon(lines_full, pixels_full)
    lat_full = spl_lat(lines_full, pixels_full)

    lon_path = os.path.join(out_dir, "geoloc_lon.tif")
    lat_path = os.path.join(out_dir, "geoloc_lat.tif")
    drv = gdal.GetDriverByName("GTiff")
    for path, arr in [(lon_path, lon_full), (lat_path, lat_full)]:
        out = drv.Create(path, W_native, H_native, 1, gdal.GDT_Float64)
        out.GetRasterBand(1).WriteArray(arr.astype("float64"))
        out.FlushCache()
        out = None

    return lon_path, lat_path, (W_native, H_native)


# ---------------------------------------------------------------------------
# Stage 2: orthorectify one band using the geolocation array
# ---------------------------------------------------------------------------
_GEOLOC_SRS = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
               '298.257223563]],PRIMEM["Greenwich",0],'
               'UNIT["degree",0.0174532925199433]]')


def orthorectify_band(raw_tif, band_index, lon_path, lat_path, out_dir,
                       grid_bounds, grid_res):
    """Warp a single band from the raw granule onto the common grid using
    GDAL's GEOLOC_ARRAY method (true physical geolocation, not image matching)."""
    left, top, W, H = grid_bounds
    ds = gdal.Open(raw_tif)
    band_ds = gdal.Translate(
        os.path.join(out_dir, f"_raw_band{band_index}.tif"), ds,
        format="GTiff", bandList=[band_index],
    )
    band_ds = None

    vrt_path = os.path.join(out_dir, f"_band{band_index}_geoloc.vrt")
    gdal.Translate(vrt_path, os.path.join(out_dir, f"_raw_band{band_index}.tif"), format="VRT")
    with open(vrt_path) as f:
        content = f.read()
    geoloc_block = (
        '  <Metadata domain="GEOLOCATION">\n'
        f'    <MDI key="X_DATASET">{lon_path}</MDI>\n'
        '    <MDI key="X_BAND">1</MDI>\n'
        f'    <MDI key="Y_DATASET">{lat_path}</MDI>\n'
        '    <MDI key="Y_BAND">1</MDI>\n'
        '    <MDI key="PIXEL_OFFSET">0</MDI><MDI key="LINE_OFFSET">0</MDI>\n'
        '    <MDI key="PIXEL_STEP">1</MDI><MDI key="LINE_STEP">1</MDI>\n'
        f'    <MDI key="SRS">{_GEOLOC_SRS}</MDI>\n'
        "  </Metadata>\n"
    )
    content = content.replace("</VRTDataset>", geoloc_block + "</VRTDataset>")
    with open(vrt_path, "w") as f:
        f.write(content)

    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputBounds=(left, top - H * grid_res, left + W * grid_res, top),
        xRes=grid_res, yRes=grid_res, dstSRS="EPSG:4326", resampleAlg="bilinear",
        transformerOptions=["SRC_METHOD=GEOLOC_ARRAY"], dstNodata=0,
    )
    warped_ds = gdal.Warp("", vrt_path, options=warp_opts)
    return warped_ds.GetRasterBand(1).ReadAsArray().astype("float32")


def compute_common_grid(modis_tif, raw_bounds_lonlat, grid_res):
    """Snap the AVHRR footprint onto the MODIS pixel grid's phase, matching
    the convention used throughout this project (origin on MODIS's own grid,
    not an arbitrary round number)."""
    modis_ds = gdal.Open(modis_tif)
    mgt = modis_ds.GetGeoTransform()
    lon_min, lon_max, lat_min, lat_max = raw_bounds_lonlat

    def snap(v, origin, res, mode):
        n = (v - origin) / res
        return origin + (np.floor(n) if mode == "floor" else np.ceil(n)) * res

    left = snap(lon_min, mgt[0], grid_res, "floor")
    top = snap(lat_max, mgt[3], grid_res, "ceil")
    right = snap(lon_max, mgt[0], grid_res, "ceil")
    bottom = snap(lat_min, mgt[3], grid_res, "floor")
    W = int(round((right - left) / grid_res))
    H = int(round((top - bottom) / grid_res))
    return left, top, W, H


# ---------------------------------------------------------------------------
# Stage 3: automatic fine-tuning (gradient-structure NCC tile matching)
# ---------------------------------------------------------------------------
def gradient_structure(arr, valid, gauss=1.5):
    f = arr.astype(np.float32).copy()
    f[~valid] = np.nanmedian(arr[valid]) if valid.any() else 0
    f = gaussian_filter(f, gauss)
    return np.hypot(ndi_sobel(f, 0), ndi_sobel(f, 1))


def dense_finetune_match(a_arr, m_arr, cloud_mask=None,
                          tile=120, step=60, search=40, score_thresh=0.35,
                          valid_frac_min=0.85, texture_std_min=3.0):
    """Small-search NCC tile matching for the residual sub-pixel correction
    that's left after physical orthorectification (NOT for finding a bulk
    100+ km shift -- that's what Stage 1/2 already fixed)."""
    a_valid = a_arr > 0
    m_valid = np.isfinite(m_arr) & (m_arr > 0)
    if cloud_mask is not None:
        a_valid = a_valid & ~cloud_mask

    a_st = gradient_structure(a_arr, a_valid)
    m_st = gradient_structure(m_arr, m_valid)

    H, W = a_arr.shape
    half = tile // 2
    results = []
    for r in range(search + tile, H - search - tile, step):
        for c in range(search + tile, W - search - tile, step):
            av = a_valid[r:r + tile, c:c + tile]
            if av.mean() < valid_frac_min:
                continue
            t = a_st[r:r + tile, c:c + tile]
            if t.std() < texture_std_min:
                continue
            sr0, sr1 = r - search, r + tile + search
            sc0, sc1 = c - search, c + tile + search
            mv = m_valid[sr0:sr1, sc0:sc1]
            if mv.mean() < valid_frac_min:
                continue
            res = cv2.matchTemplate(m_st[sr0:sr1, sc0:sc1].astype("float32"),
                                     t.astype("float32"), cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < score_thresh:
                continue
            dx, dy = maxloc[0] - search, maxloc[1] - search
            results.append((c + half, r + half, dx, dy, maxval))
    return np.array(results)


def corroboration_filter(ta, off, radius=200, tol=10, min_agree=2):
    """Keep a candidate tie point only if at least `min_agree` other nearby
    candidates (within `radius` px) agree with its offset within `tol` px.
    This replaces manual 'does this look right' checking."""
    n = len(ta)
    tree = cKDTree(ta)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        idx = [j for j in tree.query_ball_point(ta[i], radius) if j != i]
        if not idx:
            continue
        nb_off = off[idx]
        agree = np.hypot(nb_off[:, 0] - off[i, 0], nb_off[:, 1] - off[i, 1]) < tol
        if agree.sum() >= min_agree:
            keep[i] = True
    return ta[keep], off[keep]


def loo_errors(ta, off, smoothing=10.0):
    n = len(ta)
    errs = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rx = RBFInterpolator(ta[mask], off[mask, 0], kernel="thin_plate_spline", smoothing=smoothing)
        ry = RBFInterpolator(ta[mask], off[mask, 1], kernel="thin_plate_spline", smoothing=smoothing)
        px, py = rx(ta[i:i + 1])[0], ry(ta[i:i + 1])[0]
        errs[i] = np.hypot(px - off[i, 0], py - off[i, 1])
    return errs


def iterative_loo_prune(ta, off, cutoff_px=12.0, max_iter=200, min_points=30):
    """Repeatedly drop the single worst-LOO-error point until the worst
    remaining error is under `cutoff_px`. Replaces manual outlier removal."""
    ta, off = ta.copy(), off.copy()
    for _ in range(max_iter):
        if len(ta) < min_points:
            break
        errs = loo_errors(ta, off)
        worst = np.argmax(errs)
        if errs[worst] <= cutoff_px:
            break
        ta = np.delete(ta, worst, axis=0)
        off = np.delete(off, worst, axis=0)
    return ta, off


# ---------------------------------------------------------------------------
# Stage 4: fit + apply the final correction field
# ---------------------------------------------------------------------------
def fit_tps_field(ta, off, W, H, smoothing=10.0, grid_step=40, clip_margin=15):
    rbf_dx = RBFInterpolator(ta, off[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
    rbf_dy = RBFInterpolator(ta, off[:, 1], kernel="thin_plate_spline", smoothing=smoothing)
    gx = np.linspace(0, W - 1, W // grid_step + 2)
    gy = np.linspace(0, H - 1, H // grid_step + 2)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    dx_grid = rbf_dx(grid_pts).reshape(GX.shape).astype(np.float32)
    dy_grid = rbf_dy(grid_pts).reshape(GX.shape).astype(np.float32)
    dx_grid = np.clip(dx_grid, off[:, 0].min() - clip_margin, off[:, 0].max() + clip_margin)
    dy_grid = np.clip(dy_grid, off[:, 1].min() - clip_margin, off[:, 1].max() + clip_margin)
    dxf = cv2.resize(dx_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    dyf = cv2.resize(dy_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    return dxf, dyf


def apply_field(arr, dxf, dyf):
    """cv2.remap performs INVERSE mapping (samples input at src_x/src_y for
    each output pixel) -- the correct sign here is `xs - dxf`, NOT `xs + dxf`.
    (A real bug in an earlier version of this pipeline had this backwards;
    confirmed by direct ground-truth pixel-value comparison.)"""
    H, W = arr.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    src_x = (xs - dxf).astype(np.float32)
    src_y = (ys - dyf).astype(np.float32)
    return cv2.remap(arr, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderValue=0)


def build_confidence_band(ta, W, H, well_supported_px=250, flagged_boxes=None,
                           grid_bounds=None, grid_res=None):
    """3 = within `well_supported_px` of a fine-tune point (sub-pixel accurate)
       2 = physically geolocated baseline (good, but not locally fine-tuned)
       1 = explicitly flagged as unreliable (pass flagged_boxes as a list of
           (lon0, lon1, lat0, lat1) tuples for any region you've separately
           determined has too little/contradictory signal to trust -- see
           Stage 3 note below on how to decide this)"""
    tree = cKDTree(ta)
    yy, xx = np.mgrid[0:H:20, 0:W:20]
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dist, _ = tree.query(pts)
    dist_grid = dist.reshape(xx.shape)
    dist_full = cv2.resize(dist_grid.astype("float32"), (W, H), interpolation=cv2.INTER_LINEAR)

    conf = np.full((H, W), 2, dtype="float32")
    conf[dist_full <= well_supported_px] = 3

    if flagged_boxes and grid_bounds and grid_res:
        left, top = grid_bounds
        for lon0, lon1, lat0, lat1 in flagged_boxes:
            r0 = int((top - lat1) / grid_res); r1 = int((top - lat0) / grid_res)
            c0 = int((lon0 - left) / grid_res); c1 = int((lon1 - left) / grid_res)
            conf[max(r0, 0):r1, max(c0, 0):c1] = 1
    return conf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Stage 1: geolocation arrays from embedded GCPs -------------------
    print("Stage 1: building geolocation arrays from embedded GCPs...")
    lon_path, lat_path, (W_native, H_native) = build_geolocation_arrays(RAW_GRANULE_TIF, OUTPUT_DIR)

    ds = gdal.Open(RAW_GRANULE_TIF)
    n_bands = ds.RasterCount
    lon_ds = gdal.Open(lon_path)
    lat_ds = gdal.Open(lat_path)
    lon_full = lon_ds.GetRasterBand(1).ReadAsArray()
    lat_full = lat_ds.GetRasterBand(1).ReadAsArray()
    raw_bounds = (lon_full.min(), lon_full.max(), lat_full.min(), lat_full.max())

    left, top, W, H = compute_common_grid(MODIS_TIF, raw_bounds, GRID_RES)
    print(f"  common grid: {W} x {H}, origin ({left}, {top})")

    # ---- Stage 2: orthorectify only what's needed --------------------------
    # We only need the OUTPUT band (channel 2, NIR) and the CLOUD-MASK band
    # (channel 4, thermal) -- no need to warp all 5 bands if only band 2 is
    # wanted in the final output.
    print("Stage 2: orthorectifying band 2 (output) and band 4 (cloud mask) via GEOLOC_ARRAY warp...")
    OUTPUT_BAND = 2
    CLOUD_MASK_BAND = 4 if n_bands >= 4 else None

    avhrr_b2 = orthorectify_band(RAW_GRANULE_TIF, OUTPUT_BAND, lon_path, lat_path, OUTPUT_DIR,
                                  (left, top, W, H), GRID_RES)
    print(f"  band {OUTPUT_BAND}: valid frac = {(avhrr_b2 > 0).mean():.3f}")

    cloud_arr = None
    if CLOUD_MASK_BAND is not None:
        cloud_arr = orthorectify_band(RAW_GRANULE_TIF, CLOUD_MASK_BAND, lon_path, lat_path, OUTPUT_DIR,
                                       (left, top, W, H), GRID_RES)

    modis_warp_ds = gdal.Warp(
        "", MODIS_TIF, options=gdal.WarpOptions(
            format="MEM", outputBounds=(left, top - H * GRID_RES, left + W * GRID_RES, top),
            xRes=GRID_RES, yRes=GRID_RES, resampleAlg="bilinear")
    )
    modis_arr = modis_warp_ds.GetRasterBand(1).ReadAsArray().astype("float64")

    # ---- Stage 3: automatic fine-tuning -----------------------------------
    print("Stage 3: automatic fine-tune tile matching...")
    cloud_mask = (cloud_arr > CLOUD_DN_THRESHOLD) if cloud_arr is not None else None

    matches = dense_finetune_match(avhrr_b2, modis_arr, cloud_mask=cloud_mask)
    print(f"  {len(matches)} raw candidate matches")
    ta, off = matches[:, :2], matches[:, 2:4]
    ta, off = corroboration_filter(ta, off)
    print(f"  {len(ta)} survive corroboration")
    ta, off = iterative_loo_prune(ta, off)
    errs = loo_errors(ta, off)
    print(f"  FINAL: {len(ta)} tie points, LOO RMSE={np.sqrt(np.mean(errs**2)):.2f}px "
          f"median={np.median(errs):.2f}px")

    # ---- Stage 4: fit + apply the correction field -------------------------
    print("Stage 4: fitting and applying final correction field...")
    dxf, dyf = fit_tps_field(ta, off, W, H)
    final_b2 = apply_field(avhrr_b2, dxf, dyf)

    # ---- write final output (band 2 ONLY -- single band, no ambiguity) ----
    print(f"Writing final output to {FINAL_OUTPUT} ...")
    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(FINAL_OUTPUT, W, H, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out.SetGeoTransform((left, GRID_RES, 0, top, 0, -GRID_RES))
    modis_proj_ds = gdal.Open(MODIS_TIF)
    out.SetProjection(modis_proj_ds.GetProjection())
    b2_band = out.GetRasterBand(1)
    b2_band.WriteArray(final_b2)
    b2_band.SetNoDataValue(0)
    b2_band.SetDescription("AVHRR channel 2, registered")
    out.FlushCache()
    print("Done. Open this in grayscale (single band, percentile stretch) to compare against MODIS.")


if __name__ == "__main__":
    main()