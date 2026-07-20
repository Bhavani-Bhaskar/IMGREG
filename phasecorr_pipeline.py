"""
phasecorr_pipeline.py
---------------------

Transparent phase-correlation geolocation correction for AVHRR (MetOp-C)
against a MODIS 1km reference, plus a fixed AROSICS run on the *same*
preprocessed inputs for apples-to-apples comparison.

Why this exists: the previous AROSICS runs (arosics_orig/run_coreg_local.py)
produced a corrected image that never changed regardless of parameters. Root
cause diagnosed empirically - the matching stage was producing noise (only
373/2217 windows matched, AROSICS' own SSIM_IMPROVED False for all of them,
ANGLE std ~98deg, 0 tie points surviving filtering). Tuning the filters can't
fix noise-level matches. This pipeline attacks the matching stage instead:

  1. Common-grid preprocessing (the new inputs are NOT co-gridded).
  2. Masking (NaN / dark / deep-ocean, keeping a coastal buffer).
  3. Edge/structure representation (gradient magnitude) so matching is robust
     to cross-sensor radiometric differences.
  4. Coarse-to-fine masked phase correlation (the ~100km bulk shift is ~90px,
     too big for one window without a coarse pass first).
  5. Independent NCC verification of every tie point (the gate the old run
     failed completely).
  6. Smooth deformation model + warp.
  7. Fixed AROSICS run on the same common-grid inputs, for comparison.

Run with the `geo` conda env:
    conda run -n geo python phasecorr_pipeline.py
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import (
    shift as nd_shift,
    map_coordinates,
    sobel,
    gaussian_filter,
    distance_transform_edt,
)
from scipy.interpolate import griddata
from skimage.registration import phase_cross_correlation

gdal.UseExceptions()

# ============================================================
# Configuration
# ============================================================

AVHRR_FILE = "arosics_orig/05_avhrr_reflectance_ch2.tif"   # target (to correct)
MODIS_FILE = "arosics_orig/modis_1km.tif"                  # reference

OUTPUT_DIR = "phasecorr_output"

# Common grid: MODIS native pixel size, overlap extent.
COMMON_PIXEL_SIZE = 0.01          # degrees
MODIS_DARK_THRESHOLD = 0.01       # reflectance below this = no signal
OCEAN_BUFFER_PIXELS = 80          # coastal ocean kept (best cross-sensor edges)

# Coarse pass (bulk-shift removal on downsampled structure images). The scene
# is a diagonal swath with lots of masked area, so the coarse global estimate
# gets a much lower valid-fraction floor than the per-window fine pass.
COARSE_DOWNSAMPLE = 4
COARSE_UPSAMPLE = 10             # sub-pixel refinement factor
COARSE_MIN_VALID = 0.10          # coarse global shift tolerates heavy masking

# Fine pass (dense local grid on full-res structure images). The search limit
# applies to the RESIDUAL after the coarse bulk shift is removed, so it can be
# modest even when the total shift is large.
FINE_WINDOW = 96                 # matching window size (px)
FINE_GRID_RES = 48               # tie-point spacing (px)
FINE_SEARCH_LIMIT = 40           # reject residual shifts larger than this (px)
FINE_UPSAMPLE = 10
MIN_VALID_FRACTION = 0.5         # skip fine windows with less valid data than this

# Independent NCC verification.
NCC_MIN_SHIFT_PX = 1.0           # below this, keep as "already aligned" anchor

# Smooth shift field: linear interpolation inside the tie-point hull, nearest
# fill outside it (bounded - never extrapolates), then light Gaussian smoothing.
FIELD_SMOOTH_SIGMA = 25.0        # px; regularizes the interpolated field


# ============================================================
# I/O helpers
# ============================================================

def save_geotiff(array, geotransform, projection, output_path, dtype=gdal.GDT_Float32):
    driver = gdal.GetDriverByName("GTiff")
    ysize, xsize = array.shape
    out = driver.Create(output_path, xsize, ysize, 1, dtype)
    out.SetGeoTransform(geotransform)
    out.SetProjection(projection)
    out.GetRasterBand(1).WriteArray(array)
    out.FlushCache()
    out = None


# ============================================================
# Step 1: common-grid preprocessing
# ============================================================

def overlap_bounds(ds_a, ds_b):
    def bounds(ds):
        gt = ds.GetGeoTransform()
        x, y = ds.RasterXSize, ds.RasterYSize
        return (gt[0], gt[3] + y * gt[5], gt[0] + x * gt[1], gt[3])
    ax0, ay0, ax1, ay1 = bounds(ds_a)
    bx0, by0, bx1, by1 = bounds(ds_b)
    return (max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1))


def warp_to_common_grid(src_path, bounds, out_path):
    """Resample a raster onto the shared overlap grid (cubic, float32)."""
    minx, miny, maxx, maxy = bounds
    gdal.Warp(
        out_path,
        src_path,
        outputBounds=(minx, miny, maxx, maxy),
        xRes=COMMON_PIXEL_SIZE,
        yRes=COMMON_PIXEL_SIZE,
        resampleAlg="cubic",
        outputType=gdal.GDT_Float32,
        dstNodata=np.nan,
        targetAlignedPixels=True,
    )
    return out_path


def preprocess_common_grid():
    ds_a = gdal.Open(AVHRR_FILE)
    ds_m = gdal.Open(MODIS_FILE)

    bounds = overlap_bounds(ds_a, ds_m)
    print(f"Overlap bounds (minx,miny,maxx,maxy): {tuple(round(v, 3) for v in bounds)}")

    avhrr_common = warp_to_common_grid(AVHRR_FILE, bounds, os.path.join(OUTPUT_DIR, "avhrr_common.tif"))
    modis_common = warp_to_common_grid(MODIS_FILE, bounds, os.path.join(OUTPUT_DIR, "modis_common.tif"))

    ds_ac = gdal.Open(avhrr_common)
    ds_mc = gdal.Open(modis_common)

    assert (ds_ac.RasterXSize, ds_ac.RasterYSize) == (ds_mc.RasterXSize, ds_mc.RasterYSize), \
        "Common-grid rasters differ in size after warp"

    print(f"Common grid: {ds_ac.RasterXSize} x {ds_ac.RasterYSize} @ {COMMON_PIXEL_SIZE} deg")

    avhrr = ds_ac.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis = ds_mc.GetRasterBand(1).ReadAsArray().astype(np.float32)

    return avhrr, modis, ds_mc.GetGeoTransform(), ds_mc.GetProjection()


# ============================================================
# Step 2: masks
# ============================================================

def build_valid_mask(avhrr, modis, geotransform):
    """AVHRR finite & >0, MODIS finite & > dark, minus deep ocean (coast kept)."""
    avhrr_valid = np.isfinite(avhrr) & (avhrr > 0)
    modis_valid = np.isfinite(modis) & (modis > MODIS_DARK_THRESHOLD)

    ocean_ok = build_land_or_coastal_mask(modis.shape, geotransform)

    combined = avhrr_valid & modis_valid & ocean_ok
    print(f"Valid mask: AVHRR={avhrr_valid.mean():.2f} MODIS={modis_valid.mean():.2f} "
          f"land/coast={ocean_ok.mean():.2f} combined={combined.mean():.2f}")
    return combined


def build_land_or_coastal_mask(shape, geotransform):
    """
    True where matching is allowed = land OR coastal ocean. Deep open ocean is
    excluded (phase correlation there is unreliable), but a coastal buffer is
    kept because coastlines are the strongest cross-sensor features. Mirrors
    arosics_orig/modis_land_mask_v2.py, regenerated on the common grid.
    """
    from global_land_mask import globe
    from scipy.ndimage import binary_dilation

    ysize, xsize = shape
    gt = geotransform
    cols, rows = np.meshgrid(np.arange(xsize), np.arange(ysize))
    lon = gt[0] + cols * gt[1] + rows * gt[2]
    lat = gt[3] + cols * gt[4] + rows * gt[5]

    is_land = ~globe.is_ocean(lat, lon)
    land_buffered = binary_dilation(is_land, iterations=OCEAN_BUFFER_PIXELS)
    return land_buffered  # True = land or coastal ocean


# ============================================================
# Step 3: edge/structure representation
# ============================================================

def structure_image(arr, valid):
    """
    Band-agnostic structure image = gradient magnitude of the mildly-smoothed
    reflectance. NaNs/invalid are filled by nearest-valid before differencing
    (so edges aren't fabricated at mask borders), then re-masked to NaN.
    """
    filled = arr.copy()
    invalid = ~valid | ~np.isfinite(filled)
    if invalid.any():
        idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        filled = filled[tuple(idx)]

    filled = gaussian_filter(filled, sigma=1.0)
    gx = sobel(filled, axis=1)
    gy = sobel(filled, axis=0)
    grad = np.hypot(gx, gy)

    grad[~valid] = np.nan
    return grad


# ============================================================
# Step 4: coarse-to-fine phase correlation
# ============================================================

def _masked_pcc(ref_win, tgt_win, upsample, min_valid=MIN_VALID_FRACTION):
    """
    Masked phase correlation returning (drow, dcol) that aligns tgt onto ref,
    or None if there isn't enough joint valid data. Uses skimage's
    reference_mask/moving_mask path; NaNs are zero-filled under the mask.
    """
    ref_valid = np.isfinite(ref_win)
    tgt_valid = np.isfinite(tgt_win)

    if ref_valid.mean() < min_valid or tgt_valid.mean() < min_valid:
        return None

    ref0 = np.where(ref_valid, ref_win, 0.0).astype(np.float64)
    tgt0 = np.where(tgt_valid, tgt_win, 0.0).astype(np.float64)

    shift, _, _ = phase_cross_correlation(
        ref0, tgt0,
        reference_mask=ref_valid,
        moving_mask=tgt_valid,
        upsample_factor=upsample,
        overlap_ratio=0.3,
    )
    return float(shift[0]), float(shift[1])


def coarse_shift(ref_struct, tgt_struct):
    """Bulk (drow, dcol) on downsampled structure images."""
    r = ref_struct[::COARSE_DOWNSAMPLE, ::COARSE_DOWNSAMPLE]
    t = tgt_struct[::COARSE_DOWNSAMPLE, ::COARSE_DOWNSAMPLE]
    res = _masked_pcc(r, t, COARSE_UPSAMPLE, min_valid=COARSE_MIN_VALID)
    if res is None:
        print("Coarse pass: insufficient valid data, assuming zero bulk shift")
        return 0.0, 0.0
    drow, dcol = res[0] * COARSE_DOWNSAMPLE, res[1] * COARSE_DOWNSAMPLE
    print(f"Coarse bulk shift: drow={drow:.2f} dcol={dcol:.2f} px")
    return drow, dcol


def fine_tie_points(ref_struct, tgt_struct, valid, coarse_drow, coarse_dcol):
    """
    Dense-grid local matching. The target is pre-shifted by the coarse estimate
    so each window only searches the small residual; the reported tie-point
    shift is coarse + residual (total shift aligning target onto reference).
    """
    tgt_coarse = nd_shift(
        np.where(np.isfinite(tgt_struct), tgt_struct, 0.0),
        shift=(coarse_drow, coarse_dcol), order=1, mode="constant", cval=0.0,
    )
    tgt_coarse_valid = nd_shift(
        valid.astype(np.float32),
        shift=(coarse_drow, coarse_dcol), order=0, mode="constant", cval=0.0,
    ) > 0.5
    tgt_coarse[~tgt_coarse_valid] = np.nan

    ysize, xsize = ref_struct.shape
    half = FINE_WINDOW // 2
    rows = range(half, ysize - half, FINE_GRID_RES)
    cols = range(half, xsize - half, FINE_GRID_RES)

    records = []
    for cy in rows:
        for cx in cols:
            r0, r1 = cy - half, cy + half
            c0, c1 = cx - half, cx + half

            ref_win = ref_struct[r0:r1, c0:c1]
            tgt_win = tgt_coarse[r0:r1, c0:c1]

            res = _masked_pcc(ref_win, tgt_win, FINE_UPSAMPLE)
            if res is None:
                continue

            res_drow, res_dcol = res
            if np.hypot(res_drow, res_dcol) > FINE_SEARCH_LIMIT:
                continue

            total_drow = coarse_drow + res_drow
            total_dcol = coarse_dcol + res_dcol

            records.append({
                "Y_IM": cy, "X_IM": cx,
                "X_SHIFT_PX": total_dcol, "Y_SHIFT_PX": total_drow,
                "X_WIN_SIZE": FINE_WINDOW, "Y_WIN_SIZE": FINE_WINDOW,
            })

    df = pd.DataFrame(records)
    print(f"Fine pass: {len(df)} raw tie points from grid")
    return df


# ============================================================
# Step 5: independent NCC verification (ported from arosics_pipeline.py)
# ============================================================

def normalized_cross_correlation(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < a.size * 0.3:
        return np.nan
    a_valid = a[valid] - a[valid].mean()
    b_valid = b[valid] - b[valid].mean()
    denom = np.sqrt((a_valid ** 2).sum() * (b_valid ** 2).sum())
    return float((a_valid * b_valid).sum() / denom) if denom else np.nan


def verify_tie_points(tie_points, avhrr_arr, modis_arr):
    """
    Recompute NCC before/after each tie point's shift directly from the
    reflectance rasters (not the structure images), and keep only points that
    genuinely improve.

    Crucial correctness point (this is the failure mode that silently "passed"
    every large-shift point before): the "after" window is pulled from the FULL
    AVHRR image at the shifted location, NOT produced by nd_shift-ing the small
    reference-sized crop. For a ~110px shift and a 96px window, nd_shift empties
    the crop to all-NaN, NCC returns NaN, and NaN->0 spuriously "beats" a
    negative baseline. Sampling the real image at (cy-dy, cx-dx) instead
    (dest = src + shift => source content sits at p-shift) gives an actual,
    populated window, so the NCC comparison is meaningful even for large shifts.
    """
    if len(tie_points) == 0:
        return tie_points

    tp = tie_points.copy()
    tp["shift_px"] = np.hypot(tp["X_SHIFT_PX"], tp["Y_SHIFT_PX"])
    H, W = avhrr_arr.shape

    keep, ncc_before_list, ncc_after_list = [], [], []
    for _, row in tp.iterrows():
        cy, cx = int(row.Y_IM), int(row.X_IM)
        hy, hx = int(row.Y_WIN_SIZE) // 2, int(row.X_WIN_SIZE) // 2
        dx, dy = float(row.X_SHIFT_PX), float(row.Y_SHIFT_PX)

        r0, r1, c0, c1 = cy - hy, cy + hy, cx - hx, cx + hx
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
            keep.append(False); ncc_before_list.append(np.nan); ncc_after_list.append(np.nan)
            continue

        modis_crop = modis_arr[r0:r1, c0:c1]
        before = normalized_cross_correlation(avhrr_arr[r0:r1, c0:c1], modis_crop)

        if row.shift_px < NCC_MIN_SHIFT_PX:
            # near-zero anchor: keep as long as it has a real (finite) NCC
            keep.append(bool(np.isfinite(before)))
            ncc_before_list.append(before)
            ncc_after_list.append(before)
            continue

        # pull the AVHRR window from the shifted source location
        sr, sc = int(round(cy - dy)), int(round(cx - dx))
        if sr - hy < 0 or sc - hx < 0 or sr + hy > H or sc + hx > W:
            keep.append(False); ncc_before_list.append(before); ncc_after_list.append(np.nan)
            continue

        avhrr_after = avhrr_arr[sr - hy:sr + hy, sc - hx:sc + hx]
        after = normalized_cross_correlation(avhrr_after, modis_crop)

        improved = np.isfinite(after) and np.isfinite(before) and (after > before)
        keep.append(bool(improved))
        ncc_before_list.append(before)
        ncc_after_list.append(after)

    tp["ncc_before"] = ncc_before_list
    tp["ncc_after"] = ncc_after_list
    tp["ncc_verified"] = keep

    return tp[tp["ncc_verified"]].reset_index(drop=True)


# ============================================================
# Step 6: smooth deformation model + warp
# ============================================================

def build_smooth_shift_field(tie_points, shape, sigma=FIELD_SMOOTH_SIGMA):
    """
    Bounded, smooth shift field from the verified tie points:
      1. linear interpolation inside the tie-point convex hull,
      2. nearest-neighbour fill outside it (holds the boundary shift - never
         extrapolates, unlike a global polynomial which blew up to +400px when
         the points didn't reach the scene edges),
      3. light Gaussian smoothing to regularize per-point noise, suiting a
         smooth swath distortion.
    Every output value is therefore within the range of observed tie-point
    shifts, so the warp can't invent a wild correction at the un-covered edges.
    """
    points = tie_points[["Y_IM", "X_IM"]].to_numpy(dtype=np.float64)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    grid = np.column_stack([rows.ravel(), cols.ravel()])

    def field(values):
        lin = griddata(points, values, grid, method="linear").reshape(shape)
        near = griddata(points, values, grid, method="nearest").reshape(shape)
        filled = np.where(np.isnan(lin), near, lin)
        return gaussian_filter(filled, sigma=sigma).astype(np.float32)

    dx = field(tie_points["X_SHIFT_PX"].to_numpy())
    dy = field(tie_points["Y_SHIFT_PX"].to_numpy())

    print(f"Smooth field (linear+nearest, sigma={sigma}px) from {len(tie_points)} points; "
          f"dx[min,max]=[{dx.min():.1f},{dx.max():.1f}] dy[min,max]=[{dy.min():.1f},{dy.max():.1f}]")
    return dx, dy


def warp_with_field(avhrr, dx_field, dy_field, order=1):
    """destination = source + shift -> sample source at (row-dy, col-dx)."""
    rows, cols = np.mgrid[0:dx_field.shape[0], 0:dx_field.shape[1]].astype(np.float32)
    return map_coordinates(
        avhrr, [rows - dy_field, cols - dx_field],
        order=order, mode="constant", cval=np.nan,
    )


# ============================================================
# Step 7: fixed AROSICS run on the same common-grid inputs
# ============================================================

def run_arosics_comparison(avhrr_common_path, modis_common_path, valid_mask, geotransform, projection):
    """
    Second engine: AROSICS COREG_LOCAL on the identical preprocessed inputs,
    with a bad-data mask matching our valid mask. Saves its tie table for a
    like-for-like comparison. Isolated in a try/except so an AROSICS failure
    doesn't sink the primary pipeline.
    """
    from arosics import COREG_LOCAL

    bad_mask_path = os.path.join(OUTPUT_DIR, "arosics_baddata_ref.tif")
    save_geotiff((~valid_mask).astype(np.uint8), geotransform, projection,
                 bad_mask_path, dtype=gdal.GDT_Byte)

    try:
        CRL = COREG_LOCAL(
            im_ref=modis_common_path,
            im_tgt=avhrr_common_path,
            grid_res=FINE_GRID_RES,
            window_size=(FINE_WINDOW, FINE_WINDOW),
            path_out=os.path.join(OUTPUT_DIR, "avhrr_arosics_corrected.tif"),
            fmt_out="GTiff",
            projectDir=OUTPUT_DIR,
            max_shift=100,
            tieP_filter_level=1,
            min_reliability=0,
            mask_baddata_ref=bad_mask_path,
            nodata=(np.nan, np.nan),
            CPUs=4,
            q=True,
            progress=False,
            ignore_errors=True,
        )
        CRL.calculate_spatial_shifts()
        table = CRL.CoRegPoints_table
        table.drop(columns="geometry").to_csv(
            os.path.join(OUTPUT_DIR, "arosics_tie_points.csv"), index=False)
        matched = table[table["ABS_SHIFT"] != -9999]
        valid = matched[matched["OUTLIER"] == False] if "OUTLIER" in matched else matched  # noqa: E712
        print(f"AROSICS: {len(table)} grid pts, {len(matched)} matched, {len(valid)} valid")
        return len(table), len(matched), len(valid)
    except Exception as e:
        print(f"AROSICS comparison run failed (non-fatal): {e}")
        return None


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: COMMON-GRID PREPROCESSING")
    print("=" * 60)
    avhrr, modis, geotransform, projection = preprocess_common_grid()

    print("\n" + "=" * 60)
    print("STEP 2: MASKS")
    print("=" * 60)
    valid = build_valid_mask(avhrr, modis, geotransform)
    save_geotiff(valid.astype(np.uint8), geotransform, projection,
                 os.path.join(OUTPUT_DIR, "valid_mask.tif"), dtype=gdal.GDT_Byte)

    print("\n" + "=" * 60)
    print("STEP 3: EDGE/STRUCTURE REPRESENTATION")
    print("=" * 60)
    avhrr_struct = structure_image(avhrr, valid)
    modis_struct = structure_image(modis, valid)
    save_geotiff(np.nan_to_num(avhrr_struct), geotransform, projection,
                 os.path.join(OUTPUT_DIR, "avhrr_structure.tif"))
    save_geotiff(np.nan_to_num(modis_struct), geotransform, projection,
                 os.path.join(OUTPUT_DIR, "modis_structure.tif"))
    print("Structure images (gradient magnitude) written.")

    print("\n" + "=" * 60)
    print("STEP 4: COARSE-TO-FINE PHASE CORRELATION")
    print("=" * 60)
    c_drow, c_dcol = coarse_shift(modis_struct, avhrr_struct)
    tie_points = fine_tie_points(modis_struct, avhrr_struct, valid, c_drow, c_dcol)

    print("\n" + "=" * 60)
    print("STEP 5: INDEPENDENT NCC VERIFICATION")
    print("=" * 60)
    verified = verify_tie_points(tie_points, avhrr, modis)
    print(f"Verified tie points: {len(verified)} / {len(tie_points)}")
    if len(verified):
        print(f"Max verified shift: {verified['shift_px'].max():.1f} px "
              f"(~{verified['shift_px'].max() * 1.11:.1f} km)")
    verified.to_csv(os.path.join(OUTPUT_DIR, "tie_points_verified.csv"), index=False)

    print("\n" + "=" * 60)
    print("STEP 6: SMOOTH DEFORMATION MODEL + WARP")
    print("=" * 60)
    if len(verified) == 0:
        print("No verified tie points - cannot build a correction. Stopping.")
        return
    dx_field, dy_field = build_smooth_shift_field(verified, avhrr.shape)
    save_geotiff(dx_field, geotransform, projection, os.path.join(OUTPUT_DIR, "shift_field_dx.tif"))
    save_geotiff(dy_field, geotransform, projection, os.path.join(OUTPUT_DIR, "shift_field_dy.tif"))

    corrected = warp_with_field(avhrr, dx_field, dy_field, order=1)
    save_geotiff(corrected, geotransform, projection,
                 os.path.join(OUTPUT_DIR, "avhrr_phasecorr_corrected.tif"))
    print("Wrote avhrr_phasecorr_corrected.tif")

    print("\n" + "=" * 60)
    print("STEP 7: FIXED AROSICS RUN (SAME INPUTS) FOR COMPARISON")
    print("=" * 60)
    run_arosics_comparison(
        os.path.join(OUTPUT_DIR, "avhrr_common.tif"),
        os.path.join(OUTPUT_DIR, "modis_common.tif"),
        valid, geotransform, projection,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Verified tie points : {os.path.join(OUTPUT_DIR, 'tie_points_verified.csv')}")
    print(f"Corrected AVHRR     : {os.path.join(OUTPUT_DIR, 'avhrr_phasecorr_corrected.tif')}")


if __name__ == "__main__":
    main()
