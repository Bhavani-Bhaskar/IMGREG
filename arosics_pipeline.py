"""
arosics_pipeline.py
--------------------

Clean, single-file AROSICS-based geolocation correction pipeline.

Inputs (already preprocessed, verified below): 2_outputs/05_avhrr_float32.tif
(target, to be corrected) and 2_outputs/05_modis_float32.tif (reference).
Both are single-band Float32 GeoTIFFs on the identical pixel grid (same CRS,
origin, pixel size, dimensions) - confirmed via gdalinfo before writing this
script, so no reprojection/resampling/band-selection is required here.

Why local, not global: the known error (near-zero shift mid-scene, ~100km
shift at the top/bottom) is spatially varying. A single global affine/shift
was tested in an earlier version of this project and found to dilute real
local corrections (points needing 15-42px got averaged down to ~3px by
nearby near-zero points). This script instead uses AROSICS' COREG_LOCAL to
generate a dense grid of local tie points, independently re-verifies each
one via normalized cross-correlation (AROSICS' own reliability score is
poorly calibrated for this cross-sensor AVHRR-DN vs MODIS-reflectance pair),
and applies each tie point's own correction via a nearest-tie-point
(Voronoi tiling) shift field so no point's correction gets blended away.

Run with the `geo` conda env, which has arosics installed:
    conda run -n geo python arosics_pipeline.py
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import shift as nd_shift, map_coordinates
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from arosics import COREG_LOCAL

gdal.UseExceptions()

# ============================================================
# Configuration
# ============================================================

AVHRR_FILE = "2_outputs/05_avhrr_float32.tif"   # target (to be corrected)
MODIS_FILE = "2_outputs/05_modis_float32.tif"   # reference

OUTPUT_DIR = "arosics_output"

# Two-pass matching. A single 64px window (below) resolves fine local
# shifts well but structurally can't lock onto shifts approaching or
# exceeding its own size - the corresponding feature may sit outside the
# window entirely. Confirmed empirically on this dataset: the fine pass
# alone topped out around 36px (~40km) measured shift, well short of the
# ~100km observed manually at the scene edges. So a coarse, large-window
# pass runs first to catch those large shifts, and its points are merged
# with the fine pass rather than replacing it (see merge_tie_points).

# Fine pass: tuned empirically in an earlier version of this project -
# window=256px gave only 7 valid tie points, window=150px gave 18,
# window=64px (grid_res=32, i.e. half-window stride) gave ~5400
# candidates while still retaining genuine local shifts.
FINE_WINDOW_SIZE = 64
FINE_GRID_RES = 32
FINE_MAX_SHIFT = 100

# Coarse pass: large window so the search area can still contain the
# true corresponding feature even at a ~100px/100km offset. Sparser grid
# (fewer, larger windows -> fewer, but more shift-capable, tie points).
COARSE_WINDOW_SIZE = 256
COARSE_GRID_RES = 128
COARSE_MAX_SHIFT = 130

MAX_ITERATIONS = 5

# Tie-point filtering inside AROSICS: 0=none, 1=+reliability, 2=+SSIM,
# 3=+SSIM+RANSAC. Keep RANSAC active.
TIEP_FILTER_LEVEL = 3

# AROSICS' own reliability score (its Eq. 6 formula) was checked
# independently and found poorly calibrated for this cross-sensor pair -
# most genuinely-good tie points still score near 0. Left at 0 (disabled);
# quality control instead comes from SSIM/RANSAC above plus the
# independent NCC re-verification below.
MIN_RELIABILITY = 0

# MODIS reflectance below this is treated as no-signal (dark
# water/no-data) when building the bad-data mask.
MODIS_DARK_THRESHOLD = 0.01

# A tie point claiming a shift of at least this many pixels must
# independently show NCC improvement to be accepted. Verified empirically
# that ~31% of AROSICS "valid"-flagged large-shift points don't hold up
# under independent re-checking. Points below this threshold are kept as
# "already aligned" anchor points without requiring NCC improvement, since
# NCC barely moves for a sub-pixel correction by construction.
NCC_MIN_SHIFT_PX = 1.0


# ============================================================
# Step 0: validate inputs need no further preprocessing
# ============================================================

def validate_inputs(avhrr_path, modis_path):

    avhrr_ds = gdal.Open(avhrr_path)
    modis_ds = gdal.Open(modis_path)

    assert avhrr_ds.GetProjection() == modis_ds.GetProjection(), \
        "AVHRR/MODIS CRS mismatch - reprojection required before this pipeline"

    avhrr_gt = avhrr_ds.GetGeoTransform()
    modis_gt = modis_ds.GetGeoTransform()
    assert np.allclose(avhrr_gt, modis_gt, atol=1e-8), \
        "AVHRR/MODIS geotransform mismatch - not on the same pixel grid"

    assert (avhrr_ds.RasterXSize, avhrr_ds.RasterYSize) == \
           (modis_ds.RasterXSize, modis_ds.RasterYSize), \
        "AVHRR/MODIS raster size mismatch"

    assert avhrr_ds.RasterCount == 1 and modis_ds.RasterCount == 1, \
        "Expected single-band rasters - run band selection first"

    print("Input validation passed: identical CRS, geotransform, size, single band.")
    print(f"  Size: {avhrr_ds.RasterXSize} x {avhrr_ds.RasterYSize}")
    print(f"  Pixel size: {avhrr_gt[1]:.6f} deg")

    return avhrr_ds, modis_ds


# ============================================================
# Step 1: bad-data masks (fill/no-signal regions, not pixel edits)
# ============================================================

def save_geotiff(array, reference_ds, output_path, dtype=gdal.GDT_Float32):

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        output_path, reference_ds.RasterXSize, reference_ds.RasterYSize, 1, dtype
    )
    out.SetGeoTransform(reference_ds.GetGeoTransform())
    out.SetProjection(reference_ds.GetProjection())
    out.GetRasterBand(1).WriteArray(array)
    out.FlushCache()
    out = None


def build_baddata_masks(avhrr_ds, modis_ds):
    """
    AVHRR: exact-zero pixels are off-swath/limb fill (confirmed: 33% of
    the raster is exact 0 with 0% NaN, consistent with a non-rectangular
    swath inside a rectangular array - not real signal).

    MODIS: non-finite or near-zero reflectance (already NaN-filled at
    9.6% of pixels; near-zero reflectance indicates no usable signal,
    e.g. deep water).
    """

    avhrr_path = os.path.join(OUTPUT_DIR, "avhrr_baddata_mask.tif")
    modis_path = os.path.join(OUTPUT_DIR, "modis_baddata_mask.tif")

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray()
    bad_avhrr = (avhrr == 0).astype(np.uint8)
    save_geotiff(bad_avhrr, avhrr_ds, avhrr_path, dtype=gdal.GDT_Byte)

    modis = modis_ds.GetRasterBand(1).ReadAsArray()
    bad_modis = (~np.isfinite(modis) | (modis < MODIS_DARK_THRESHOLD)).astype(np.uint8)
    save_geotiff(bad_modis, modis_ds, modis_path, dtype=gdal.GDT_Byte)

    print(f"AVHRR bad-data mask: {avhrr_path} (bad fraction={bad_avhrr.mean():.3f})")
    print(f"MODIS bad-data mask: {modis_path} (bad fraction={bad_modis.mean():.3f})")

    return avhrr_path, modis_path


# ============================================================
# Step 2: AROSICS COREG_LOCAL - tie point generation
# ============================================================

def run_coreg_local(modis_bad_mask, avhrr_bad_mask, window_size, grid_res, max_shift, tag):
    """
    Only computes the tie-point grid (CRL.CoRegPoints_table). We do NOT
    call CRL.correct_shifts() - AROSICS' own warp isn't used because this
    script applies its own independently-verified tiled shift field
    (Step 4) instead, which is the transform actually decided on for this
    spatially-varying-error dataset.
    """

    CRL = COREG_LOCAL(
        im_ref=MODIS_FILE,
        im_tgt=AVHRR_FILE,
        grid_res=grid_res,
        window_size=(window_size, window_size),
        path_out=os.path.join(OUTPUT_DIR, f"avhrr_corrected_arosics_{tag}.tif"),
        fmt_out="GTiff",
        projectDir=OUTPUT_DIR,
        r_b4match=1,
        s_b4match=1,
        max_iter=MAX_ITERATIONS,
        max_shift=max_shift,
        tieP_filter_level=TIEP_FILTER_LEVEL,
        min_reliability=MIN_RELIABILITY,
        mask_baddata_ref=modis_bad_mask,
        mask_baddata_tgt=avhrr_bad_mask,
        CPUs=4,
        q=False,
        progress=True,
    )

    CRL.calculate_spatial_shifts()

    return CRL


# ============================================================
# Step 3: independent tie-point verification (don't trust AROSICS alone)
# ============================================================

def normalized_cross_correlation(a, b):

    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < a.size * 0.3:
        return np.nan

    a_valid = a[valid] - a[valid].mean()
    b_valid = b[valid] - b[valid].mean()
    denom = np.sqrt((a_valid ** 2).sum() * (b_valid ** 2).sum())

    return float((a_valid * b_valid).sum() / denom) if denom else np.nan


def verify_tie_points_independently(tie_points_table, avhrr_arr, modis_arr):
    """
    Recompute NCC before/after each AROSICS-accepted tie point's claimed
    shift directly from the source rasters, and drop points that don't
    independently hold up. Near-zero-shift points are kept as anchors
    without this check (NCC barely moves for a sub-pixel correction by
    construction, and they're still useful constraints for the transform).
    """

    accepted = tie_points_table[tie_points_table["OUTLIER"] == False].copy()  # noqa: E712
    accepted["shift_px"] = np.hypot(accepted["X_SHIFT_PX"], accepted["Y_SHIFT_PX"])

    keep, ncc_before_list, ncc_after_list = [], [], []

    for _, row in accepted.iterrows():

        cx, cy = float(row.X_IM), float(row.Y_IM)
        wx, wy = float(row.X_WIN_SIZE), float(row.Y_WIN_SIZE)
        dx, dy = float(row.X_SHIFT_PX), float(row.Y_SHIFT_PX)

        r0, r1 = int(cy - wy / 2), int(cy + wy / 2)
        c0, c1 = int(cx - wx / 2), int(cx + wx / 2)

        avhrr_crop = avhrr_arr[r0:r1, c0:c1]
        modis_crop = modis_arr[r0:r1, c0:c1]

        before = normalized_cross_correlation(avhrr_crop, modis_crop)

        if row.shift_px < NCC_MIN_SHIFT_PX:
            keep.append(True)
            ncc_before_list.append(before)
            ncc_after_list.append(before)
            continue

        shifted = nd_shift(avhrr_crop, shift=(dy, dx), order=1, mode="constant", cval=np.nan)
        after = normalized_cross_correlation(shifted, modis_crop)
        improved = np.nan_to_num(after) > np.nan_to_num(before)

        keep.append(bool(improved))
        ncc_before_list.append(before)
        ncc_after_list.append(after)

    accepted["ncc_before"] = ncc_before_list
    accepted["ncc_after"] = ncc_after_list
    accepted["ncc_verified"] = keep

    return accepted[accepted["ncc_verified"]].reset_index(drop=True)


# ============================================================
# Step 4: merge fine + coarse tie points
# ============================================================

def merge_tie_points(fine_tp, coarse_tp, conflict_radius_px=150, disagreement_px=10):
    """
    The fine pass (small window) is dense and precise but structurally
    blind to shifts approaching/exceeding its own window size. The coarse
    pass (large window) can see those large shifts but is sparse and
    coarser. Keep all fine points (they're the better estimate wherever
    they have one), and add a coarse point only where it's filling a real
    gap: no fine point nearby, or the nearest fine point's shift disagrees
    sharply with a larger coarse-detected shift (suggesting the fine pass
    missed/undershot the true correction there).
    """

    if len(coarse_tp) == 0:
        return fine_tp.reset_index(drop=True)

    if len(fine_tp) == 0:
        return coarse_tp.reset_index(drop=True)

    fine_pts = fine_tp[["Y_IM", "X_IM"]].to_numpy(dtype=np.float64)
    tree = cKDTree(fine_pts)

    keep_coarse = []

    for _, row in coarse_tp.iterrows():

        dist, idx = tree.query([row.Y_IM, row.X_IM])

        if dist > conflict_radius_px:
            keep_coarse.append(True)
            continue

        nearest_fine_shift = fine_tp.iloc[int(idx)]["shift_px"]
        keep_coarse.append(bool(row.shift_px - nearest_fine_shift > disagreement_px))

    coarse_kept = coarse_tp[keep_coarse].reset_index(drop=True)

    print(f"Coarse-pass points kept as gap-fillers: {len(coarse_kept)} / {len(coarse_tp)}")

    return pd.concat([fine_tp, coarse_kept], ignore_index=True)


# ============================================================
# Step 5: apply corrections - tiled (nearest tie point) shift field
# ============================================================

def build_dense_shift_field(tie_points, shape):
    """
    Voronoi tiling by nearest tie point: every pixel takes its single
    nearest tie point's exact (dx, dy). Preserves each point's own
    verified correction exactly (no dilution from averaging with
    neighbors), at the cost of hard seams at tile boundaries - the
    tradeoff was compared against smooth linear interpolation, which
    diluted individual corrections and was rejected.
    """

    points = tie_points[["Y_IM", "X_IM"]].to_numpy(dtype=np.float64)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    grid_points = np.column_stack([rows.ravel(), cols.ravel()])

    dx = griddata(points, tie_points["X_SHIFT_PX"].to_numpy(), grid_points, method="nearest")
    dy = griddata(points, tie_points["Y_SHIFT_PX"].to_numpy(), grid_points, method="nearest")

    return dx.reshape(shape).astype(np.float32), dy.reshape(shape).astype(np.float32)


def warp_with_field(avhrr, dx_field, dy_field, order=1):
    """
    destination = source + shift (verified against AROSICS' own corrected
    raster), so for map_coordinates we sample the source at
    (row - dy_field, col - dx_field).
    """

    rows, cols = np.mgrid[0:dx_field.shape[0], 0:dx_field.shape[1]].astype(np.float32)
    src_rows = rows - dy_field
    src_cols = cols - dx_field

    return map_coordinates(avhrr, [src_rows, src_cols], order=order, mode="constant", cval=np.nan)


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 0: VALIDATE INPUTS")
    print("=" * 60)
    avhrr_ds, modis_ds = validate_inputs(AVHRR_FILE, MODIS_FILE)

    print("\n" + "=" * 60)
    print("STEP 1: BUILD BAD-DATA MASKS")
    print("=" * 60)
    avhrr_bad_mask, modis_bad_mask = build_baddata_masks(avhrr_ds, modis_ds)

    avhrr_arr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis_arr = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    print("\n" + "=" * 60)
    print("STEP 2a: AROSICS COREG_LOCAL - FINE PASS (window=%d, grid=%d)"
          % (FINE_WINDOW_SIZE, FINE_GRID_RES))
    print("=" * 60)
    CRL_fine = run_coreg_local(
        modis_bad_mask, avhrr_bad_mask, FINE_WINDOW_SIZE, FINE_GRID_RES, FINE_MAX_SHIFT, "fine"
    )
    fine_raw = CRL_fine.CoRegPoints_table
    fine_verified = verify_tie_points_independently(fine_raw, avhrr_arr, modis_arr)
    aros_valid = fine_raw[fine_raw["OUTLIER"] == False]  # noqa: E712
    print(f"Candidate grid points        : {len(fine_raw)}")
    print(f"AROSICS-accepted tie points  : {len(aros_valid)}")
    print(f"NCC-verified tie points      : {len(fine_verified)}")

    print("\n" + "=" * 60)
    print("STEP 2b: AROSICS COREG_LOCAL - COARSE PASS (window=%d, grid=%d)"
          % (COARSE_WINDOW_SIZE, COARSE_GRID_RES))
    print("=" * 60)
    CRL_coarse = run_coreg_local(
        modis_bad_mask, avhrr_bad_mask, COARSE_WINDOW_SIZE, COARSE_GRID_RES, COARSE_MAX_SHIFT, "coarse"
    )
    coarse_raw = CRL_coarse.CoRegPoints_table
    coarse_verified = verify_tie_points_independently(coarse_raw, avhrr_arr, modis_arr)
    aros_valid_c = coarse_raw[coarse_raw["OUTLIER"] == False]  # noqa: E712
    print(f"Candidate grid points        : {len(coarse_raw)}")
    print(f"AROSICS-accepted tie points  : {len(aros_valid_c)}")
    print(f"NCC-verified tie points      : {len(coarse_verified)}")

    print("\n" + "=" * 60)
    print("STEP 3: MERGE FINE + COARSE TIE POINTS")
    print("=" * 60)
    merged_tie_points = merge_tie_points(fine_verified, coarse_verified)
    print(f"Merged tie point count: {len(merged_tie_points)} "
          f"(fine={len(fine_verified)}, coarse contributed the rest)")

    tie_points_csv = os.path.join(OUTPUT_DIR, "tie_points_fine_raw.csv")
    coarse_csv = os.path.join(OUTPUT_DIR, "tie_points_coarse_raw.csv")
    verified_csv = os.path.join(OUTPUT_DIR, "tie_points_verified.csv")
    fine_raw.drop(columns="geometry").to_csv(tie_points_csv, index=False)
    coarse_raw.drop(columns="geometry").to_csv(coarse_csv, index=False)
    merged_tie_points.to_csv(verified_csv, index=False)

    max_shift_found = merged_tie_points["shift_px"].max()
    print(f"Max verified shift (merged): {max_shift_found:.1f} px "
          f"(~{max_shift_found * 1.11:.1f} km at 0.01deg/px)")

    print("\n" + "=" * 60)
    print("STEP 4: APPLY LOCAL CORRECTION (TILED SHIFT FIELD)")
    print("=" * 60)
    output_shape = (modis_ds.RasterYSize, modis_ds.RasterXSize)
    dx_field, dy_field = build_dense_shift_field(merged_tie_points, output_shape)

    save_geotiff(dx_field, modis_ds, os.path.join(OUTPUT_DIR, "shift_field_dx.tif"))
    save_geotiff(dy_field, modis_ds, os.path.join(OUTPUT_DIR, "shift_field_dy.tif"))

    corrected = warp_with_field(avhrr_arr, dx_field, dy_field, order=1)
    final_output = os.path.join(OUTPUT_DIR, "avhrr_corrected_tiled.tif")
    save_geotiff(corrected, modis_ds, final_output)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Merged verified tie points : {verified_csv}")
    print(f"Final corrected AVHRR      : {final_output}")


if __name__ == "__main__":
    main()
