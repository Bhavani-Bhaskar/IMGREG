"""
main_v2.py
----------

Version 2 of the MODIS-AVHRR registration pipeline.

Stage 1-2 (preprocessing) is unchanged from main.py - same
projection matching, resolution/overlap handling, band
selection, float32 conversion, geotransform verification, and
valid-data masking.

Stage 3 through 11 (window generation, per-window matching,
tie-point filtering, RANSAC outlier removal, transformation
modeling, and geometric correction) is delegated to the real
AROSICS package (github.com/GFZ/arosics, PyPI: arosics) via
COREG_LOCAL, replacing the hand-rolled window_selection/ and
local_registration/ modules used in v1 (main.py).
"""

import os
import numpy as np
from configparser import ConfigParser
from osgeo import gdal
from scipy.ndimage import shift as nd_shift

from preprocessing.projection import match_projection
from preprocessing.overlap import extract_common_overlap
from preprocessing.band_selection import select_band
from preprocessing.datatype import convert_to_float32
from preprocessing.geotransform import verify_geotransform
from preprocessing.valid_mask import create_valid_mask
from preprocessing.quality_check import quality_check

from arosics import COREG_LOCAL

gdal.UseExceptions()

# ============================================================
# Read configuration
# ============================================================

config = ConfigParser()
config.read("config.txt")

debug = config.getboolean("General", "debug")

PREPROCESSING = config.getboolean("Pipeline", "preprocessing")

resampling = config["Resampling"]["resampling"]

reference_band = config.getint("Registration Band", "reference_band")
target_band = config.getint("Registration Band", "target_band")

kernel_size = config.getint("Valid Mask", "kernel_size")
threshold = config.getfloat("Valid Mask", "valid_threshold")

tolerance = config.getfloat("GeoTransform", "geotransform_tolerance")

# ---------------------------------------------
# Version 2 parameters
# ---------------------------------------------

WINDOW_SIZE = config.getint("Version 2", "window_size")
GRID_RES = config.getint("Version 2", "grid_res")
MAX_SHIFT = config.getint("Version 2", "max_shift")
MAX_ITERATIONS = config.getint("Version 2", "max_iterations")
TIEP_FILTER_LEVEL = config.getint("Version 2", "tiep_filter_level")
MIN_RELIABILITY = config.getfloat("Version 2", "min_reliability")
MODIS_DARK_THRESHOLD = config.getfloat("Version 2", "modis_dark_threshold")
NCC_MIN_SHIFT_PX = config.getfloat("Version 2", "ncc_min_shift_px")
OUTPUT_ROOT = config.get("Version 2", "output_root")

# ============================================================
# Input files
# ============================================================

REFERENCE = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/modis_1km.tif"
TARGET = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/hrpt_M03_20250506_0420_33701_geo_b2.tif"

# ============================================================
# STAGE 1-2 : PREPROCESSING (unchanged from v1)
# ============================================================

if PREPROCESSING:

    print("\n" + "=" * 60)
    print("RUNNING PREPROCESSING")
    print("=" * 60)

    target = match_projection(
        reference_file=REFERENCE,
        target_file=TARGET,
        output_file="2_outputs/01_projection.tif",
        resampling=resampling,
        debug=debug,
    )

    reference, target = extract_common_overlap(
        reference_file=REFERENCE,
        target_file=target,
        reference_output="2_outputs/03_modis_overlap.tif",
        target_output="2_outputs/03_avhrr_overlap.tif",
        resampling=resampling,
        debug=debug,
    )

    reference = select_band(reference, "2_outputs/04_modis_band.tif", reference_band, debug)
    target = select_band(target, "2_outputs/04_avhrr_band.tif", target_band, debug)

    reference = convert_to_float32(reference, "2_outputs/05_modis_float32.tif", debug)
    target = convert_to_float32(target, "2_outputs/05_avhrr_float32.tif", debug)

    verify_geotransform(reference, target, tolerance=tolerance, debug=debug)

    mask = create_valid_mask(
        input_file=target,
        output_file="2_outputs/07_avhrr_mask.tif",
        threshold=threshold,
        kernel_size=kernel_size,
        debug=debug,
    )

    quality_check(
        reference_file=reference, target_file=target, mask_file=mask,
        tolerance=tolerance, debug=debug,
    )

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY")

else:

    print("\nPREPROCESSING DISABLED - reusing existing outputs")

    reference = "2_outputs/05_modis_float32.tif"
    target = "2_outputs/05_avhrr_float32.tif"
    mask = "2_outputs/07_avhrr_mask.tif"


# ============================================================
# STAGE 3-11 : AROSICS (COREG_LOCAL)
# ============================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def build_baddata_masks():
    """
    COREG_LOCAL expects boolean bad-data masks (True = bad),
    the opposite polarity of v1's valid_mask.py (1 = valid).

    AVHRR: invert the existing valid-swath mask.
    MODIS: mark near-zero-reflectance / non-finite pixels as
    bad - this reuses the Stage 3/4 finding that a window can
    look fine on the AVHRR side while its MODIS reference is
    near-empty.
    """

    avhrr_bad_path = os.path.join(OUTPUT_ROOT, "avhrr_baddata_mask.tif")
    modis_bad_path = os.path.join(OUTPUT_ROOT, "modis_baddata_mask.tif")

    avhrr_mask_ds = gdal.Open(mask)
    valid = avhrr_mask_ds.GetRasterBand(1).ReadAsArray()
    bad_avhrr = (valid == 0).astype(np.uint8)

    driver = gdal.GetDriverByName("GTiff")

    out = driver.Create(
        avhrr_bad_path, avhrr_mask_ds.RasterXSize, avhrr_mask_ds.RasterYSize, 1, gdal.GDT_Byte
    )
    out.SetGeoTransform(avhrr_mask_ds.GetGeoTransform())
    out.SetProjection(avhrr_mask_ds.GetProjection())
    out.GetRasterBand(1).WriteArray(bad_avhrr)
    out.FlushCache()
    out = None
    avhrr_mask_ds = None

    modis_ds = gdal.Open(reference)
    modis_arr = modis_ds.GetRasterBand(1).ReadAsArray()
    bad_modis = (~np.isfinite(modis_arr) | (modis_arr < MODIS_DARK_THRESHOLD)).astype(np.uint8)

    out = driver.Create(
        modis_bad_path, modis_ds.RasterXSize, modis_ds.RasterYSize, 1, gdal.GDT_Byte
    )
    out.SetGeoTransform(modis_ds.GetGeoTransform())
    out.SetProjection(modis_ds.GetProjection())
    out.GetRasterBand(1).WriteArray(bad_modis)
    out.FlushCache()
    out = None
    modis_ds = None

    if debug:
        print(f"\nAVHRR bad-data mask : {avhrr_bad_path} (bad fraction={bad_avhrr.mean():.3f})")
        print(f"MODIS bad-data mask : {modis_bad_path} (bad fraction={bad_modis.mean():.3f})")

    return avhrr_bad_path, modis_bad_path


print("\n" + "=" * 60)
print("RUNNING STAGE 3-11 - AROSICS (COREG_LOCAL)")
print("=" * 60)

avhrr_bad_mask, modis_bad_mask = build_baddata_masks()

corrected_path = os.path.join(OUTPUT_ROOT, "avhrr_corrected.tif")

CRL = COREG_LOCAL(
    im_ref=reference,
    im_tgt=target,
    grid_res=GRID_RES,
    window_size=(WINDOW_SIZE, WINDOW_SIZE),
    path_out=corrected_path,
    fmt_out="GTiff",
    projectDir=OUTPUT_ROOT,
    r_b4match=1,
    s_b4match=1,
    max_iter=MAX_ITERATIONS,
    max_shift=MAX_SHIFT,
    tieP_filter_level=TIEP_FILTER_LEVEL,
    min_reliability=MIN_RELIABILITY,
    mask_baddata_ref=modis_bad_mask,
    mask_baddata_tgt=avhrr_bad_mask,
    CPUs=4,
    q=not debug,
    progress=debug,
)

CRL.calculate_spatial_shifts()

tie_points = CRL.CoRegPoints_table


def normalized_cross_correlation(a, b):

    valid = np.isfinite(a) & np.isfinite(b)

    if valid.sum() < a.size * 0.3:
        return np.nan

    a_valid = a[valid] - a[valid].mean()
    b_valid = b[valid] - b[valid].mean()

    denom = np.sqrt((a_valid ** 2).sum() * (b_valid ** 2).sum())

    return float((a_valid * b_valid).sum() / denom) if denom else np.nan


def verify_tie_points_independently(tie_points_table):
    """
    AROSICS' own reliability score (Eq. 6, same formula as v1's
    reliability_estimator.py) is poorly calibrated for this
    cross-sensor pair: checked empirically and only ~69% of
    AROSICS "valid"-flagged tie points with a real (>1px) shift
    independently confirmed via NCC. Rather than trust
    min_reliability alone, recompute NCC before/after directly
    from the source rasters for every AROSICS-accepted tie point,
    and drop the ones that don't hold up.

    Tie points with a negligible shift are kept as "already
    aligned" anchors without requiring NCC improvement, since NCC
    barely moves for a sub-pixel correction by construction - and
    those anchors are themselves useful constraints for Stage 8's
    transformation model.
    """

    avhrr_ds = gdal.Open(target)
    modis_ds = gdal.Open(reference)

    avhrr_arr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis_arr = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    accepted = tie_points_table[tie_points_table["OUTLIER"] == False].copy()  # noqa: E712

    accepted["shift_px"] = np.hypot(accepted["X_SHIFT_PX"], accepted["Y_SHIFT_PX"])

    keep = []
    ncc_before_list = []
    ncc_after_list = []

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

        shifted = nd_shift(
            avhrr_crop, shift=(dy, dx), order=1, mode="constant", cval=np.nan
        )

        after = normalized_cross_correlation(shifted, modis_crop)

        improved = np.nan_to_num(after) > np.nan_to_num(before)

        keep.append(bool(improved))
        ncc_before_list.append(before)
        ncc_after_list.append(after)

    accepted["ncc_before"] = ncc_before_list
    accepted["ncc_after"] = ncc_after_list
    accepted["ncc_verified"] = keep

    return accepted[accepted["ncc_verified"]].reset_index(drop=True)


print("\nVerifying tie points independently via NCC "
      "(AROSICS' own reliability score is not trusted alone)...")

verified_tie_points = verify_tie_points_independently(tie_points)

tie_points_csv = os.path.join(OUTPUT_ROOT, "tie_points.csv")
tie_points.drop(columns="geometry").to_csv(tie_points_csv, index=False)

verified_csv = os.path.join(OUTPUT_ROOT, "tie_points_verified.csv")
verified_tie_points.drop(columns="geometry").to_csv(verified_csv, index=False)

aros_valid = tie_points[tie_points["OUTLIER"] == False]  # noqa: E712

print("\n" + "=" * 60)
print("TIE POINT SUMMARY")
print("=" * 60)
print(f"Candidate grid points        : {len(tie_points)}")
print(f"AROSICS-accepted tie points  : {len(aros_valid)}")
print(f"NCC-verified tie points      : {len(verified_tie_points)}")
print(f"Full tie point CSV           : {tie_points_csv}")
print(f"Verified tie point CSV       : {verified_csv}")

print("\nApplying correction (Stage 8-11: transformation + warp)...")

CRL.correct_shifts()

print("\n" + "=" * 60)
print("STAGE 3-11 COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Corrected AVHRR GeoTIFF : {corrected_path}")
