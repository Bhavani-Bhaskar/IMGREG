# import numpy as np
# from arosics import COREG_LOCAL

# im_reference = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
# im_target    = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"

# kwargs = {
#     # --- guide-specified parameters ---
#     "grid_res":          64,          # tie point grid resolution [target px]
#     "window_size":       (256, 256),  # matching window size [px]
#     "min_reliability":   20,          # informed by diagnostic run (25th pct=0, median=2, 75th pct=25)
#     "max_points":        5000,        # cap on number of tie points used
#     "max_shift":         100,         # max expected shift [reference image px]
#     "tieP_filter_level": 3,           # 0=none, 1=reliability, 2=+SSIM, 3=+RANSAC

#     # --- data-quality fix identified during input checks ---
#     "nodata": (np.nan, np.nan),       # reference has no NoData set at file level; target already uses nan.
#                                        # Setting explicitly ensures both are excluded from matching.

#     # --- RANSAC tuning: raised after finding ~74 "valid" points still had a near-random
#     # ANGLE spread (13-347 deg, std=92 deg) and mean shift dominated by outliers vs a much
#     # smaller, more consistent median shift. Default rs_max_outlier=10 wasn't catching this
#     # -- raising it tells RANSAC to expect (and actually filter) a higher contamination rate.
#     "rs_max_outlier":     25,         # was: 10 (library default)
#     "rs_tolerance":       2.5,        # unchanged
#     "rs_random_state":    0,

#     # --- remaining parameters: library defaults ---
#     "tieP_random_state":  0,
#     "align_grids":        True,
#     "match_gsd":          False,
#     "out_gsd":            None,
#     "target_xyGrid":      None,
#     "resamp_alg_deshift": "cubic",
#     "resamp_alg_calc":    "cubic",
#     "r_b4match":          1,
#     "s_b4match":          1,
#     "max_iter":           5,
#     "calc_corners":       True,
#     "binary_ws":          True,
#     "force_quadratic_win": True,
#     "outFillVal":         -9999,
#     "CPUs":               None,
#     "progress":           True,
#     "v":                  False,
#     "q":                  False,
#     "ignore_errors":      True,

#     # --- output ---
#     "path_out":           "auto",     # writes <target>__shifted_to__<reference>.<ext> next to target
#     "fmt_out":            "GTIFF",
# }

# CRL = COREG_LOCAL(im_reference, im_target, **kwargs)

# # Calculate the tie point grid first, before committing to the warp
# CRL.calculate_spatial_shifts()

# # Save the full tie point table for offline diagnostics (median vs mean, ANGLE
# # spread, RELIABILITY distribution, etc. -- see check_tiepoint_consistency.py)
# CRL.CoRegPoints_table.to_csv(
#     "/home/bhaskar/Documents/ImageReg/arosics_orig/latest_tiepoints.csv", index=False
# )

# # Apply the local correction
# result = CRL.correct_shifts(
#     max_GCP_count=None,
#     cliptoextent=False,
#     min_points_local_corr=5,
# )

# print("Success:", CRL.success)
# print("Mean shift (px):", CRL.coreg_info["mean_shifts_px"])
# print("Mean shift (map units):", CRL.coreg_info["mean_shifts_map"])
# print("\nRun view_coregpoints.py separately to visually inspect the tie point grid.")


import numpy as np
from arosics import COREG_LOCAL

im_reference = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
im_target    = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"
land_mask    = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km_deep_ocean_mask.tif"

kwargs = {
    # --- guide-specified parameters ---
    "window_size":       (128, 128),
    "min_reliability":   20,
    "max_points":        8000,        # raised from 5000: ocean exclusion frees budget for land points
    "max_shift":         100,
    "tieP_filter_level": 1,

    # --- densified grid: now that ocean is masked out, a finer grid puts more
    # candidate points on land -- including near the AVHRR swath edges, where
    # bow-tie distortion needs the most tie points to constrain the local warp
    "grid_res":          32,          # was: 64 -- roughly 4x more candidate points

    # --- ocean exclusion: keeps matching windows off open water, where phase
    # correlation was producing unreliable/false-positive tie points
    "mask_baddata_ref":  land_mask,

    # --- nodata handling (reference has no NoData set at file level; target
    # already uses nan at the GDAL level, but pass explicitly to be safe)
    "nodata": (np.nan, np.nan),

    # --- RANSAC tuning from earlier diagnostic (ANGLE spread was near-random,
    # default rs_max_outlier=10 wasn't catching the contamination)
    "rs_max_outlier":     25,
    "rs_tolerance":       2.5,
    "rs_random_state":    0,

    # --- remaining parameters: library defaults ---
    "tieP_random_state":  0,
    "align_grids":        True,
    "match_gsd":          False,
    "out_gsd":            None,
    "target_xyGrid":      None,
    "resamp_alg_deshift": "cubic",
    "resamp_alg_calc":    "cubic",
    "r_b4match":          1,
    "s_b4match":          1,
    "max_iter":           5,
    "calc_corners":       True,
    "binary_ws":          True,
    "force_quadratic_win": True,
    "outFillVal":         -9999,
    "CPUs":               None,
    "progress":           True,
    "v":                  False,
    "q":                  False,
    "ignore_errors":      True,

    # --- output ---
    "path_out":           "auto",
    "fmt_out":             "GTIFF",
}

CRL = COREG_LOCAL(im_reference, im_target, **kwargs)

CRL.calculate_spatial_shifts()

CRL.CoRegPoints_table.to_csv(
    "/home/bhaskar/Documents/ImageReg/latest_tiepoints.csv", index=False
)

n_valid = (CRL.CoRegPoints_table['OUTLIER'] == False).sum()
n_total = len(CRL.CoRegPoints_table)
print(f"\nValid tie points: {n_valid} / {n_total} candidates "
      f"({n_valid/n_total*100:.1f}%)")

result = CRL.correct_shifts(
    max_GCP_count=None,
    cliptoextent=False,
    min_points_local_corr=5,
)

print("Success:", CRL.success)
print("Mean shift (px):", CRL.coreg_info["mean_shifts_px"])
print("Mean shift (map units):", CRL.coreg_info["mean_shifts_map"])