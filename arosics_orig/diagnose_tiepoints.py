import numpy as np
from arosics import COREG_LOCAL

im_reference = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
im_target    = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"

kwargs = {
    "grid_res":          64,
    "window_size":       (256, 256),
    "min_reliability":   0,          # <- relax fully for diagnosis, decide threshold AFTER seeing the data
    "max_points":        5000,
    "max_shift":         100,
    "tieP_filter_level": 1,          # <- only reliability-level filtering for now, skip SSIM/RANSAC while diagnosing
    "nodata":            (np.nan, np.nan),
    "tieP_random_state": 0,
    "progress":          True,
    "q":                 False,
}

CRL = COREG_LOCAL(im_reference, im_target, **kwargs)
CRL.calculate_spatial_shifts()

table = CRL.CoRegPoints_table
print("\nTotal grid points:", len(table))
print("Points with a found match (not fill value):", (table['ABS_SHIFT'] != CRL.outFillVal).sum())

matched = table[table['ABS_SHIFT'] != CRL.outFillVal]
if len(matched):
    print("\nReliability stats among matched points:")
    print(matched['RELIABILITY'].describe())
    print("\nReliability percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th pct: {np.percentile(matched['RELIABILITY'], p):.2f}")
    print("\nABS_SHIFT stats (map units) among matched points:")
    print(matched['ABS_SHIFT'].describe())

table.to_csv("/home/bhaskar/Documents/ImageReg/arosics_orig/tiepoint_diagnostics.csv", index=False)
print("\nSaved full table to tiepoint_diagnostics.csv")