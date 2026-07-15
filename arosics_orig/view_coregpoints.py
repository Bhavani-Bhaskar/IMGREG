import matplotlib
matplotlib.use("Agg")  # headless-safe; savefigPath still writes the file regardless

import numpy as np
from arosics import COREG_LOCAL

im_reference = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
im_target    = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"

# Keep these identical to run_coreg_local.py so the tie point grid matches
# exactly what that run produced.
kwargs = {
    "grid_res":          64,
    "window_size":       (256, 256),
    "min_reliability":   20,
    "max_points":        5000,
    "max_shift":         100,
    "tieP_filter_level": 3,
    "nodata":            (np.nan, np.nan),
    "rs_max_outlier":    25,
    "rs_tolerance":      2.5,
    "rs_random_state":   0,
    "tieP_random_state": 0,
    "progress":          True,
    "q":                 False,
}

CRL = COREG_LOCAL(im_reference, im_target, **kwargs)
CRL.calculate_spatial_shifts()

print(f"\nValid tie points after filtering: "
      f"{(CRL.CoRegPoints_table['OUTLIER'] == False).sum()}")

# 1. Reliability map: where are the trusted vs. rejected points located?
CRL.view_CoRegPoints(
    shapes2plot="points",
    attribute2plot="RELIABILITY",
    backgroundIm="ref",
    hide_filtered=False,      # show filtered-out points too, colored by filter level (see legend)
    figsize=(12, 12),
    title="Tie points colored by RELIABILITY (all points, filtered ones marked)",
    savefigPath="/home/bhaskar/Documents/ImageReg/arosics_orig/coregpoints_reliability.png",
    showFig=False,
)

# 2. Shift vectors: do the surviving points point in a consistent direction,
#    or scatter randomly (the symptom we found in the CSV analysis)?
CRL.view_CoRegPoints(
    shapes2plot="vectors",
    attribute2plot="ABS_SHIFT",
    backgroundIm="ref",
    hide_filtered=True,       # only show points that survived all filter levels
    figsize=(12, 12),
    vector_scale=1.0,
    title="Shift vectors of valid tie points (after L1/L2/L3 filtering)",
    savefigPath="/home/bhaskar/Documents/ImageReg/arosics_orig/coregpoints_vectors.png",
    showFig=False,
)

print("\nSaved:")
print("  coregpoints_reliability.png  -- spatial distribution + reliability of ALL candidate points")
print("  coregpoints_vectors.png      -- shift vectors of only the points that survived filtering")