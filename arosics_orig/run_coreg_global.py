import numpy as np
from arosics import COREG

im_reference = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
im_target    = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"

CR = COREG(im_reference, im_target,
           max_shift=100,
           nodata=(np.nan, np.nan),
           q=False)

CR.calculate_spatial_shifts()
print("\nGlobal shift result:")
print("X shift (px):", CR.x_shift_px, " Y shift (px):", CR.y_shift_px)
print("X shift (map units):", CR.x_shift_map, " Y shift (map units):", CR.y_shift_map)
print("Reliability:", CR.shift_reliability)