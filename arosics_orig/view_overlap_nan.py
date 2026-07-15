from osgeo import gdal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gdal.UseExceptions()

REF_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
TGT_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"

def get_bounds(path):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    x0, y0 = gt[0], gt[3]
    x1 = x0 + ds.RasterXSize * gt[1]
    y1 = y0 + ds.RasterYSize * gt[5]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

ref_bounds = get_bounds(REF_PATH)
tgt_bounds = get_bounds(TGT_PATH)
overlap = (max(ref_bounds[0], tgt_bounds[0]), max(ref_bounds[1], tgt_bounds[1]),
           min(ref_bounds[2], tgt_bounds[2]), min(ref_bounds[3], tgt_bounds[3]))

ds = gdal.Open(REF_PATH)
gt = ds.GetGeoTransform()
band = ds.GetRasterBand(1)
nodata = band.GetNoDataValue()

# Convert overlap map-coords -> pixel row/col window, then read ONLY that window
# (much faster than reading the whole 6232x5448 array)
inv_gt = gdal.InvGeoTransform(gt)
col0, row0 = gdal.ApplyGeoTransform(inv_gt, overlap[0], overlap[3])  # upper-left  (minx, maxy)
col1, row1 = gdal.ApplyGeoTransform(inv_gt, overlap[2], overlap[1])  # lower-right (maxx, miny)
col0, col1 = sorted([int(round(col0)), int(round(col1))])
row0, row1 = sorted([int(round(row0)), int(round(row1))])
col0, row0 = max(col0, 0), max(row0, 0)
col1, row1 = min(col1, ds.RasterXSize), min(row1, ds.RasterYSize)

overlap_arr = band.ReadAsArray(col0, row0, col1 - col0, row1 - row0)

if nodata is not None and isinstance(nodata, float) and np.isnan(nodata):
    bad_mask = np.isnan(overlap_arr)
elif nodata is not None:
    bad_mask = (overlap_arr == nodata) | np.isnan(overlap_arr)
else:
    bad_mask = np.isnan(overlap_arr)

plt.figure(figsize=(8, 8))
plt.imshow(bad_mask, cmap="gray")
plt.title(f"Reference NaN/fill mask WITHIN overlap region\n"
          f"({bad_mask.mean()*100:.2f}% bad)")
plt.savefig("/home/bhaskar/Documents/ImageReg/arosics_orig/overlap_nan_mask.png", dpi=100, bbox_inches="tight")
print("Saved overlap_nan_mask.png")
print(f"Overlap window shape: {overlap_arr.shape}")
print(f"Bad fraction in this window: {bad_mask.mean()*100:.2f}%")