from osgeo import gdal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gdal.UseExceptions()

REF_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
TGT_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"

# Center of the matching window AROSICS used, from your global COREG run
CENTER_LON = 79.62435769648198
CENTER_LAT = 20.420796564693784
HALF_SIZE_DEG = 1.0  # ~1 degree box around the center, adjust as needed

def read_patch(path, lon0, lat0, half_deg):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    col0, row0 = gdal.ApplyGeoTransform(inv_gt, lon0 - half_deg, lat0 + half_deg)
    col1, row1 = gdal.ApplyGeoTransform(inv_gt, lon0 + half_deg, lat0 - half_deg)
    col0, col1 = sorted([int(round(col0)), int(round(col1))])
    row0, row1 = sorted([int(round(row0)), int(round(row1))])
    col0, row0 = max(col0, 0), max(row0, 0)
    col1 = min(col1, ds.RasterXSize)
    row1 = min(row1, ds.RasterYSize)
    arr = ds.GetRasterBand(1).ReadAsArray(col0, row0, col1 - col0, row1 - row0)
    return arr

ref_patch = read_patch(REF_PATH, CENTER_LON, CENTER_LAT, HALF_SIZE_DEG)
tgt_patch = read_patch(TGT_PATH, CENTER_LON, CENTER_LAT, HALF_SIZE_DEG)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(ref_patch, cmap="gray")
axes[0].set_title(f"REFERENCE (MODIS) patch\nshape={ref_patch.shape}")
axes[1].imshow(tgt_patch, cmap="gray")
axes[1].set_title(f"TARGET (AVHRR ch2) patch\nshape={tgt_patch.shape}")
plt.suptitle(f"Same geographic area (~{CENTER_LON:.2f}E, {CENTER_LAT:.2f}N, ±{HALF_SIZE_DEG} deg)")
plt.savefig("/home/bhaskar/Documents/ImageReg/arosics_orig/visual_compare.png", dpi=110, bbox_inches="tight")
print("Saved visual_compare.png")
print("Reference patch stats: min=%.4f max=%.4f mean=%.4f" % (np.nanmin(ref_patch), np.nanmax(ref_patch), np.nanmean(ref_patch)))
print("Target patch stats:    min=%.4f max=%.4f mean=%.4f" % (np.nanmin(tgt_patch), np.nanmax(tgt_patch), np.nanmean(tgt_patch)))