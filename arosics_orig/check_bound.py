from osgeo import gdal
import numpy as np

gdal.UseExceptions()

REF_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
TGT_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"

def get_bounds(path):
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open: {path}")
    gt = ds.GetGeoTransform()
    x0, y0 = gt[0], gt[3]
    x1 = x0 + ds.RasterXSize * gt[1]
    y1 = y0 + ds.RasterYSize * gt[5]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

ref_bounds = get_bounds(REF_PATH)
tgt_bounds = get_bounds(TGT_PATH)
overlap = (max(ref_bounds[0], tgt_bounds[0]), max(ref_bounds[1], tgt_bounds[1]),
           min(ref_bounds[2], tgt_bounds[2]), min(ref_bounds[3], tgt_bounds[3]))
print("Reference bounds:", ref_bounds)
print("Target bounds:   ", tgt_bounds)
print("Overlap bounds:  ", overlap)

ds = gdal.Open(REF_PATH)
gt = ds.GetGeoTransform()
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()
nodata = band.GetNoDataValue()

rows, cols = np.indices(arr.shape)
x = gt[0] + cols * gt[1]
y = gt[3] + rows * gt[5]
in_overlap = (x >= overlap[0]) & (x <= overlap[2]) & (y >= overlap[1]) & (y <= overlap[3])

is_nan_or_fill = np.isnan(arr) if np.isnan(arr).any() else np.zeros_like(arr, dtype=bool)
if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
    is_nan_or_fill = is_nan_or_fill | (arr == nodata)

nan_in_overlap = is_nan_or_fill & in_overlap
total_overlap_px = in_overlap.sum()
bad_overlap_px = nan_in_overlap.sum()

print(f"\nOverlap region pixel count: {total_overlap_px}")
print(f"NaN/fill pixels inside overlap: {bad_overlap_px}")
print(f"Fraction of overlap that is bad data: {bad_overlap_px/total_overlap_px*100:.4f}%" if total_overlap_px else "No overlap pixels found in reference grid.")