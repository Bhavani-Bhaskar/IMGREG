"""
Build a land/ocean mask matching the reference (MODIS) grid, for use with
COREG_LOCAL's mask_baddata_ref parameter.

Requires: pip install global-land-mask   (offline lookup, no downloads needed
at runtime -- it's a bundled coastline dataset, ~1/12 degree resolution,
interpolated per-pixel)
"""
from osgeo import gdal, osr
import numpy as np
from global_land_mask import globe

gdal.UseExceptions()

REF_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
OUT_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km_oceanmask.tif"

ds = gdal.Open(REF_PATH)
gt = ds.GetGeoTransform()
xsize, ysize = ds.RasterXSize, ds.RasterYSize
proj = ds.GetProjection()

# Build lon/lat coordinate grids matching every pixel of the reference image
cols, rows = np.meshgrid(np.arange(xsize), np.arange(ysize))
lon = gt[0] + cols * gt[1] + rows * gt[2]
lat = gt[3] + cols * gt[4] + rows * gt[5]

print(f"Computing land/ocean mask for {xsize} x {ysize} pixels...")
is_ocean = globe.is_ocean(lat, lon)   # True where ocean -> this is what we want to EXCLUDE

print(f"Ocean fraction: {is_ocean.mean()*100:.1f}%")
print(f"Land fraction:  {(~is_ocean).mean()*100:.1f}%")

# Write out as a GeoTIFF: 1 = bad/exclude (ocean), 0 = good/keep (land)
# This matches what mask_baddata_ref expects: True/1 = invalid pixel
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(OUT_PATH, xsize, ysize, 1, gdal.GDT_Byte,
                        options=["COMPRESS=LZW"])
out_ds.SetGeoTransform(gt)
out_ds.SetProjection(proj)
out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(is_ocean.astype(np.uint8))
out_band.SetNoDataValue(255)  # not really used, just a placeholder
out_ds.FlushCache()
out_ds = None

print(f"Saved ocean/land mask to: {OUT_PATH}")
print("Pass this to COREG_LOCAL as: mask_baddata_ref=<path above>")