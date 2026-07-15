"""
v2 land/ocean mask: exclude ONLY deep open ocean.
Keeps a buffer zone of ocean pixels near coastlines (which give phase correlation
its best high-contrast features) by dilating the land region before inversion.
"""
from osgeo import gdal
import numpy as np
from scipy.ndimage import binary_dilation
from global_land_mask import globe

gdal.UseExceptions()

REF_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
OUT_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km_deep_ocean_mask.tif"

# Buffer size: dilate land by this many pixels before inverting. Reference GSD
# is ~0.01 degrees ~1.1 km, so 80 px = ~88 km coastal buffer preserved on both sides.
BUFFER_PIXELS = 80

ds = gdal.Open(REF_PATH)
gt = ds.GetGeoTransform()
xsize, ysize = ds.RasterXSize, ds.RasterYSize
proj = ds.GetProjection()

cols, rows = np.meshgrid(np.arange(xsize), np.arange(ysize))
lon = gt[0] + cols * gt[1] + rows * gt[2]
lat = gt[3] + cols * gt[4] + rows * gt[5]

print(f"Computing land/ocean mask for {xsize} x {ysize} pixels...")
is_ocean = globe.is_ocean(lat, lon)
is_land = ~is_ocean

# Grow the land region by BUFFER_PIXELS in every direction; anything still ocean
# after that is deep-ocean and gets excluded. Coastal ocean falls inside the
# dilated land zone and is retained.
print(f"Dilating land region by {BUFFER_PIXELS} pixels (~{BUFFER_PIXELS*1.1:.0f} km)...")
land_buffered = binary_dilation(is_land, iterations=BUFFER_PIXELS)
is_deep_ocean = ~land_buffered

print(f"Original ocean fraction:        {is_ocean.mean()*100:.1f}%")
print(f"Deep-ocean (excluded) fraction: {is_deep_ocean.mean()*100:.1f}%")
print(f"Coastal buffer preserved:       "
      f"{(is_ocean & ~is_deep_ocean).mean()*100:.1f}% of image = coastline zone kept in")

driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(OUT_PATH, xsize, ysize, 1, gdal.GDT_Byte,
                        options=["COMPRESS=LZW"])
out_ds.SetGeoTransform(gt)
out_ds.SetProjection(proj)
out_ds.GetRasterBand(1).WriteArray(is_deep_ocean.astype(np.uint8))
out_ds.FlushCache()
out_ds = None

print(f"\nSaved to: {OUT_PATH}")
print("Update run_coreg_local.py: mask_baddata_ref -> this file")