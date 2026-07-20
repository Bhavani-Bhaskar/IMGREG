import numpy as np
from osgeo import gdal
from pygac.calibration.noaa import Calibrator, calibrate_solar

gdal.UseExceptions()

# -----------------------------
# Input
# -----------------------------
src_path = "Data/psdd_metop/metop/hrpt_M03_20250506_0420_33701_geo_b2.tif"

ds = gdal.Open(src_path)
band = ds.GetRasterBand(1)

counts = band.ReadAsArray().astype(np.float64)

# -----------------------------
# Calibration setup
# -----------------------------
spacecraft = "metopc"

cal = Calibrator(spacecraft)

year = 2025
jday = 126

# Earth-Sun distance correction
corr = 1.0

# -----------------------------
# Calibrate Channel 2
# -----------------------------
# PyGAC uses zero-based reflective channel indices:
# CH1 -> 0
# CH2 -> 1
# CH3A -> 2

reflectance = calibrate_solar(
    counts,
    1,          # Channel 2
    year,
    jday,
    cal,
    corr
)

# Convert percent -> reflectance
reflectance = reflectance.astype(np.float32) / 100.0

# Mask invalid pixels
reflectance[counts <= 0] = np.nan

# -----------------------------
# Save GeoTIFF
# -----------------------------
driver = gdal.GetDriverByName("GTiff")

out_path = "2_outputs/05_avhrr_reflectance_ch2.tif"

out_ds = driver.Create(
    out_path,
    ds.RasterXSize,
    ds.RasterYSize,
    1,
    gdal.GDT_Float32,
)

out_ds.SetGeoTransform(ds.GetGeoTransform())
out_ds.SetProjection(ds.GetProjection())

out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(reflectance)
out_band.SetNoDataValue(np.nan)
out_band.FlushCache()

out_ds.FlushCache()

out_ds = None
ds = None

print("Saved:", out_path)
print("Min:", np.nanmin(reflectance))
print("Max:", np.nanmax(reflectance))