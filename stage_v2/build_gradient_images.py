"""
Build gradient-magnitude versions of AVHRR and MODIS.

Rationale: AVHRR (DN counts) and MODIS (reflectance) can look
structurally similar but radiometrically very different for the
same ground feature - different spectral response, calibration,
sun-angle sensitivity. Phase correlation on raw values fights
that mismatch directly. Sobel gradient magnitude keeps shared
structure (edges: coastlines, terrain, cloud boundaries) while
discarding the absolute radiometric relationship neither sensor
agrees on - a standard technique for cross-sensor registration.

Output
------
stage_v2/avhrr_gradient.tif
stage_v2/modis_gradient.tif
"""

import os
import numpy as np
import cv2
from osgeo import gdal

gdal.UseExceptions()

ROOT = os.path.join(os.path.dirname(__file__), "..")

AVHRR_FILE = os.path.join(ROOT, "2_outputs", "05_avhrr_float32.tif")
MODIS_FILE = os.path.join(ROOT, "2_outputs", "05_modis_float32.tif")

AVHRR_GRADIENT_OUT = os.path.join(ROOT, "stage_v2", "avhrr_gradient.tif")
MODIS_GRADIENT_OUT = os.path.join(ROOT, "stage_v2", "modis_gradient.tif")


def gradient_magnitude(array):

    finite = np.isfinite(array)

    filled = np.where(finite, array, 0.0).astype(np.float32)

    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # preserve original no-data footprint
    magnitude[~finite] = np.nan

    return magnitude


def save_geotiff(array, reference_ds, output_path):

    driver = gdal.GetDriverByName("GTiff")

    out = driver.Create(
        output_path, reference_ds.RasterXSize, reference_ds.RasterYSize, 1, gdal.GDT_Float32
    )

    out.SetGeoTransform(reference_ds.GetGeoTransform())
    out.SetProjection(reference_ds.GetProjection())

    out.GetRasterBand(1).WriteArray(array)
    out.GetRasterBand(1).FlushCache()

    out.FlushCache()
    out = None


def main():

    avhrr_ds = gdal.Open(AVHRR_FILE)
    modis_ds = gdal.Open(MODIS_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    print("Computing AVHRR gradient magnitude...")
    avhrr_grad = gradient_magnitude(avhrr)
    save_geotiff(avhrr_grad, avhrr_ds, AVHRR_GRADIENT_OUT)

    print("Computing MODIS gradient magnitude...")
    modis_grad = gradient_magnitude(modis)
    save_geotiff(modis_grad, modis_ds, MODIS_GRADIENT_OUT)

    print(f"\nSaved: {AVHRR_GRADIENT_OUT}")
    print(f"Saved: {MODIS_GRADIENT_OUT}")


if __name__ == "__main__":
    main()
