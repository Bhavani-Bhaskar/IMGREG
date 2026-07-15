"""
preprocessing.py
----------------

Build the bow-tie pipeline's inputs directly from the RAW granule bands, so the
pipeline no longer depends on the manual_registration folder.

Reproduces the manual pipeline's Stage 1-3 (grid alignment, cloud mask, land
masks):
  Stage 1 - resample AVHRR b2 (visible), b3a (SWIR), b4 (thermal) and MODIS
            onto ONE common 0.01deg grid snapped to the MODIS pixel origin.
  Stage 2 - thermal cloud mask (ch4 DN > 600 = cold cloud tops).
  Stage 3 - row-banded Otsu land/water masks on AVHRR-SWIR and MODIS.

Inputs (raw):  Data/psdd_metop/metop/hrpt_M03_20250506_0420_33701_geo_b2/b3a/b4.tif
               Data/psdd_metop/metop/modis_1km.tif
Outputs:       bowtie_coreg/inputs/{a_arr,s_arr,b4_arr,m_arr,s_land,m_land,
               cloud_mask}.npy  +  grid.npz (geotransform, projection)

Run:  conda run -n geo python bowtie_coreg/preprocessing.py
"""

import os
import math
import numpy as np
import cv2
from osgeo import gdal

gdal.UseExceptions()

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
RAW = os.path.join(_ROOT, "Data", "psdd_metop", "metop")
GRANULE = "hrpt_M03_20250506_0420_33701"

AVHRR_B2 = os.path.join(RAW, GRANULE + "_geo_b2.tif")    # visible ch2  -> a_arr (warped)
AVHRR_B3A = os.path.join(RAW, GRANULE + "_geo_b3a.tif")  # SWIR ch3a    -> s_arr (land mask)
AVHRR_B4 = os.path.join(RAW, GRANULE + "_geo_b4.tif")    # thermal ch4  -> b4_arr (cloud)
MODIS = os.path.join(RAW, "modis_1km.tif")               # reference    -> m_arr

OUT = os.path.join(_BASE, "inputs")

RES = 0.01           # common grid resolution (deg) = MODIS native
CLOUD_DN = 600       # thermal DN above this = cloud
OTSU_BAND = 400      # row-band height for latitude-banded Otsu


def build_common_grid():
    """Snap the AVHRR footprint onto the MODIS pixel grid at RES (Stage 1)."""
    a = gdal.Open(AVHRR_B2)
    ga = a.GetGeoTransform()
    aL, aT = ga[0], ga[3]
    aR = aL + a.RasterXSize * ga[1]
    aB = aT + a.RasterYSize * ga[5]
    m = gdal.Open(MODIS)
    gm = m.GetGeoTransform()
    ox, oy = gm[0], gm[3]                      # MODIS origin sets the grid phase
    left = ox + math.floor((aL - ox) / RES) * RES
    top = oy + math.floor((aT - oy) / RES) * RES
    right = ox + math.ceil((aR - ox) / RES) * RES
    bottom = oy + math.ceil((aB - oy) / RES) * RES
    width = int(round((right - left) / RES))
    height = int(round((top - bottom) / RES))
    return {"left": left, "top": top, "right": right, "bottom": bottom,
            "width": width, "height": height, "proj": m.GetProjection()}


def resample(path, grid, fill):
    # AVHRR (fill=0): leave dstNodata unset so genuine source 0s stay 0 (uncovered
    # pixels fall back to the MEM band's 0 init). Setting dstNodata=0 makes GDAL
    # bump source 0s to 1, which perturbs the Otsu land mask - avoid it.
    kw = {} if fill == 0 else {"dstNodata": fill}
    ds = gdal.Warp(
        "", path, format="MEM",
        outputBounds=(grid["left"], grid["bottom"], grid["right"], grid["top"]),
        xRes=RES, yRes=RES, resampleAlg="bilinear",
        dstSRS=grid["proj"], **kw,
    )
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    ds = None
    return arr


def row_banded_otsu(arr, valid, band=OTSU_BAND):
    """Binary land mask via Otsu computed per latitude band (Stage 3)."""
    H, W = arr.shape
    mask = np.zeros((H, W), dtype=bool)
    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)
        sub = arr[r0:r1]
        sv = valid[r0:r1]
        if sv.sum() < 1000:
            continue
        p99 = np.nanpercentile(sub[sv], 99)
        if p99 <= 0:
            continue
        norm = np.clip(sub / p99 * 255, 0, 255)
        thr, _ = cv2.threshold(norm[sv].astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        mask[r0:r1] = (norm > thr) & sv
    return mask


def main():
    os.makedirs(OUT, exist_ok=True)

    grid = build_common_grid()
    print(f"Common grid: {grid['width']} x {grid['height']} @ {RES} deg, "
          f"origin ({grid['left']:.5f}, {grid['top']:.5f})")

    a_arr = resample(AVHRR_B2, grid, 0.0)      # AVHRR visible (0 = off-swath)
    s_arr = resample(AVHRR_B3A, grid, 0.0)     # AVHRR SWIR
    b4_arr = resample(AVHRR_B4, grid, 0.0)     # AVHRR thermal
    m_arr = resample(MODIS, grid, np.nan)      # MODIS (nan = no data)

    cloud_mask = (b4_arr > CLOUD_DN) & (b4_arr > 0)
    s_land = row_banded_otsu(s_arr, s_arr > 0)
    m_land = row_banded_otsu(m_arr, np.isfinite(m_arr) & (m_arr > 0))

    for name, arr in [("a_arr", a_arr), ("s_arr", s_arr), ("b4_arr", b4_arr),
                      ("m_arr", m_arr), ("s_land", s_land), ("m_land", m_land),
                      ("cloud_mask", cloud_mask)]:
        np.save(os.path.join(OUT, name + ".npy"), arr)

    gt = (grid["left"], RES, 0.0, grid["top"], 0.0, -RES)
    np.savez(os.path.join(OUT, "grid.npz"), geotransform=np.array(gt), projection=grid["proj"])

    print(f"Cloud fraction: {cloud_mask.mean():.3f}  "
          f"AVHRR land frac: {s_land[s_arr>0].mean():.3f}  "
          f"MODIS land frac: {m_land[np.isfinite(m_arr)&(m_arr>0)].mean():.3f}")
    print(f"Saved 7 input arrays + grid.npz to {OUT}")


if __name__ == "__main__":
    main()
