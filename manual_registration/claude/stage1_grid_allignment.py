# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:59:34 2026

@author: laksh
"""

"""
STAGE 1: Grid alignment
========================
Problem found: AVHRR native pixel size (0.009939 deg) != MODIS pixel size
(0.01 deg exactly). Left uncorrected, this ~0.6% scale mismatch accumulates
to ~31px of drift over the image height and gets silently absorbed into
whatever geometric correction you fit next (contaminating your real
geolocation-error estimate with a pure resampling artifact).

Fix: define ONE clean target grid at exactly 0.01 deg/pixel, snapped to
MODIS's pixel origin (so pixel edges align exactly, not just approximately),
and resample everything (AVHRR target band, SWIR band, MODIS reference) onto
that single grid. After this step, a tie point's (row, col) means the same
location in every array.
"""
import math
import pickle
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def build_common_grid(avhrr_path, modis_path, res=0.01):
    with rasterio.open(avhrr_path) as a:
        a_bounds = a.bounds

    with rasterio.open(modis_path) as m:
        m_transform = m.transform
        m_crs = m.crs

    # Snap AVHRR's bounding box onto the MODIS pixel grid (same origin/phase)
    ox, oy = m_transform.c, m_transform.f
    left = ox + math.floor((a_bounds.left - ox) / res) * res
    top = oy + math.floor((a_bounds.top - oy) / res) * res
    right = ox + math.ceil((a_bounds.right - ox) / res) * res
    bottom = oy + math.ceil((a_bounds.bottom - oy) / res) * res

    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    transform = rasterio.transform.from_origin(left, top, res, res)
    return {'transform': transform, 'crs': m_crs, 'width': width, 'height': height}


def resample_to_grid(path, grid, resampling=Resampling.bilinear, fill=0.0):
    dst = np.full((grid['height'], grid['width']), fill, dtype=np.float64)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=grid['transform'], dst_crs=grid['crs'],
            resampling=resampling,
        )
    return dst


if __name__ == '__main__':
    AVHRR = '1782792581465_hrpt_M03_20250506_0420_33701_geo_b2.tif'   # visible
    SWIR = '1782793932232_hrpt_M03_20250506_0420_33701_geo_b3a.tif'   # SWIR 1.6um
    THERM = '1782810919162_hrpt_M03_20250506_0420_33701_geo_b4.tif'   # thermal IR ~11um
    MODIS = '1782792594224_modis_1km.tif'

    grid = build_common_grid(AVHRR, MODIS, res=0.01)
    print('Common grid:', grid['width'], grid['height'], grid['transform'])

    a_arr = resample_to_grid(AVHRR, grid)          # 0 = nodata
    s_arr = resample_to_grid(SWIR, grid)
    b4_arr = resample_to_grid(THERM, grid)
    m_arr = resample_to_grid(MODIS, grid, fill=np.nan)

    np.save('a_arr.npy', a_arr)
    np.save('s_arr.npy', s_arr)
    np.save('b4_arr.npy', b4_arr)
    np.save('m_arr.npy', m_arr)
    with open('grid.pkl', 'wb') as f:
        pickle.dump(grid, f)