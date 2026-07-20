# -*- coding: utf-8 -*-
"""
STAGE 6: TPS warp
==================
Fits a Thin Plate Spline (TPS) displacement field from validated tie points
and warps the AVHRR image onto the corrected grid.

Transforms tried and rejected (in order):
  1. Global affine via RANSAC -- extrapolated catastrophically (>1000px error)
     because distortion is spatially-varying, not rigid-body.
  2. 2nd-order polynomial regression -- even worse extrapolation (~1500px).
  3. Piecewise-affine (Delaunay) -- accurate inside hull but coverage was
     only 5-10% of image (hull of our tie points is small).
  4. TPS chosen: matches local accuracy at tie points, extrapolates smoothly,
     no hull cutoff, controlled by smoothing parameter.

Bug fixed vs earlier version: warp_with_field previously used (xs + dxf)
which is the FORWARD mapping -- this pulls pixels from the wrong location.
Correct inverse mapping is (xs - dxf): to find what belongs at output pixel
(x,y), look at input pixel (x - dx, y - dy).
"""
import numpy as np
import cv2
import rasterio
import pickle
from scipy.interpolate import RBFInterpolator


def fit_tps_field(ta, off, W, H, smoothing=15.0, grid_step=40, clip_margin=25):
    """
    Fit TPS to displacement offsets, evaluate on a coarse grid,
    upsample to full resolution. Clip to observed range +/- margin
    to prevent runaway extrapolation at swath edges.
    """
    rbf_dx = RBFInterpolator(ta, off[:, 0],
                              kernel='thin_plate_spline', smoothing=smoothing)
    rbf_dy = RBFInterpolator(ta, off[:, 1],
                              kernel='thin_plate_spline', smoothing=smoothing)

    gx = np.linspace(0, W - 1, W // grid_step + 2)
    gy = np.linspace(0, H - 1, H // grid_step + 2)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)

    dx_grid = rbf_dx(grid_pts).reshape(GX.shape).astype(np.float32)
    dy_grid = rbf_dy(grid_pts).reshape(GX.shape).astype(np.float32)

    # clip to prevent extrapolation blow-up at edges
    dx_grid = np.clip(dx_grid,
                       off[:, 0].min() - clip_margin,
                       off[:, 0].max() + clip_margin)
    dy_grid = np.clip(dy_grid,
                       off[:, 1].min() - clip_margin,
                       off[:, 1].max() + clip_margin)

    dxf = cv2.resize(dx_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    dyf = cv2.resize(dy_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    return dxf, dyf


def warp_with_field(arr, dxf, dyf):
    """
    Apply displacement field using inverse mapping (correct):
        output(x,y) = input(x - dx, y - dy)
    i.e. to fill output pixel (x,y), sample input at (x-dx, y-dy).
    Note: earlier version had (xs + dxf) which is WRONG -- that is the
    forward mapping and produces incorrect results.
    """
    H, W = arr.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    src_x = (xs - dxf).astype(np.float32)   # <-- MINUS, not plus
    src_y = (ys - dyf).astype(np.float32)
    return cv2.remap(arr.astype(np.float32), src_x, src_y,
                      interpolation=cv2.INTER_LINEAR, borderValue=0)


def mask_no_modis_coverage(warped, m_arr, cutoff_lat=6.0,
                            grid_top=42.07, grid_res=0.01):
    """
    Two post-processing steps:
    1. Crop at cutoff_lat (default 6N): below this, MODIS coverage drops to
       near-zero (<5%) so there is no reference to register against.
    2. Within the crop, zero out any pixel where MODIS has no valid data,
       so nodata=0 everywhere there is no ground truth.
    """
    cutoff_row = int((grid_top - cutoff_lat) / grid_res)
    w = warped[:cutoff_row].copy()
    m = m_arr[:cutoff_row]
    m_valid = ~np.isnan(m) & (m > 0)
    w[~m_valid] = 0
    return w, cutoff_row


if __name__ == '__main__':
    ta  = np.load('curated_ta.npy')
    off = np.load('curated_off.npy')

    a_arr = np.load('a_arr.npy').astype(np.float32)
    m_arr = np.load('m_arr.npy')
    H, W = a_arr.shape

    dxf, dyf = fit_tps_field(ta, off, W, H)
    warped    = warp_with_field(a_arr, dxf, dyf)
    warped, cutoff_row = mask_no_modis_coverage(warped, m_arr)

    print(f'Valid (registered) fraction: {(warped>0).mean():.1%}')
    print(f'Output size: {warped.shape[1]} x {warped.shape[0]} px')

    with open('grid.pkl', 'rb') as f:
        grid = pickle.load(f)

    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'nodata': 0,
        'width': W, 'height': cutoff_row,
        'crs': grid['crs'], 'transform': grid['transform'],
    }
    with rasterio.open('avhrr_registered_to_modis.tif', 'w', **profile) as dst:
        dst.write(warped.astype('float32'), 1)
    print('Saved avhrr_registered_to_modis.tif')