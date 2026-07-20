# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:04:45 2026

@author: laksh
"""

# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cv2
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.interpolate import RBFInterpolator


# =========================
# Load + align MODIS
# =========================
def load_and_align_modis(avhrr_path, modis_path):
    with rasterio.open(avhrr_path) as a:
        a_arr = a.read(1).astype(np.float32)
        a_transform = a.transform
        a_crs = a.crs
        a_profile = a.profile.copy()

    with rasterio.open(modis_path) as m:
        m_dst = np.full(a_arr.shape, np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(m, 1),
            destination=m_dst,
            src_transform=m.transform,
            src_crs=m.crs,
            dst_transform=a_transform,
            dst_crs=a_crs,
            resampling=Resampling.bilinear,
        )

    return a_arr, m_dst, a_transform, a_crs, a_profile


# =========================
# Row band land mask
# =========================
def row_banded_land_mask(arr, valid_mask, band=400):
    H, W = arr.shape
    land = np.zeros((H, W), dtype=np.uint8)

    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)

        sub = arr[r0:r1]
        mask = valid_mask[r0:r1]

        if np.sum(mask) < 1000:
            continue

        vals = sub[mask]
        p99 = np.nanpercentile(vals, 99)

        if p99 <= 0:
            continue

        norm = np.clip(sub / (p99 + 1e-6) * 255, 0, 255).astype(np.uint8)

        # make masked image safe
        norm_masked = norm.copy()
        norm_masked[~mask] = 0

        # Otsu threshold on full band
        _, thr = cv2.threshold(
            norm_masked,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        land[r0:r1] = thr

    return land
    H, W = arr.shape
    land = np.zeros((H, W), dtype=np.uint8)

    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)

        sub = arr[r0:r1]
        subvalid = valid_mask[r0:r1]

        if subvalid.sum() < 1000:
            continue

        vals = sub[subvalid]
        p99 = np.nanpercentile(vals, 99)

        if p99 <= 0:
            continue

        norm = np.clip(sub / (p99 + 1e-6) * 255, 0, 255)

        _, thr = cv2.threshold(
            norm[subvalid].astype(np.uint8),
            0, 255,
            cv2.THRESH_OTSU
        )

        land[r0:r1] = (norm > thr).astype(np.uint8)

    return land


# =========================
# Dense matching
# =========================
def dense_tile_match(
    templ_img, templ_valid,
    ref_img, ref_valid,
    tile=220, step=110, search=200,
    score_thresh=0.65,
    std_thresh=8
):

    H, W = templ_img.shape
    ta, tb, scores = [], [], []

    for r in range(search + tile, H - search - tile, step):
        for c in range(search + tile, W - search - tile, step):

            tmask = templ_valid[r:r+tile, c:c+tile]
            if tmask.mean() < 0.9:
                continue

            t = templ_img[r:r+tile, c:c+tile]

            if np.std(t) < std_thresh:
                continue

            sr0, sr1 = r - search, r + tile + search
            sc0, sc1 = c - search, c + tile + search

            rmask = ref_valid[sr0:sr1, sc0:sc1]
            if rmask.mean() < 0.9:
                continue

            region = ref_img[sr0:sr1, sc0:sc1]

            res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)

            if maxval < score_thresh:
                continue

            dx = maxloc[0] - search
            dy = maxloc[1] - search

            if abs(dx) > search - 5 or abs(dy) > search - 5:
                continue

            ta.append((c + tile/2, r + tile/2))
            tb.append((c + tile/2 + dx, r + tile/2 + dy))
            scores.append(maxval)

    return np.array(ta), np.array(tb), np.array(scores)


# =========================
# Robust TPS warp (IMPROVED)
# =========================
def fit_tps_and_warp(target_arr, ta, tb, scores,
                     smoothing=10.0, grid_step=50):

    H, W = target_arr.shape

    if len(ta) < 6:
        raise RuntimeError("Not enough tie points")

    # =========================
    # OUTLIER REMOVAL (NEW)
    # =========================
    scores = scores / (scores.max() + 1e-6)
    keep = scores > 0.7

    ta = ta[keep]
    tb = tb[keep]

    if len(ta) < 6:
        raise RuntimeError("Not enough good tie points after filtering")

    # displacement
    off = tb - ta

    # =========================
    # Weighted TPS (NEW IMPROVEMENT)
    # =========================
    weights = np.clip(scores[keep], 0.1, 1.0)

    rbf_dx = RBFInterpolator(
        ta, off[:, 0],
        kernel='thin_plate_spline',
        smoothing=smoothing,
        neighbors=12
    )

    rbf_dy = RBFInterpolator(
        ta, off[:, 1],
        kernel='thin_plate_spline',
        smoothing=smoothing,
        neighbors=12
    )

    gx = np.linspace(0, W-1, W//grid_step + 2)
    gy = np.linspace(0, H-1, H//grid_step + 2)

    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])

    dx = rbf_dx(grid).reshape(GX.shape)
    dy = rbf_dy(grid).reshape(GY.shape)

    dx = cv2.resize(dx.astype(np.float32), (W, H))
    dy = cv2.resize(dy.astype(np.float32), (W, H))

    xs, ys = np.meshgrid(np.arange(W), np.arange(H))

    src_x = (xs - dx).astype(np.float32)
    src_y = (ys - dy).astype(np.float32)

    warped = cv2.remap(
        target_arr.astype(np.float32),
        src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("avhrr", help="Path to AVHRR GeoTIFF")
    p.add_argument("modis", help="Path to MODIS GeoTIFF")
    p.add_argument("--out", default="registered.tif")
    args = p.parse_args()

    a_arr, m_arr, transform, crs, profile = load_and_align_modis(
        args.avhrr, args.modis
    )

    a_valid = a_arr > 0
    m_valid = ~np.isnan(m_arr) & (m_arr > 0)

    s_land = row_banded_land_mask(a_arr, a_valid)
    m_land = row_banded_land_mask(m_arr, m_valid)

    s_u8 = cv2.GaussianBlur((s_land*255).astype(np.uint8), (5,5), 1)
    m_u8 = cv2.GaussianBlur((m_land*255).astype(np.uint8), (5,5), 1)

    ta, tb, scores = dense_tile_match(
        s_u8, a_valid,
        m_u8, m_valid
    )

    print("Tie points:", len(ta))

    warped = fit_tps_and_warp(a_arr, ta, tb, scores)

    profile.update(dtype="float32", count=1, nodata=0)

    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(warped.astype(np.float32), 1)
        dst.transform = transform
        dst.crs = crs

    print("Saved:", args.out)


if __name__ == "__main__":
    main()