# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:00:45 2026

@author: laksh
"""

"""
STAGE 3: Land/water masking (for shape-based tie-point matching)
===================================================================
Raw-intensity matching (ORB/SIFT, or direct NCC on equalized brightness)
between AVHRR and MODIS was unreliable: different sensors, different
acquisition times (different cloud states), different radiometry. Initial
ORB attempt: 8000 keypoints per image, only 56 passed the ratio test, and
RANSAC kept only 13 of those as geometrically consistent -- far too few and
too noisy to fit anything.

Fix: convert both images to binary land/water masks and match shapes
instead of raw intensity. The coastline is a sensor-invariant, time-invariant
feature (clouds move, water/land boundaries don't), so this turns a fragile
cross-modal matching problem into a much more robust same-modal (binary
silhouette) matching problem.

SWIR (ch3a, 1.6um) was used for the AVHRR side because water strongly
absorbs at this wavelength (water = very dark) while land stays bright --
much higher land/water contrast than the visible band, and a far cleaner
coastline outline.

A single global Otsu threshold failed on MODIS: the brightness range is
dominated by snow/ice in the north, so global Otsu picked a threshold that
classified most of the bright snow as "land" and most of the actual
peninsula (a different brightness range) as "water". Fixed by computing
Otsu separately within row bands (latitude bands), since brightness here
varies mainly with latitude (vegetation/desert vs snow/ice), not by some
fixed global rule.
"""
import numpy as np
import cv2


def row_banded_otsu_mask(arr, valid_mask, band=400):
    """Binary mask via Otsu threshold computed separately per row-band,
    to handle a brightness gradient that varies with latitude."""
    H, W = arr.shape
    mask = np.zeros((H, W), dtype=bool)
    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)
        sub = arr[r0:r1]
        subvalid = valid_mask[r0:r1]
        if subvalid.sum() < 1000:
            continue
        vals = sub[subvalid]
        p99 = np.nanpercentile(vals, 99)
        norm = np.clip(sub / p99 * 255, 0, 255)
        thr, _ = cv2.threshold(norm[subvalid].astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        mask[r0:r1] = (norm > thr) & subvalid
    return mask


if __name__ == '__main__':
    s_arr = np.load('s_arr.npy')          # SWIR, resampled to common grid
    m_arr = np.load('m_arr.npy')          # MODIS, resampled to common grid

    s_valid = s_arr > 0
    m_valid = ~np.isnan(m_arr) & (m_arr > 0)

    s_land = row_banded_otsu_mask(s_arr, s_valid)
    m_land = row_banded_otsu_mask(m_arr, m_valid)

    np.save('s_land.npy', s_land)
    np.save('m_land.npy', m_land)
    print('SWIR land frac:', s_land[s_valid].mean())
    print('MODIS land frac:', m_land[m_valid].mean())