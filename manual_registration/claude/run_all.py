# -*- coding: utf-8 -*-
"""
RUN_ALL: AVHRR -> MODIS registration, full pipeline
======================================================
Edit the four path variables below then run this file.
Each stage can also be run independently as a standalone script.

Pipeline summary:
  Stage 1: Resample everything onto a common 0.01deg grid (fixes AVHRR/MODIS
            pixel-size mismatch: 0.009939 vs 0.01 deg)
  Stage 2: Build thermal-IR cloud mask (ch4, threshold DN=600)
  Stage 3: Build binary land/water masks (SWIR for AVHRR, MODIS for reference)
            using row-banded Otsu thresholding
  Stage 4: Dense tile-based NCC matching (cloud-gated, binary mask matching)
            to generate candidate tie points
  Stage 5: Load QGIS .points file, filter outliers (neighbour deviation + LOO),
            validate with leave-one-out cross-validation
  Stage 6: Fit TPS displacement field, warp AVHRR, crop to 6N, mask no-MODIS

Final accuracy (60 validated GCPs):
  LOO RMSE: ~15px (~15km), median ~7.6px, max ~44px
  Coverage: ~63% of cropped scene (lat 42N to 6N)

Known gaps: far left (lon 64-70E) and far right (lon 91-96E) have no
reliable anchor points due to cloud cover -- TPS extrapolates there.
"""

import numpy as np
import cv2
import pickle
import rasterio

import stage1_grid_alignment  as s1
import stage2_cloud_mask      as s2
import stage3_land_water_mask as s3
import stage4_tile_matching   as s4
import stage5_validation      as s5
import stage6_tps_warp        as s6

# ── Edit these four paths ────────────────────────────────────────────────────
AVHRR  = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\hrpt_M03_20250506_0420_33701_geo_b2.tif"
SWIR   = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\hrpt_M03_20250506_0420_33701_geo_b3a.tif"
THERM  = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\hrpt_M03_20250506_0420_33701_geo_b4.tif"
MODIS  = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\modis_1km.tif"
POINTS = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\hrpt_M03_b2(tie_points_4).tif.points"
OUTPUT = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\avhrr_registered_to_modis.tif"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ------------------------------------------------------------------
    # Stage 1: grid alignment
    # ------------------------------------------------------------------
    print('\n[Stage 1] Building common 0.01deg grid...')
    grid  = s1.build_common_grid(AVHRR, MODIS, res=0.01)
    a_arr = s1.resample_to_grid(AVHRR, grid)
    s_arr = s1.resample_to_grid(SWIR,  grid)
    b4_arr= s1.resample_to_grid(THERM, grid)
    m_arr = s1.resample_to_grid(MODIS, grid, fill=np.nan)

    np.save('a_arr.npy',  a_arr)
    np.save('s_arr.npy',  s_arr)
    np.save('b4_arr.npy', b4_arr)
    np.save('m_arr.npy',  m_arr)
    with open('grid.pkl', 'wb') as f:
        pickle.dump(grid, f)
    print(f'  Grid: {grid["width"]}x{grid["height"]} px')

    # ------------------------------------------------------------------
    # Stage 2: cloud mask
    # ------------------------------------------------------------------
    print('\n[Stage 2] Building thermal cloud mask...')
    cloud, _ = s2.thermal_cloud_mask(b4_arr, threshold=600)
    np.save('cloud_mask.npy', cloud)
    print(f'  Cloud fraction: {cloud[b4_arr>0].mean():.1%}')

    # ------------------------------------------------------------------
    # Stage 3: land/water masks
    # ------------------------------------------------------------------
    print('\n[Stage 3] Building land/water masks...')
    s_valid = s_arr > 0
    m_valid = ~np.isnan(m_arr) & (m_arr > 0)
    s_land  = s3.row_banded_otsu_mask(s_arr, s_valid)
    m_land  = s3.row_banded_otsu_mask(m_arr, m_valid)
    np.save('s_land.npy', s_land)
    np.save('m_land.npy', m_land)
    print(f'  SWIR land frac: {s_land[s_valid].mean():.1%}')
    print(f'  MODIS land frac: {m_land[m_valid].mean():.1%}')

    # ------------------------------------------------------------------
    # Stage 4: dense tile matching (auto candidates)
    # ------------------------------------------------------------------
    print('\n[Stage 4] Dense NCC tile matching...')
    s_land_u8 = cv2.GaussianBlur((s_land.astype(np.uint8)*255), (5,5), 1.0)
    m_land_u8 = cv2.GaussianBlur((m_land.astype(np.uint8)*255), (5,5), 1.0)
    clear = ~cloud
    ta_auto, tm_auto, scores_auto = s4.dense_tile_match(
        s_land_u8, s_valid, m_land_u8, m_valid,
        extra_gate=clear, tile=200, step=100, search=170, score_thresh=0.6
    )
    print(f'  Auto candidates: {len(ta_auto)}')
    # (these are candidates only; final tie points come from Stage 5 GCPs)

    # ------------------------------------------------------------------
    # Stage 5: load GCPs, filter outliers, LOO validation
    # ------------------------------------------------------------------
    print('\n[Stage 5] Loading and validating GCPs from .points file...')
    ta_gcp, off_gcp = s5.load_qgis_points(POINTS)
    print(f'  Loaded: {len(ta_gcp)} points')

    ta_clean, off_clean = s5.filter_outliers(
        ta_gcp, off_gcp, dev_thresh=60, loo_thresh=40
    )
    errs = s5.leave_one_out_rmse(ta_clean, off_clean)
    print(f'  Clean points: {len(ta_clean)}')
    print(f'  LOO RMSE: {np.sqrt((errs**2).mean()):.1f}px  '
          f'median: {np.median(errs):.1f}px  max: {errs.max():.1f}px')
    print(f'  < 10px: {(errs<10).sum()}/{len(errs)} '
          f'({100*(errs<10).mean():.0f}%)')

    np.save('curated_ta.npy',  ta_clean)
    np.save('curated_off.npy', off_clean)

    # ------------------------------------------------------------------
    # Stage 6: TPS warp + crop + mask
    # ------------------------------------------------------------------
    print('\n[Stage 6] Fitting TPS and warping...')
    H, W = a_arr.shape
    dxf, dyf = s6.fit_tps_field(ta_clean, off_clean, W, H)
    warped    = s6.warp_with_field(a_arr, dxf, dyf)
    warped, cutoff_row = s6.mask_no_modis_coverage(warped, m_arr, cutoff_lat=6.0)
    print(f'  Valid coverage: {(warped>0).mean():.1%}')
    print(f'  Output extent: lat 42.07N to 6.0N  '
          f'({warped.shape[1]}x{warped.shape[0]} px)')

    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'nodata': 0,
        'width': W, 'height': cutoff_row,
        'crs': grid['crs'], 'transform': grid['transform'],
    }
    with rasterio.open(OUTPUT, 'w', **profile) as dst:
        dst.write(warped.astype('float32'), 1)
    print(f'\nSaved: {OUTPUT}')
    print('\nDone.')


if __name__ == '__main__':
    main()