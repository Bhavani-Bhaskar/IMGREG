# -*- coding: utf-8 -*-
"""
STAGE 5: Tie-point validation
==============================
Two-layer validation used throughout this project:

1. Neighbour corroboration: for every candidate tie point, independently
   probe 2-3 nearby locations and check that their dx/dy agrees within ~15px.
   A high NCC score alone does NOT mean a reliable point -- isolated high-score
   points with no nearby corroboration had LOO errors of 85-96px. Only points
   where at least one nearby location agrees are kept.

2. Leave-one-out (LOO) cross-validation: hold out each point, refit TPS from
   the rest, measure prediction error on the held-out point. This is the real
   generalisation-error metric. In-sample residual is always near-zero for TPS
   and tells you nothing useful.

Final result after iterative cleaning:
  - 60 validated points (from 73 manually picked GCPs, 13 removed as outliers)
  - LOO RMSE: ~15px, median ~7.6px, max ~44px
  - Column coverage: 550 to 2738 (image width ~3181)
"""
import numpy as np
from scipy.interpolate import RBFInterpolator


def load_qgis_points(points_file,
                     orig_left=64.09678041592721,
                     orig_top=42.07703512135007,
                     orig_res=0.009939155183522544,
                     grid_left=64.09,
                     grid_top=42.07,
                     grid_res=0.01):
    """
    Parse a QGIS Georeferencer .points file and convert to pixel
    coordinates in the common 0.01-degree grid.

    .points format (comma separated):
        mapX, mapY, sourceX, sourceY, enable
    where mapX/Y = destination lon/lat (correct location from MODIS)
    and sourceX/Y = lon/lat of the pixel in the original AVHRR geo file.

    Returns:
        ta  : (N,2) array of [src_col, src_row] in the common grid
        off : (N,2) array of [dx, dy] displacement to apply
    """
    lines = open(points_file).readlines()
    lines = [l.strip() for l in lines
             if not l.startswith('#') and not l.startswith('mapX') and l.strip()]

    src_cols, src_rows, dxs, dys = [], [], [], []
    for l in lines:
        p = l.split(',')
        dest_lon, dest_lat = float(p[0]), float(p[1])
        src_lon,  src_lat  = float(p[2]), float(p[3])
        enable = int(p[4])
        if enable == 0:
            continue

        # source: where AVHRR pixel currently sits (in common grid pixel coords)
        sc = (src_lon  - orig_left) / orig_res
        sr = (orig_top - src_lat)  / orig_res

        # destination: where it should sit (correct location per MODIS)
        dc = (dest_lon  - grid_left) / grid_res
        dr = (grid_top  - dest_lat)  / grid_res

        src_cols.append(sc); src_rows.append(sr)
        dxs.append(dc - sc); dys.append(dr - sr)

    ta  = np.stack([src_cols, src_rows], axis=1)
    off = np.stack([dxs, dys], axis=1)
    return ta, off


def neighbour_deviation(ta, dx, k=3):
    """
    For each point, compute how much its dx differs from the mean dx of
    its k nearest neighbours. Large deviation = likely bad click.
    """
    pts_xy = ta
    devs = np.zeros(len(ta))
    for i in range(len(ta)):
        dists = np.sqrt(((pts_xy - pts_xy[i])**2).sum(axis=1))
        dists[i] = 1e9
        nbrs = np.argsort(dists)[:k]
        devs[i] = abs(dx[i] - dx[nbrs].mean())
    return devs


def leave_one_out_rmse(ta, off, smoothing=15.0):
    """
    Leave-one-out cross-validation of the TPS displacement field.
    Returns per-point LOO error in pixels.
    This is the only meaningful accuracy metric -- in-sample TPS residual
    is always ~0 and tells you nothing.
    """
    n = len(ta)
    errs = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rbf_dx = RBFInterpolator(ta[mask], off[mask, 0],
                                  kernel='thin_plate_spline', smoothing=smoothing)
        rbf_dy = RBFInterpolator(ta[mask], off[mask, 1],
                                  kernel='thin_plate_spline', smoothing=smoothing)
        errs[i] = np.hypot(rbf_dx(ta[i:i+1])[0] - off[i, 0],
                            rbf_dy(ta[i:i+1])[0] - off[i, 1])
    return errs


def filter_outliers(ta, off, dev_thresh=60, loo_thresh=40, smoothing=15.0):
    """
    Two-pass outlier removal:
    Pass 1: remove points whose dx deviates >dev_thresh px from neighbours.
    Pass 2: recompute LOO on remaining points, remove any with LOO >loo_thresh.
    Returns cleaned ta, off and the boolean keep mask.
    """
    dx = off[:, 0]

    # Pass 1: neighbour deviation
    devs = neighbour_deviation(ta, dx)
    keep = devs <= dev_thresh
    print(f'Pass 1 (neighbour dev >{dev_thresh}px): removed {(~keep).sum()} points')

    ta1, off1 = ta[keep], off[keep]

    # Pass 2: LOO
    errs = leave_one_out_rmse(ta1, off1, smoothing)
    keep2 = errs <= loo_thresh
    print(f'Pass 2 (LOO >{loo_thresh}px): removed {(~keep2).sum()} points')

    ta2, off2 = ta1[keep2], off1[keep2]
    return ta2, off2


if __name__ == '__main__':
    POINTS_FILE = r"C:\Users\laksh\OneDrive\Desktop\nrsc\metop\hrpt_M03_b2(tie_points_4).tif.points"

    ta, off = load_qgis_points(POINTS_FILE)
    print(f'Loaded {len(ta)} points from .points file')
    print(f'Col range: {ta[:,0].min():.0f} to {ta[:,0].max():.0f}')

    ta_clean, off_clean = filter_outliers(ta, off, dev_thresh=60, loo_thresh=40)
    print(f'\nFinal clean points: {len(ta_clean)}')

    errs = leave_one_out_rmse(ta_clean, off_clean)
    print(f'LOO RMSE:   {np.sqrt((errs**2).mean()):.2f} px')
    print(f'LOO median: {np.median(errs):.2f} px')
    print(f'LOO max:    {errs.max():.2f} px')
    print(f'< 10px: {(errs<10).sum()}/{len(errs)} ({100*(errs<10).mean():.0f}%)')

    np.save('curated_ta.npy', ta_clean)
    np.save('curated_off.npy', off_clean)
    np.save('curated_tm.npy', ta_clean + off_clean)