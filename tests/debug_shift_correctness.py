"""
Standalone diagnostic for Stage 5 (phase correlation) correctness.

Three checks, run independently of the CSV outputs already on disk:

1. Synthetic circular-shift recovery  - is the FFT/peak/shift-conversion
   math itself correct, on data with no edge effects?
2. Synthetic non-periodic-shift recovery - same test but with a linear
   (non-wrapping) shift, like a real AVHRR/MODIS misregistration would
   produce. Run with and without Hann apodization.
3. Real accepted windows - for a sample of rows from
   validated_integer_shifts.csv, does applying the reported (dx, dy)
   actually raise the normalized cross-correlation between the AVHRR
   and MODIS crops, or not?
"""

import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import shift as nd_shift

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "local_registration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "local_registration", "core"))

from core.phase_engine import PhaseCorrelationEngine

gdal.UseExceptions()

ROOT = os.path.join(os.path.dirname(__file__), "..")


def peak_to_shift(peak_row, peak_col, rows, cols):
    dy = peak_row - rows if peak_row > rows // 2 else peak_row
    dx = peak_col - cols if peak_col > cols // 2 else peak_col
    return dy, dx


def ncc(a, b):
    a = a[np.isfinite(a) & np.isfinite(b)]
    b_full = b
    mask = np.isfinite(b_full)
    a2, b2 = a, b_full[mask][:len(a)]
    if a.size == 0:
        return np.nan
    a = a - a.mean()
    b2 = b2 - b2.mean()
    denom = np.sqrt((a ** 2).sum() * (b2 ** 2).sum())
    if denom == 0:
        return np.nan
    return float((a * b2).sum() / denom)


def normalized_cc(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < a.size * 0.5:
        return np.nan
    av = a[valid] - a[valid].mean()
    bv = b[valid] - b[valid].mean()
    denom = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    if denom == 0:
        return np.nan
    return float((av * bv).sum() / denom)


def hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def check1_circular_recovery(reference_window, engine, trials):
    print("\n" + "=" * 60)
    print("CHECK 1: circular (np.roll) shift recovery")
    print("=" * 60)
    all_ok = True
    for true_dy, true_dx in trials:
        target = np.roll(reference_window, shift=(true_dy, true_dx), axis=(0, 1))
        surface = engine.compute(reference_window, target)
        peak_row, peak_col = np.unravel_index(np.argmax(surface), surface.shape)
        rows, cols = surface.shape
        rec_dy, rec_dx = peak_to_shift(peak_row, peak_col, rows, cols)
        ok = (rec_dy, rec_dx) == (true_dy, true_dx)
        all_ok &= ok
        print(f"true=({true_dy:>4},{true_dx:>4})  recovered=({rec_dy:>4},{rec_dx:>4})  "
              f"peak_value={surface[peak_row, peak_col]:.4f}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n-> circular recovery all correct: {all_ok}")
    return all_ok


def check2_linear_recovery(reference_window, engine, trials):
    print("\n" + "=" * 60)
    print("CHECK 2: linear (edge-truncated) shift recovery, no window vs Hann")
    print("=" * 60)
    win = hann2d(reference_window.shape)
    for true_dy, true_dx in trials:
        target = nd_shift(
            reference_window, shift=(true_dy, true_dx),
            order=1, mode="nearest"
        )

        surface_plain = engine.compute(reference_window, target)
        pr, pc = np.unravel_index(np.argmax(surface_plain), surface_plain.shape)
        rows, cols = surface_plain.shape
        dy_plain, dx_plain = peak_to_shift(pr, pc, rows, cols)

        surface_hann = engine.compute(reference_window * win, target * win)
        pr2, pc2 = np.unravel_index(np.argmax(surface_hann), surface_hann.shape)
        dy_hann, dx_hann = peak_to_shift(pr2, pc2, rows, cols)

        print(f"true=({true_dy:>4},{true_dx:>4})  "
              f"no-window=({dy_plain:>4},{dx_plain:>4}) peak={surface_plain[pr, pc]:.4f}   "
              f"hann=({dy_hann:>4},{dx_hann:>4}) peak={surface_hann[pr2, pc2]:.4f}")


def check3_degenerate_wraparound(reference_window, engine, window_size):
    print("\n" + "=" * 60)
    print("CHECK 3: reproduce the >=window_size wraparound bug")
    print("=" * 60)
    bad_shift = window_size + 10
    shifted = nd_shift(
        reference_window, shift=(0, bad_shift),
        order=0, mode="constant", cval=np.nan, prefilter=False
    )
    nan_fraction = np.isnan(shifted).mean()
    filled = np.nan_to_num(shifted, nan=0.0)
    surface = engine.compute(reference_window, filled)
    pr, pc = np.unravel_index(np.argmax(surface), surface.shape)
    print(f"applied shift={bad_shift} (>= window_size={window_size})")
    print(f"NaN fraction after shift : {nan_fraction:.2f}")
    print(f"resulting peak           : ({pr},{pc})  value={surface[pr, pc]:.6f}")
    print(f"-> trivially 'validated' at (0,0) despite total garbage window: "
          f"{pr == 0 and pc == 0 and abs(surface[pr, pc]) < 1e-6}")


def check4_real_windows(avhrr, modis, df, window_ids):
    print("\n" + "=" * 60)
    print("CHECK 4: real accepted windows - does the reported shift help?")
    print("=" * 60)
    for wid in window_ids:
        row = df[df["window_id"] == wid]
        if row.empty:
            continue
        row = row.iloc[0]
        r0, r1 = int(row.row_start), int(row.row_end)
        c0, c1 = int(row.col_start), int(row.col_end)
        dx, dy = float(row.validated_dx), float(row.validated_dy)

        avhrr_win = avhrr[r0:r1, c0:c1]
        modis_win = modis[r0:r1, c0:c1]

        before = normalized_cc(avhrr_win, modis_win)

        shifted = nd_shift(
            avhrr_win, shift=(dy, dx),
            order=1, mode="constant", cval=np.nan
        )
        after = normalized_cc(shifted, modis_win)

        print(f"window {wid:>4}  dx={dx:>7.1f} dy={dy:>7.1f}  "
              f"NCC before={before:>7.4f}  after={after:>7.4f}  "
              f"{'IMPROVED' if (np.nan_to_num(after) > np.nan_to_num(before)) else 'NOT improved'}  "
              f"validated={bool(row.validated)}  reported_peak={row.peak_value:.5f}")


def main():
    engine = PhaseCorrelationEngine()

    modis_path = os.path.join(ROOT, "2_outputs", "05_modis_float32.tif")
    avhrr_path = os.path.join(ROOT, "2_outputs", "05_avhrr_float32.tif")

    modis_ds = gdal.Open(modis_path)
    avhrr_ds = gdal.Open(avhrr_path)

    modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    r0, c0 = 384, 1536
    size = 512
    reference_window = np.nan_to_num(modis[r0:r0 + size, c0:c0 + size])

    trials = [(0, 0), (5, -3), (-40, 12), (100, -80)]
    check1_circular_recovery(reference_window, engine, trials)
    check2_linear_recovery(reference_window, engine, trials)
    check3_degenerate_wraparound(reference_window, engine, size)

    csv_path = os.path.join(
        ROOT, "stage5_phase_correlation", "stage5_shift_validation",
        "validated_integer_shifts.csv"
    )
    df = pd.read_csv(csv_path)

    big_shift_ids = df[(df["validated"] == True) &
                        ((df["validated_dx"].abs() >= 512) | (df["validated_dy"].abs() >= 512))
                        ]["window_id"].head(3).tolist()

    zero_shift_ids = df[(df["validated"] == True) &
                         (df["validated_dx"] == 0) & (df["validated_dy"] == 0)
                         ]["window_id"].head(3).tolist()

    unvalidated_ids = df[df["validated"] == False]["window_id"].head(3).tolist()

    check4_real_windows(avhrr, modis, df, big_shift_ids + zero_shift_ids + unvalidated_ids)


if __name__ == "__main__":
    main()
