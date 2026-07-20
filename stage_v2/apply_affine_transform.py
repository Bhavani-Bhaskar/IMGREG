"""
Stage 8 (Transformation Modeling) + Stage 10 (Geometric
Correction), applied explicitly and transparently on top of the
v2 (AROSICS) tie points - rather than relying only on AROSICS'
own internal (locally-varying) correction in avhrr_corrected.tif.

Fits a single global 2D affine transform (6 parameters, least
squares) from the 18 valid tie points, then resamples the full
AVHRR raster onto the MODIS grid using that one transform, with
a choice of bilinear or cubic interpolation.

Sign convention (verified empirically against AROSICS' own
avhrr_corrected.tif, not assumed): a tie point at source pixel
(X_IM, Y_IM) with shift (X_SHIFT_PX, Y_SHIFT_PX) belongs at
destination (X_IM + X_SHIFT_PX, Y_IM + Y_SHIFT_PX). Confirmed by
comparing scipy.ndimage.shift(avhrr, +shift) vs (-shift) against
AROSICS' own corrected crop at tie point 420: MSE 3994 for +shift
vs 10712 for -shift.

Outputs
-------
stage_v2/avhrr_affine_bilinear.tif
stage_v2/avhrr_affine_cubic.tif
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import affine_transform

gdal.UseExceptions()

ROOT = os.path.join(os.path.dirname(__file__), "..")

AVHRR_FILE = os.path.join(ROOT, "2_outputs", "05_avhrr_float32.tif")
MODIS_FILE = os.path.join(ROOT, "2_outputs", "05_modis_float32.tif")
TIE_POINTS_CSV = os.path.join(ROOT, "stage_v2", "tie_points.csv")

OUTPUT_BILINEAR = os.path.join(ROOT, "stage_v2", "avhrr_affine_bilinear.tif")
OUTPUT_CUBIC = os.path.join(ROOT, "stage_v2", "avhrr_affine_cubic.tif")


def fit_affine(src_rc, dst_rc):
    """
    Least-squares fit of a 2D affine transform mapping source
    (row, col) points to destination (row, col) points.

    dst = M @ [row, col, 1]

    Returns
    -------
    M : ndarray, shape (2, 3)
    residuals : ndarray, shape (n, 2)
        Per-point (row, col) fit residual, for reporting.
    """

    n = src_rc.shape[0]

    design = np.column_stack([src_rc, np.ones(n)])  # (n, 3)

    M = np.zeros((2, 3))

    for axis in (0, 1):
        coeffs, _, _, _ = np.linalg.lstsq(design, dst_rc[:, axis], rcond=None)
        M[axis, :] = coeffs

    predicted = design @ M.T

    residuals = predicted - dst_rc

    return M, residuals


def invert_affine(M):
    """
    Invert a forward (src -> dst) 2x3 affine matrix into the
    (dst -> src) matrix + offset that scipy.ndimage.affine_transform
    expects (it pulls samples: output[coord] = input[matrix @
    coord + offset]).
    """

    homogeneous = np.vstack([M, [0, 0, 1]])

    inverse = np.linalg.inv(homogeneous)

    matrix = inverse[:2, :2]
    offset = inverse[:2, 2]

    return matrix, offset


def warp(avhrr, matrix, offset, output_shape, order):

    return affine_transform(
        avhrr,
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=np.nan,
    )


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

    df = pd.read_csv(TIE_POINTS_CSV)
    valid = df[df["OUTLIER"] == "False"].reset_index(drop=True)

    print(f"Fitting affine transform from {len(valid)} valid tie points...")

    src_rc = valid[["Y_IM", "X_IM"]].to_numpy(dtype=np.float64)

    dst_rc = src_rc + valid[["Y_SHIFT_PX", "X_SHIFT_PX"]].to_numpy(dtype=np.float64)

    M, residuals = fit_affine(src_rc, dst_rc)

    rmse_row = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
    rmse_col = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
    rmse_total = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    print("\nFitted affine (row, col in pixels):")
    print(f"  row' = {M[0, 0]:.6f}*row + {M[0, 1]:.6f}*col + {M[0, 2]:.4f}")
    print(f"  col' = {M[1, 0]:.6f}*row + {M[1, 1]:.6f}*col + {M[1, 2]:.4f}")
    print(f"\nFit residuals (tie points vs. affine prediction):")
    print(f"  RMSE row : {rmse_row:.3f} px")
    print(f"  RMSE col : {rmse_col:.3f} px")
    print(f"  RMSE total : {rmse_total:.3f} px")

    matrix, offset = invert_affine(M)

    avhrr_ds = gdal.Open(AVHRR_FILE)
    modis_ds = gdal.Open(MODIS_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    output_shape = (modis_ds.RasterYSize, modis_ds.RasterXSize)

    print("\nWarping (bilinear, order=1)...")
    bilinear = warp(avhrr, matrix, offset, output_shape, order=1)
    save_geotiff(bilinear, modis_ds, OUTPUT_BILINEAR)
    print(f"  Saved: {OUTPUT_BILINEAR}")

    print("Warping (cubic, order=3)...")
    cubic = warp(avhrr, matrix, offset, output_shape, order=3)
    save_geotiff(cubic, modis_ds, OUTPUT_CUBIC)
    print(f"  Saved: {OUTPUT_CUBIC}")


if __name__ == "__main__":
    main()
