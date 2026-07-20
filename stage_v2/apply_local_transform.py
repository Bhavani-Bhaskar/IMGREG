"""
Stage 8 (Transformation Modeling) + Stage 10 (Geometric
Correction), local/spatially-varying version - built from ALL
625 independently-verified tie points (stage_v2/
tie_points_verified.csv), not a single global affine.

Why not one global affine: already tested (apply_affine_transform.py)
with the earlier 18-tie-point set and found it dilutes real local
corrections - 3 points needing a 15-42px correction got averaged
down to ~3px by the other 15 near-zero points. With 625 points
spanning most of the scene, the right model is a smoothly-varying
shift FIELD, not one fixed matrix - exactly what context.md's
Stage 8 calls a "continuous shift surface".

Method: interpolate the 625 tie points' (dx, dy) into a dense
per-pixel shift field (linear interpolation inside the tie-point
hull, nearest-neighbor fill outside it so there are no gaps), then
resample the whole AVHRR raster through that field in one pass -
this is what "mosaicing every tile's local correction into one
image" means in practice: every output pixel gets its own locally
appropriate correction, blended continuously rather than as
separate tiles with visible seams.

Resampling: bilinear (order=1) by default; cubic (order=3) also
produced for comparison.

Outputs
-------
stage_v2/avhrr_local_bilinear.tif
stage_v2/avhrr_local_cubic.tif
stage_v2/shift_field_dx.tif / _dy.tif  (diagnostic - the fitted field itself)
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

gdal.UseExceptions()

ROOT = os.path.join(os.path.dirname(__file__), "..")

AVHRR_FILE = os.path.join(ROOT, "2_outputs", "05_avhrr_float32.tif")
MODIS_FILE = os.path.join(ROOT, "2_outputs", "05_modis_float32.tif")
TIE_POINTS_CSV = os.path.join(ROOT, "stage_v2", "tie_points_verified.csv")

OUTPUT_BILINEAR = os.path.join(ROOT, "stage_v2", "avhrr_local_bilinear.tif")
OUTPUT_CUBIC = os.path.join(ROOT, "stage_v2", "avhrr_local_cubic.tif")
OUTPUT_TILED = os.path.join(ROOT, "stage_v2", "avhrr_local_tiled.tif")
SHIFT_FIELD_DX = os.path.join(ROOT, "stage_v2", "shift_field_dx.tif")
SHIFT_FIELD_DY = os.path.join(ROOT, "stage_v2", "shift_field_dy.tif")


def build_dense_shift_field(tie_points, shape, method="linear"):
    """
    Interpolate the sparse (X_IM, Y_IM, X_SHIFT_PX, Y_SHIFT_PX)
    tie points into dense dx/dy fields covering the whole raster.

    method="linear": smooth blend between neighboring tie points.
    Seam-free, but when nearby points disagree sharply it dilutes
    each one's own correction toward their average - checked
    empirically: NCC at a tie point's own location dropped from
    0.563 (its own isolated correction) to 0.122 (the blended
    field) in the worst case found.

    method="nearest": each pixel takes its single nearest tie
    point's exact shift ("mosaic of tiles" - literally a Voronoi
    tiling by nearest tie point). Preserves every point's own
    correction exactly; the cost is visible seams where
    neighboring tiles disagree.

    Either way, nearest-neighbor fill covers pixels outside the
    tie points' convex hull (where "linear" would otherwise be
    undefined/NaN).
    """

    points = tie_points[["Y_IM", "X_IM"]].to_numpy(dtype=np.float64)

    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]

    grid_points = np.column_stack([rows.ravel(), cols.ravel()])

    dx_nearest = griddata(
        points, tie_points["X_SHIFT_PX"].to_numpy(), grid_points, method="nearest"
    ).reshape(shape)

    dy_nearest = griddata(
        points, tie_points["Y_SHIFT_PX"].to_numpy(), grid_points, method="nearest"
    ).reshape(shape)

    if method == "nearest":
        return dx_nearest.astype(np.float32), dy_nearest.astype(np.float32)

    dx_linear = griddata(
        points, tie_points["X_SHIFT_PX"].to_numpy(), grid_points, method="linear"
    ).reshape(shape)

    dy_linear = griddata(
        points, tie_points["Y_SHIFT_PX"].to_numpy(), grid_points, method="linear"
    ).reshape(shape)

    dx = np.where(np.isnan(dx_linear), dx_nearest, dx_linear)
    dy = np.where(np.isnan(dy_linear), dy_nearest, dy_linear)

    return dx.astype(np.float32), dy.astype(np.float32)


def warp_with_field(avhrr, dx_field, dy_field, order):
    """
    For every output pixel, sample the source AVHRR at
    (row - dy_field, col - dx_field) - the inverse of the
    forward destination = source + shift convention verified
    earlier against AROSICS' own corrected raster.
    """

    rows, cols = np.mgrid[0:dx_field.shape[0], 0:dx_field.shape[1]].astype(np.float32)

    src_rows = rows - dy_field
    src_cols = cols - dx_field

    warped = map_coordinates(
        avhrr,
        [src_rows, src_cols],
        order=order,
        mode="constant",
        cval=np.nan,
    )

    return warped


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

    tie_points = pd.read_csv(TIE_POINTS_CSV)

    print(f"Building dense shift field from {len(tie_points)} verified tie points...")

    avhrr_ds = gdal.Open(AVHRR_FILE)
    modis_ds = gdal.Open(MODIS_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    output_shape = (modis_ds.RasterYSize, modis_ds.RasterXSize)

    dx_field, dy_field = build_dense_shift_field(tie_points, output_shape, method="linear")

    print(f"  dx field (linear): mean={dx_field.mean():.3f}px  min={dx_field.min():.3f}  max={dx_field.max():.3f}")
    print(f"  dy field (linear): mean={dy_field.mean():.3f}px  min={dy_field.min():.3f}  max={dy_field.max():.3f}")

    save_geotiff(dx_field, modis_ds, SHIFT_FIELD_DX)
    save_geotiff(dy_field, modis_ds, SHIFT_FIELD_DY)

    print("\nWarping (bilinear, order=1, smooth field)...")
    bilinear = warp_with_field(avhrr, dx_field, dy_field, order=1)
    save_geotiff(bilinear, modis_ds, OUTPUT_BILINEAR)
    print(f"  Saved: {OUTPUT_BILINEAR}")

    print("Warping (cubic, order=3, smooth field)...")
    cubic = warp_with_field(avhrr, dx_field, dy_field, order=3)
    save_geotiff(cubic, modis_ds, OUTPUT_CUBIC)
    print(f"  Saved: {OUTPUT_CUBIC}")

    print("\nBuilding tiled (nearest-neighbor / Voronoi) shift field...")
    dx_tiled, dy_tiled = build_dense_shift_field(tie_points, output_shape, method="nearest")

    print("Warping (bilinear resampling, tiled/nearest shift field)...")
    tiled = warp_with_field(avhrr, dx_tiled, dy_tiled, order=1)
    save_geotiff(tiled, modis_ds, OUTPUT_TILED)
    print(f"  Saved: {OUTPUT_TILED}")


if __name__ == "__main__":
    main()
