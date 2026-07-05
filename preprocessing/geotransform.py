"""
geotransform.py
---------------

Verify that two rasters are on the exact same pixel grid.

This module DOES NOT modify the images.

It simply checks whether the two rasters have

    - Same Shape
    - Same GeoTransform
    - Same Projection

If any check fails, an exception is raised.

Author : Bhaskar Project
"""

import numpy as np
from osgeo import gdal
import os

gdal.UseExceptions()


def verify_geotransform(
    reference_file,
    target_file,
    tolerance=1e-8,
    debug=True,
):
    """
    Verify that two rasters share the same pixel grid.

    Parameters
    ----------
    reference_file : str

    target_file : str

    tolerance : float

    debug : bool

    Returns
    -------
    bool
        True if everything matches.
    """

    if debug:
        print("\n========== Verify Pixel Grid ==========")

    if not os.path.exists(reference_file):
        raise FileNotFoundError(reference_file)

    if not os.path.exists(target_file):
        raise FileNotFoundError(target_file)

    ref = gdal.Open(reference_file)
    tar = gdal.Open(target_file)

    if ref is None:
        raise RuntimeError("Cannot open reference raster.")

    if tar is None:
        raise RuntimeError("Cannot open target raster.")

    # --------------------------------------------------
    # Shape
    # --------------------------------------------------

    ref_shape = (
        ref.RasterYSize,
        ref.RasterXSize
    )

    tar_shape = (
        tar.RasterYSize,
        tar.RasterXSize
    )

    shape_ok = ref_shape == tar_shape

    # --------------------------------------------------
    # Projection
    # --------------------------------------------------

    projection_ok = (
        ref.GetProjection() ==
        tar.GetProjection()
    )

    # --------------------------------------------------
    # GeoTransform
    # --------------------------------------------------

    ref_gt = ref.GetGeoTransform()

    tar_gt = tar.GetGeoTransform()

    gt_ok = np.allclose(
        ref_gt,
        tar_gt,
        atol=tolerance
    )

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    if debug:

        print("Reference Shape :", ref_shape)
        print("Target Shape    :", tar_shape)

        print()

        print("Reference GeoTransform")
        print(ref_gt)

        print()

        print("Target GeoTransform")
        print(tar_gt)

        print()

        print("Shape Match        :", shape_ok)
        print("Projection Match   :", projection_ok)
        print("GeoTransform Match :", gt_ok)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if not shape_ok:

        raise RuntimeError(
            "\nRaster dimensions do not match."
        )

    if not projection_ok:

        raise RuntimeError(
            "\nRaster projections do not match."
        )

    if not gt_ok:

        raise RuntimeError(
            "\nGeoTransforms do not match."
        )

    if debug:

        print("\nVerification successful.")
        print("Both rasters are on the same pixel grid.")

    ref = None
    tar = None

    return True