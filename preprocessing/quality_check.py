"""
quality_check.py
----------------

Verify that preprocessing has completed successfully before
proceeding to Stage 3.

Checks
------
1. Shape
2. Projection
3. GeoTransform
4. Pixel Size
5. Data Type
6. Band Count
7. Mask Alignment
8. Mask Coverage

Author : Bhaskar Project
"""

import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def quality_check(
    reference_file,
    target_file,
    mask_file,
    tolerance=1e-8,
    debug=True,
):
    """
    Verify preprocessing outputs.

    Parameters
    ----------
    reference_file : str

    target_file : str

    mask_file : str

    tolerance : float

    debug : bool

    Returns
    -------
    bool
    """

    if debug:
        print("\n")
        print("=" * 60)
        print("PREPROCESSING QUALITY CHECK")
        print("=" * 60)

    # --------------------------------------------------
    # File existence
    # --------------------------------------------------

    for f in [reference_file, target_file, mask_file]:

        if not os.path.exists(f):
            raise FileNotFoundError(f)

    ref = gdal.Open(reference_file)
    tar = gdal.Open(target_file)
    mask = gdal.Open(mask_file)

    if ref is None:
        raise RuntimeError("Cannot open reference raster.")

    if tar is None:
        raise RuntimeError("Cannot open target raster.")

    if mask is None:
        raise RuntimeError("Cannot open mask raster.")

    # --------------------------------------------------
    # Shape
    # --------------------------------------------------

    ref_shape = (ref.RasterYSize, ref.RasterXSize)
    tar_shape = (tar.RasterYSize, tar.RasterXSize)
    mask_shape = (mask.RasterYSize, mask.RasterXSize)

    shape_ok = (
        ref_shape == tar_shape == mask_shape
    )

    # --------------------------------------------------
    # Projection
    # --------------------------------------------------

    projection_ok = (
        ref.GetProjection() ==
        tar.GetProjection() ==
        mask.GetProjection()
    )

    # --------------------------------------------------
    # GeoTransform
    # --------------------------------------------------

    gt_ref = ref.GetGeoTransform()
    gt_tar = tar.GetGeoTransform()
    gt_mask = mask.GetGeoTransform()

    geotransform_ok = (

        np.allclose(gt_ref, gt_tar, atol=tolerance)

        and

        np.allclose(gt_ref, gt_mask, atol=tolerance)

    )

    # --------------------------------------------------
    # Pixel Size
    # --------------------------------------------------

    pixel_ok = (

        abs(gt_ref[1] - gt_tar[1]) < tolerance

        and

        abs(gt_ref[5] - gt_tar[5]) < tolerance

    )

    # --------------------------------------------------
    # Datatype
    # --------------------------------------------------

    ref_dtype = gdal.GetDataTypeName(
        ref.GetRasterBand(1).DataType
    )

    tar_dtype = gdal.GetDataTypeName(
        tar.GetRasterBand(1).DataType
    )

    datatype_ok = (

        ref_dtype == "Float32"

        and

        tar_dtype == "Float32"

    )

    # --------------------------------------------------
    # Band Count
    # --------------------------------------------------

    bands_ok = (

        ref.RasterCount == 1

        and

        tar.RasterCount == 1

        and

        mask.RasterCount == 1

    )

    # --------------------------------------------------
    # Mask
    # --------------------------------------------------

    mask_array = mask.GetRasterBand(1).ReadAsArray()

    valid_pixels = np.count_nonzero(mask_array)

    total_pixels = mask_array.size

    coverage = 100 * valid_pixels / total_pixels

    mask_values = np.unique(mask_array)

    mask_ok = set(mask_values).issubset({0, 1})

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    if debug:

        print("\nReference")
        print("--------------------------")
        print("Shape      :", ref_shape)
        print("Datatype   :", ref_dtype)

        print("\nTarget")
        print("--------------------------")
        print("Shape      :", tar_shape)
        print("Datatype   :", tar_dtype)

        print("\nMask")
        print("--------------------------")
        print("Shape      :", mask_shape)
        print("Coverage   : %.2f%%" % coverage)
        print("Values     :", mask_values)

        print("\nVerification")
        print("--------------------------")
        print("Shape           :", shape_ok)
        print("Projection      :", projection_ok)
        print("GeoTransform    :", geotransform_ok)
        print("Pixel Size      :", pixel_ok)
        print("Datatype        :", datatype_ok)
        print("Band Count      :", bands_ok)
        print("Binary Mask     :", mask_ok)

    # --------------------------------------------------
    # Final Decision
    # --------------------------------------------------

    passed = all([

        shape_ok,

        projection_ok,

        geotransform_ok,

        pixel_ok,

        datatype_ok,

        bands_ok,

        mask_ok

    ])

    print("\n")

    if passed:

        print("=" * 60)
        print("QUALITY CHECK PASSED")
        print("READY FOR STAGE 3")
        print("=" * 60)

    else:

        print("=" * 60)
        print("QUALITY CHECK FAILED")
        print("DO NOT CONTINUE TO STAGE 3")
        print("=" * 60)

        raise RuntimeError(
            "Preprocessing quality check failed."
        )

    ref = None
    tar = None
    mask = None

    return True
