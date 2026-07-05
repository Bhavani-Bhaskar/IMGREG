"""
utils.py

Purpose
-------
Common utility functions used throughout the preprocessing package.

Author : Bhaskar
"""

import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


# ----------------------------------------------------------
# Debug Printing
# ----------------------------------------------------------

def print_header(title):
    """Print a formatted section header."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------
# File Checking
# ----------------------------------------------------------

def check_file_exists(filename):
    """Raise an error if the file does not exist."""

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"\nFile not found:\n{filename}"
        )


# ----------------------------------------------------------
# Open Raster
# ----------------------------------------------------------

def open_raster(filename):
    """
    Open a raster safely.

    Returns
    -------
    dataset
    """

    check_file_exists(filename)

    dataset = gdal.Open(filename)

    if dataset is None:
        raise RuntimeError(
            f"Unable to open raster:\n{filename}"
        )

    return dataset


# ----------------------------------------------------------
# Image Shape
# ----------------------------------------------------------

def get_shape(dataset):
    """
    Return raster shape.

    Returns
    -------
    (rows, cols)
    """

    return (
        dataset.RasterYSize,
        dataset.RasterXSize
    )


# ----------------------------------------------------------
# Pixel Size
# ----------------------------------------------------------

def get_pixel_size(dataset):
    """
    Returns
    -------
    (pixel_width, pixel_height)
    """

    gt = dataset.GetGeoTransform()

    return (
        abs(gt[1]),
        abs(gt[5])
    )


# ----------------------------------------------------------
# Raster Extent
# ----------------------------------------------------------

def get_extent(dataset):
    """
    Returns
    -------
    xmin, ymin, xmax, ymax
    """

    gt = dataset.GetGeoTransform()

    xmin = gt[0]

    ymax = gt[3]

    xmax = xmin + gt[1] * dataset.RasterXSize

    ymin = ymax + gt[5] * dataset.RasterYSize

    return (
        xmin,
        ymin,
        xmax,
        ymax
    )


# ----------------------------------------------------------
# Compare Shapes
# ----------------------------------------------------------

def same_shape(ds1, ds2):
    """
    Compare raster sizes.
    """

    return get_shape(ds1) == get_shape(ds2)


# ----------------------------------------------------------
# Compare GeoTransforms
# ----------------------------------------------------------

def same_geotransform(
    ds1,
    ds2,
    tolerance=1e-8
):
    """
    Compare geotransforms with tolerance.
    """

    return np.allclose(
        ds1.GetGeoTransform(),
        ds2.GetGeoTransform(),
        atol=tolerance
    )


# ----------------------------------------------------------
# Compare Projection
# ----------------------------------------------------------

def same_projection(ds1, ds2):
    """
    Compare CRS.
    """

    return (
        ds1.GetProjection()
        ==
        ds2.GetProjection()
    )


# ----------------------------------------------------------
# Data Type Name
# ----------------------------------------------------------

def get_dtype(dataset):

    band = dataset.GetRasterBand(1)

    return gdal.GetDataTypeName(
        band.DataType
    )


# ----------------------------------------------------------
# Print Raster Information
# ----------------------------------------------------------

def print_raster_info(dataset):

    gt = dataset.GetGeoTransform()

    print(f"Rows        : {dataset.RasterYSize}")
    print(f"Columns     : {dataset.RasterXSize}")
    print(f"Pixel Size  : ({abs(gt[1])}, {abs(gt[5])})")
    print(f"Bands       : {dataset.RasterCount}")

    band = dataset.GetRasterBand(1)

    print(f"Datatype    : {gdal.GetDataTypeName(band.DataType)}")

    print(f"NoData      : {band.GetNoDataValue()}")


def save_raster(dataset, output_file):
    """
    Save an in-memory GDAL dataset to a GeoTIFF.

    Parameters
    ----------
    dataset : gdal.Dataset
        In-memory dataset.

    output_file : str
        Output GeoTIFF path.
    """
    driver = gdal.GetDriverByName("GTiff")
    driver.CreateCopy(output_file, dataset, strict=0)
