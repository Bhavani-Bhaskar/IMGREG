"""
valid_mask.py
-------------

Create a binary mask of valid pixels.

Output
------
1 = Valid pixel
0 = Invalid / background

Author : Bhaskar Project
"""

import os
import cv2
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def create_valid_mask(
    input_file,
    output_file,
    threshold=0,
    kernel_size=5,
    debug=True,
):
    """
    Create a binary validity mask.

    Parameters
    ----------
    input_file : str

    output_file : str

    threshold : float
        Used only when neither NoData nor NaNs exist.

    kernel_size : int

    debug : bool

    Returns
    -------
    str
        Output mask filename.
    """

    if debug:
        print("\n========== Create Valid Mask ==========")

    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    ds = gdal.Open(input_file)

    if ds is None:
        raise RuntimeError("Cannot open raster.")

    band = ds.GetRasterBand(1)

    image = band.ReadAsArray()

    nodata = band.GetNoDataValue()

    # -------------------------------------------------------
    # Create Initial Binary Mask
    # -------------------------------------------------------

    if nodata is not None:

        if debug:
            print(f"Using NoData value : {nodata}")

        binary = (image != nodata)

    elif np.isnan(image).any():

        if debug:
            print("Using NaN values")

        binary = ~np.isnan(image)

    else:

        if debug:
            print(f"Using Threshold : {threshold}")

        binary = image > threshold

    binary = binary.astype(np.uint8)

    initial_valid = np.count_nonzero(binary)

    if debug:

        print("\nInitial Statistics")
        print("----------------------------")
        print(f"Total Pixels : {binary.size}")
        print(f"Valid Pixels : {initial_valid}")
        print(
            f"Coverage     : {100*initial_valid/binary.size:.2f}%"
        )

    # -------------------------------------------------------
    # Largest Connected Component
    # -------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if num_labels <= 1:
        raise RuntimeError("No connected component found.")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    mask = (labels == largest).astype(np.uint8)

    if debug:

        print("\nConnected Components")
        print("----------------------------")
        print(f"Number Found : {num_labels-1}")

        print(
            "Largest Size :",
            stats[largest, cv2.CC_STAT_AREA]
        )

    # -------------------------------------------------------
    # Morphological Closing
    # -------------------------------------------------------

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    valid_pixels = np.count_nonzero(mask)

    total_pixels = mask.size

    coverage = valid_pixels / total_pixels

    if debug:

        print("\nFinal Statistics")
        print("----------------------------")
        print(f"Valid Pixels : {valid_pixels}")
        print(f"Coverage     : {coverage*100:.2f}%")

    # -------------------------------------------------------
    # Save Mask
    # -------------------------------------------------------

    driver = gdal.GetDriverByName("GTiff")

    out = driver.Create(

        output_file,

        ds.RasterXSize,

        ds.RasterYSize,

        1,

        gdal.GDT_Byte

    )

    out.SetProjection(
        ds.GetProjection()
    )

    out.SetGeoTransform(
        ds.GetGeoTransform()
    )

    out_band = out.GetRasterBand(1)

    out_band.WriteArray(mask)

    out_band.SetNoDataValue(0)

    out_band.FlushCache()

    out.FlushCache()

    out = None
    ds = None

    if debug:

        print("\nVerification")
        print("----------------------------")
        print(f"Output File : {output_file}")
        print("Mask created successfully.")

    return output_file