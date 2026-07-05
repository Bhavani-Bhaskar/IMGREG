"""
overlap.py

Purpose
-------
Extract the common overlap while preserving the reference
raster pixel grid.

Author : Bhaskar
"""

from osgeo import gdal
import math
import os

gdal.UseExceptions()


def extract_common_overlap(
    reference_file,
    target_file,
    reference_output,
    target_output,
    resampling="bilinear",
    debug=True,
):
    """
    Extract the common overlap while preserving the reference
    raster pixel grid.

    The output rasters have

    - same CRS
    - same origin
    - same pixel size
    - same shape

    but contain ONLY the common overlap.
    """

    if debug:
        print("\n========== Extract Common Overlap ==========")

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

    # ---------------------------------------------------
    # Reference information
    # ---------------------------------------------------

    gt = ref.GetGeoTransform()

    ref_xmin = gt[0]
    ref_ymax = gt[3]

    pixel_x = gt[1]
    pixel_y = abs(gt[5])

    ref_cols = ref.RasterXSize
    ref_rows = ref.RasterYSize

    ref_xmax = ref_xmin + ref_cols * pixel_x
    ref_ymin = ref_ymax - ref_rows * pixel_y

    # ---------------------------------------------------
    # Target extent
    # ---------------------------------------------------

    gt = tar.GetGeoTransform()

    tar_xmin = gt[0]
    tar_ymax = gt[3]

    tar_cols = tar.RasterXSize
    tar_rows = tar.RasterYSize

    tar_xmax = tar_xmin + tar_cols * gt[1]
    tar_ymin = tar_ymax + tar_rows * gt[5]

    # ---------------------------------------------------
    # Floating-point overlap
    # ---------------------------------------------------

    xmin = max(ref_xmin, tar_xmin)
    xmax = min(ref_xmax, tar_xmax)

    ymin = max(ref_ymin, tar_ymin)
    ymax = min(ref_ymax, tar_ymax)

    if xmin >= xmax or ymin >= ymax:
        raise RuntimeError("No overlap found.")

    # ---------------------------------------------------
    # SNAP TO REFERENCE GRID
    # ---------------------------------------------------
    
    # Convert overlap coordinates to the nearest reference
    # pixel indices.
    
    col0 = round((xmin - ref_xmin) / pixel_x)
    col1 = round((xmax - ref_xmin) / pixel_x)
    
    row0 = round((ref_ymax - ymax) / pixel_y)
    row1 = round((ref_ymax - ymin) / pixel_y)
    
    # Convert the snapped pixel indices back to map coordinates.
    
    xmin = ref_xmin + col0 * pixel_x
    xmax = ref_xmin + col1 * pixel_x
    
    ymax = ref_ymax - row0 * pixel_y
    ymin = ref_ymax - row1 * pixel_y
    
    # Compute output image size.
    
    cols = col1 - col0
    rows = row1 - row0
    
    if cols <= 0 or rows <= 0:
        raise RuntimeError("Invalid overlap dimensions after snapping.")

    if debug:

        print("\nReference Grid")
        print("----------------------------")
        print(f"Pixel Size : ({pixel_x}, {-pixel_y})")

        print("\nOverlap")
        print("----------------------------")
        print(f"Rows    : {rows}")
        print(f"Columns : {cols}")

        print(f"Xmin    : {xmin}")
        print(f"Xmax    : {xmax}")
        print(f"Ymin    : {ymin}")
        print(f"Ymax    : {ymax}")

    algorithms = {

        "nearest": gdal.GRA_NearestNeighbour,

        "bilinear": gdal.GRA_Bilinear,

        "cubic": gdal.GRA_Cubic,

        "cubicspline": gdal.GRA_CubicSpline,

        "lanczos": gdal.GRA_Lanczos,

    }

    options = gdal.WarpOptions(

        outputBounds=(xmin, ymin, xmax, ymax),

        width=cols,

        height=rows,

        dstSRS=ref.GetProjection(),

        resampleAlg=algorithms[resampling],

    )

    if debug:
        print("\nWarping reference...")

    gdal.Warp(reference_output, ref, options=options)

    if debug:
        print("Warping target...")

    gdal.Warp(target_output, tar, options=options)

    ref = None
    tar = None

    # ---------------------------------------------------
    # Verify
    # ---------------------------------------------------

    ref = gdal.Open(reference_output)
    tar = gdal.Open(target_output)

    if debug:

        print("\nVerification")
        print("----------------------------")

        print("Reference Shape :",
              (ref.RasterYSize, ref.RasterXSize))

        print("Target Shape    :",
              (tar.RasterYSize, tar.RasterXSize))

        print()

        print("Reference GT")

        print(ref.GetGeoTransform())

        print()

        print("Target GT")

        print(tar.GetGeoTransform())

    ref = None
    tar = None

    return reference_output, target_output