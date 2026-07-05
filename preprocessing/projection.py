"""
projection.py

Purpose
-------
Match the projection (CRS) of the target raster to
the reference raster.

Author : Bhaskar
"""

from osgeo import gdal, osr
import os

gdal.UseExceptions()


def match_projection(
    reference_file,
    target_file,
    output_file,
    resampling="bilinear",
    debug=True,
):
    """
    Match the projection (CRS) of the target raster to
    the reference raster.

    Parameters
    ----------
    reference_file : str
        Reference raster (MODIS)

    target_file : str
        Target raster (AVHRR)

    output_file : str
        Reprojected target raster

    resampling : str
        nearest, bilinear, cubic,
        cubicspline, lanczos

    debug : bool

    Returns
    -------
    str
        Output filename
    """

    if debug:
        print("\n========== Projection Matching ==========")

    if not os.path.exists(reference_file):
        raise FileNotFoundError(reference_file)

    if not os.path.exists(target_file):
        raise FileNotFoundError(target_file)

    ref_ds = gdal.Open(reference_file)
    tar_ds = gdal.Open(target_file)

    if ref_ds is None or tar_ds is None:
        raise RuntimeError("Unable to open raster.")

    ref_wkt = ref_ds.GetProjection()
    tar_wkt = tar_ds.GetProjection()

    ref_srs = osr.SpatialReference()
    tar_srs = osr.SpatialReference()

    ref_srs.ImportFromWkt(ref_wkt)
    tar_srs.ImportFromWkt(tar_wkt)

    if debug:

        print("Reference CRS")
        print("-------------------------")
        print(ref_srs.GetAttrValue("AUTHORITY", 1))

        print("\nTarget CRS")
        print("-------------------------")
        print(tar_srs.GetAttrValue("AUTHORITY", 1))

    # ---------------------------------------
    # Already same projection?
    # ---------------------------------------

    if ref_srs.IsSame(tar_srs):

        if debug:
            print("\nProjection already identical.")
            print("Skipping reprojection.")

        ref_ds = None
        tar_ds = None

        return target_file

    # ---------------------------------------
    # Resampling methods
    # ---------------------------------------

    algorithms = {

        "nearest": gdal.GRA_NearestNeighbour,

        "bilinear": gdal.GRA_Bilinear,

        "cubic": gdal.GRA_Cubic,

        "cubicspline": gdal.GRA_CubicSpline,

        "lanczos": gdal.GRA_Lanczos

    }

    if resampling not in algorithms:
        raise ValueError("Unsupported resampling method.")

    if debug:
        print("\nReprojecting target raster...")

    options = gdal.WarpOptions(

        dstSRS=ref_wkt,

        resampleAlg=algorithms[resampling]

    )

    result = gdal.Warp(

        output_file,

        tar_ds,

        options=options

    )

    if result is None:
        raise RuntimeError("Projection matching failed.")

    result.FlushCache()

    result = None
    ref_ds = None
    tar_ds = None

    verify = gdal.Open(output_file)

    verify_srs = osr.SpatialReference()
    verify_srs.ImportFromWkt(verify.GetProjection())

    if debug:

        print("\nVerification")
        print("-------------------------")
        print(
            "Output CRS :",
            verify_srs.GetAttrValue("AUTHORITY", 1)
        )

        print("Projection matching completed.")

    verify = None

    return output_file
