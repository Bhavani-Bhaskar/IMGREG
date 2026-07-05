from osgeo import gdal
import os

gdal.UseExceptions()


def match_spatial_resolution(
    input_file,
    output_file,
    x_resolution,
    y_resolution,
    target_srs="EPSG:4326",
    resampling="bilinear",
    debug=True,
):
    """
    Match the spatial resolution of a raster using GDAL Warp.

    Parameters
    ----------
    input_file : str
        Path to input raster.

    output_file : str
        Path to output raster.

    x_resolution : float
        Target pixel width.

    y_resolution : float
        Target pixel height.

    target_srs : str
        Output CRS (default: EPSG:4326).

    resampling : str
        nearest, bilinear, cubic, cubicspline, lanczos

    debug : bool
        Print processing information.

    Returns
    -------
    str
        Output raster path.
    """

    if debug:
        print("\n========== Matching Spatial Resolution ==========")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found:\n{input_file}")

    dataset = gdal.Open(input_file)

    if dataset is None:
        raise RuntimeError("Unable to open input raster.")

    geo_transform = dataset.GetGeoTransform()

    current_x = abs(geo_transform[1])
    current_y = abs(geo_transform[5])

    if debug:
        print(f"Current Resolution : ({current_x}, {current_y})")
        print(f"Target Resolution  : ({x_resolution}, {y_resolution})")
        print(f"Target CRS         : {target_srs}")
        print(f"Resampling         : {resampling}")

    resample_algorithms = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
    }

    if resampling not in resample_algorithms:
        raise ValueError(f"Unsupported resampling method: {resampling}")

    options = gdal.WarpOptions(
        xRes=x_resolution,
        yRes=y_resolution,
        dstSRS=target_srs,
        resampleAlg=resample_algorithms[resampling],
    )

    result = gdal.Warp(
        destNameOrDestDS=output_file,
        srcDSOrSrcDSTab=dataset,
        options=options,
    )

    if result is None:
        raise RuntimeError("GDAL Warp failed.")

    result.FlushCache()

    result = None
    dataset = None

    if debug:
        verify = gdal.Open(output_file)
        gt = verify.GetGeoTransform()

        print("Output Resolution  :", (abs(gt[1]), abs(gt[5])))
        print("Spatial resolution matching completed.\n")

        verify = None

    return output_file
