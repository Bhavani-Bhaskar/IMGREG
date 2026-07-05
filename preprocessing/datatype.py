"""
datatype.py

Purpose
-------
Convert raster to Float32 datatype.

Author : Bhaskar
"""

from osgeo import gdal
import numpy as np
import os

gdal.UseExceptions()


def convert_to_float32(
    input_file,
    output_file,
    debug=True,
):
    """
    Convert raster to Float32.

    Parameters
    ----------
    input_file : str
        Input raster.

    output_file : str
        Output raster.

    debug : bool
        Print debugging information.

    Returns
    -------
    str
        Output filename.
    """

    if debug:
        print("\n========== Data Type Conversion ==========")

    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    dataset = gdal.Open(input_file)

    if dataset is None:
        raise RuntimeError("Unable to open raster.")

    band = dataset.GetRasterBand(1)

    current_dtype = band.DataType

    dtype_name = gdal.GetDataTypeName(current_dtype)

    if debug:
        print(f"Input File      : {input_file}")
        print(f"Current Type    : {dtype_name}")

    # --------------------------------------------------
    # Already Float32?
    # --------------------------------------------------

    if current_dtype == gdal.GDT_Float32:

        if debug:
            print("Raster already Float32.")
            print("Skipping conversion.")

        dataset = None

        return input_file

    # --------------------------------------------------
    # Read Image
    # --------------------------------------------------

    image = band.ReadAsArray().astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")

    output = driver.Create(

        output_file,

        dataset.RasterXSize,

        dataset.RasterYSize,

        1,

        gdal.GDT_Float32

    )

    # Preserve metadata

    output.SetGeoTransform(
        dataset.GetGeoTransform()
    )

    output.SetProjection(
        dataset.GetProjection()
    )

    output.SetMetadata(
        dataset.GetMetadata()
    )

    out_band = output.GetRasterBand(1)

    out_band.WriteArray(image)

    nodata = band.GetNoDataValue()

    if nodata is not None:
        out_band.SetNoDataValue(float(nodata))

    out_band.FlushCache()

    output.FlushCache()

    output = None
    dataset = None

    # --------------------------------------------------
    # Verification
    # --------------------------------------------------

    verify = gdal.Open(output_file)

    verify_band = verify.GetRasterBand(1)

    if debug:

        print("\nVerification")
        print("----------------------------")

        print(
            "Output Type :",
            gdal.GetDataTypeName(
                verify_band.DataType
            )
        )

        print("Conversion completed.")

    verify = None

    return output_file