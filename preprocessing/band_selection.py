"""
band_selection.py

Purpose
-------
Extract a single band from a multi-band raster.

Author : Bhaskar
"""

import os
from osgeo import gdal

gdal.UseExceptions()
    
def select_band(
    input_file,
    output_file,
    band_number=1,
    debug=True,
):
    """
    Extract a single band from a multi-band raster.

    Parameters
    ----------
    input_file : str
        Input raster.

    output_file : str
        Output single-band raster.

    band_number : int
        Band index (1-based).

    debug : bool
        Print processing information.

    Returns
    -------
    str
        Output raster path.
    """

    if debug:
        print("\n========== Band Selection ==========")

    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    dataset = gdal.Open(input_file)

    if dataset is None:
        raise RuntimeError("Unable to open raster.")

    total_bands = dataset.RasterCount

    if debug:
        print(f"Input File   : {input_file}")
        print(f"Total Bands  : {total_bands}")
        print(f"Selected Band: {band_number}")

    if band_number < 1 or band_number > total_bands:
        raise ValueError(
            f"Band {band_number} does not exist."
        )

    band = dataset.GetRasterBand(band_number)

    image = band.ReadAsArray()

    driver = gdal.GetDriverByName("GTiff")

    output = driver.Create(

        output_file,

        dataset.RasterXSize,

        dataset.RasterYSize,

        1,

        band.DataType

    )

    output.SetGeoTransform(
        dataset.GetGeoTransform()
    )

    output.SetProjection(
        dataset.GetProjection()
    )

    out_band = output.GetRasterBand(1)

    out_band.WriteArray(image)

    nodata = band.GetNoDataValue()

    if nodata is not None:
        out_band.SetNoDataValue(nodata)

    out_band.FlushCache()

    output.FlushCache()

    output = None
    dataset = None

    if debug:

        verify = gdal.Open(output_file)

        print("\nVerification")
        print("-------------------------")
        print(
            "Output Bands :",
            verify.RasterCount
        )

        print(
            "Output Shape :",
            (
                verify.RasterYSize,
                verify.RasterXSize
            )
        )

        print("Band extraction completed.")

        verify = None

        return output_file