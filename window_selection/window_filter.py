"""
window_filter.py

Purpose
-------
Filter candidate window pairs before Stage 5.

Filtering Criteria
------------------
1. Swath coverage (AVHRR)
2. Cloud percentage (AVHRR)
3. MODIS signal content - a window can pass both AVHRR checks
   while its MODIS reference crop is mostly no-signal
   (near-zero reflectance), which phase correlation cannot
   match against regardless of how clean the AVHRR side is.

Author : Bhaskar
"""

import numpy as np
from osgeo import gdal


# ---------------------------------------------------------
# Read Binary Mask
# ---------------------------------------------------------

def load_mask(mask_file, debug=True):
    """
    Read binary valid-data mask.

    Parameters
    ----------
    mask_file : str

    Returns
    -------
    ndarray
    """

    if debug:
        print("\n========== Reading Valid Mask ==========")

    ds = gdal.Open(mask_file)

    if ds is None:
        raise RuntimeError("Cannot open mask.")

    mask = ds.GetRasterBand(1).ReadAsArray()

    if debug:

        print("Mask Shape :", mask.shape)

        print("Valid Pixels :", np.count_nonzero(mask))

        print("Coverage (%) :",
              100 * np.count_nonzero(mask) / mask.size)

    return mask


# ---------------------------------------------------------
# Swath Coverage
# ---------------------------------------------------------

def compute_swath_coverage(mask_window):
    """
    Compute percentage of valid pixels.

    Returns
    -------
    float
    """

    valid = np.count_nonzero(mask_window)

    total = mask_window.size

    return valid / total


# ---------------------------------------------------------
# Cloud Percentage
# ---------------------------------------------------------

def compute_cloud_percentage(
        image_window,
        mask_window,
        cloud_dn_threshold):
    """
    Compute cloud percentage using only valid pixels.

    Parameters
    ----------
    image_window : ndarray

    mask_window : ndarray
        Binary valid-data mask (1=valid, 0=invalid)

    cloud_threshold : float

    Returns
    -------
    float
    """

    valid_pixels = (mask_window == 1)
    number_valid = np.count_nonzero(valid_pixels)
    if number_valid == 0:
        return 0.0
    cloud_pixels = np.count_nonzero((image_window > cloud_dn_threshold) & valid_pixels)

    return cloud_pixels / number_valid


# ---------------------------------------------------------
# MODIS Signal Content
# ---------------------------------------------------------

def compute_modis_dark_percentage(
        modis_window,
        modis_dark_threshold):
    """
    Fraction of the MODIS reference crop that has no real
    signal (reflectance below modis_dark_threshold), among
    its finite pixels.

    A window with no MODIS structure to correlate against is
    unusable regardless of AVHRR quality - this is the same
    problem cloud detection catches on the AVHRR side, but
    checked against the reference image instead of the target.

    Parameters
    ----------
    modis_window : ndarray

    modis_dark_threshold : float

    Returns
    -------
    float
        1.0 (fully invalid) if the window has no finite pixels.
    """

    finite = np.isfinite(modis_window)

    number_finite = np.count_nonzero(finite)

    if number_finite == 0:
        return 1.0

    dark_pixels = np.count_nonzero(
        (modis_window < modis_dark_threshold) & finite
    )

    return dark_pixels / number_finite


# ---------------------------------------------------------
# Main Filtering Function
# ---------------------------------------------------------

def filter_window_pairs(

        window_pairs,

        valid_mask,

        minimum_swath_coverage,

        maximum_cloud_percentage,

        cloud_dn_threshold,

        modis_dark_threshold,

        maximum_modis_dark_percentage,

        cloud_detection=True,

        debug=True):

    """
    Filter candidate window pairs.

    Returns
    -------
    accepted_pairs

    rejected_pairs

    statistics
    """

    if debug:

        print("\n========== Filtering Window Pairs ==========")

    accepted = []

    rejected = []

    reject_swath = 0

    reject_cloud = 0

    reject_modis_dark = 0

    for pair in window_pairs:

        r0 = pair["row_start"]
        r1 = pair["row_end"]

        c0 = pair["col_start"]
        c1 = pair["col_end"]

        mask_window = valid_mask[r0:r1, c0:c1]

        swath = compute_swath_coverage(
            mask_window
        )

        # ---------------------------------------------------------
        # Optional Cloud Detection
        # ---------------------------------------------------------

        if cloud_detection:

            cloud = compute_cloud_percentage(
                pair["avhrr_window"],
                mask_window,
                cloud_dn_threshold
            )

        else:

            cloud = 0.0

        modis_dark = compute_modis_dark_percentage(
            pair["modis_window"],
            modis_dark_threshold
        )

        cloud_fail = (
            cloud > maximum_cloud_percentage
        )

        swath_fail = (
            swath < minimum_swath_coverage
        )

        modis_fail = (
            modis_dark > maximum_modis_dark_percentage
        )

        pair["swath_coverage"] = swath

        pair["cloud_percentage"] = cloud

        pair["modis_dark_percentage"] = modis_dark

        pair["minimum_swath"] = minimum_swath_coverage
        pair["maximum_cloud"] = maximum_cloud_percentage
        pair["maximum_modis_dark"] = maximum_modis_dark_percentage

        if not swath_fail and not cloud_fail and not modis_fail:

            accepted.append(pair)

        else:

            rejected.append(pair)

            if swath_fail:
                reject_swath += 1

            if cloud_fail:
                reject_cloud += 1

            if modis_fail:
                reject_modis_dark += 1

    statistics = {

        "total": len(window_pairs),

        "accepted": len(accepted),

        "rejected": len(rejected),

        "reject_swath": reject_swath,

        "reject_cloud": reject_cloud,

        "reject_modis_dark": reject_modis_dark

    }

    if debug:

        print()

        print("Total Windows     :", statistics["total"])

        print("Accepted          :", statistics["accepted"])

        print("Rejected          :", statistics["rejected"])

        print()

        print("Rejected Swath      :", reject_swath)

        print("Rejected Cloud      :", reject_cloud)

        print("Rejected MODIS Dark :", reject_modis_dark)

        print("(a window can fail more than one check)")

        print()

        print("Cloud Detection :", cloud_detection)

        print("Cloud DN Threshold :", cloud_dn_threshold)

        print("Maximum Cloud Percentage :", maximum_cloud_percentage)

        print("MODIS Dark Threshold :", modis_dark_threshold)

        print("Maximum MODIS Dark Percentage :", maximum_modis_dark_percentage)

        if statistics["total"] > 0:

            rate = (

                100 *

                statistics["accepted"] /

                statistics["total"]

            )

            print(

                "Acceptance Rate  : %.2f %%"

                % rate

            )

    return accepted, rejected, statistics