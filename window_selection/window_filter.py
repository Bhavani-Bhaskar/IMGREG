"""
window_filter.py

Purpose
-------
Filter candidate window pairs before Stage 5.

Filtering Criteria
------------------
1. Swath coverage
2. Cloud percentage

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
# Main Filtering Function
# ---------------------------------------------------------

def filter_window_pairs(

        window_pairs,

        valid_mask,

        minimum_swath_coverage,

        maximum_cloud_percentage,

        cloud_dn_threshold,

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

    reject_both = 0

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

        cloud_fail = (
            cloud > maximum_cloud_percentage
        )

        swath_fail = (
            swath < minimum_swath_coverage
        )


        if not swath_fail and not cloud_fail:

            pair["swath_coverage"] = swath

            pair["cloud_percentage"] = cloud
            
            pair["minimum_swath"] = minimum_swath_coverage
            pair["maximum_cloud"] = maximum_cloud_percentage

            accepted.append(pair)

        else:

            pair["swath_coverage"] = swath

            pair["cloud_percentage"] = cloud
            
            pair["minimum_swath"] = minimum_swath_coverage
            pair["maximum_cloud"] = maximum_cloud_percentage

            rejected.append(pair)

            if swath_fail and cloud_fail:

                reject_both += 1

            elif swath_fail:

                reject_swath += 1

            elif cloud_fail:

                reject_cloud += 1

    statistics = {

        "total": len(window_pairs),

        "accepted": len(accepted),

        "rejected": len(rejected),

        "reject_swath": reject_swath,

        "reject_cloud": reject_cloud,

        "reject_both": reject_both

    }

    if debug:

        print()

        print("Total Windows     :", statistics["total"])

        print("Accepted          :", statistics["accepted"])

        print("Rejected          :", statistics["rejected"])

        print()

        print("Rejected Swath    :", reject_swath)

        print("Rejected Cloud    :", reject_cloud)

        print("Rejected Both     :", reject_both)

        print()

        print("Cloud Detection :", cloud_detection)

        print("Cloud DN Threshold :", cloud_dn_threshold)

        print("Maximum Cloud Percentage :", maximum_cloud_percentage)

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