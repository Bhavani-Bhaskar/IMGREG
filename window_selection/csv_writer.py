"""
csv_writer.py

Purpose
-------
Save accepted AVHRR-MODIS window pairs to CSV.

Only metadata is saved.
Image arrays are NOT stored.

Author : Bhaskar
"""

import os
import pandas as pd


def save_accepted_pairs_csv(
        accepted_pairs,
        output_csv,
        debug=True):
    """
    Save accepted window information.

    Parameters
    ----------
    accepted_pairs : list

    output_csv : str

    Returns
    -------
    str
    """

    if debug:
        print("\n========== Saving Accepted Window Pairs ==========")

    directory = os.path.dirname(output_csv)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    rows = []

    for pair in accepted_pairs:

        rows.append({

            "window_id": pair["window_id"],

            "row_start": pair["row_start"],
            "row_end": pair["row_end"],

            "col_start": pair["col_start"],
            "col_end": pair["col_end"],

            "center_row": pair["center_row"],
            "center_col": pair["center_col"],

            "swath_coverage": pair["swath_coverage"],

            "cloud_percentage": pair["cloud_percentage"]

        })

    df = pd.DataFrame(rows)

    df.to_csv(
        output_csv,
        index=False
    )

    if debug:

        print("CSV File :", output_csv)

        print("Accepted Windows :", len(df))

        print("\nColumns")

        for column in df.columns:
            print(" -", column)

        print("\nCSV saved successfully.")

    return output_csv



def save_rejected_pairs_csv(
        rejected_pairs,
        output_csv,
        debug=True):
    """
    Save rejected window information.

    Parameters
    ----------
    rejected_pairs : list

    output_csv : str

    Returns
    -------
    str
    """

    import os
    import pandas as pd

    if debug:
        print("\n========== Saving Rejected Window Pairs ==========")

    directory = os.path.dirname(output_csv)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    rows = []

    for pair in rejected_pairs:

        reason = []

        if pair["swath_coverage"] < pair["minimum_swath"]:
            reason.append("LOW_SWATH")

        if pair["cloud_percentage"] > pair["maximum_cloud"]:
            reason.append("HIGH_CLOUD")

        rows.append({

            "window_id": pair["window_id"],

            "row_start": pair["row_start"],
            "row_end": pair["row_end"],

            "col_start": pair["col_start"],
            "col_end": pair["col_end"],

            "center_row": pair["center_row"],
            "center_col": pair["center_col"],

            "swath_coverage": pair["swath_coverage"],

            "cloud_percentage": pair["cloud_percentage"],

            "rejection_reason": "+".join(reason)

        })

    df = pd.DataFrame(rows)

    df.to_csv(
        output_csv,
        index=False
    )

    if debug:

        print("CSV File :", output_csv)

        print("Rejected Windows :", len(df))

        print("CSV saved successfully.")

    return output_csv