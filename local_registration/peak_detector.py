"""
===============================================================
Stage 5 : Local Registration
Step 17 : Integer Peak Detection

Reference
---------
1. Foroosh et al. (2002)
2. Scheffler et al. (2017) - AROSICS

Input
-----
1. phase_correlation_summary.csv
2. correlation_surface/*.npy

Output
------
integer_peak_results.csv

NOTE
----
This module ONLY finds the integer peak location.

It DOES NOT:
    - Validate the shift
    - Estimate subpixel shifts
    - Compute reliability
===============================================================
"""

import os
import gc
import numpy as np
import pandas as pd


class PeakDetector:
    """
    Stage 5 - Step 17

    Detects the integer correlation peak for every
    accepted local window.
    """

    def __init__(
        self,
        summary_csv,
        correlation_directory,
        output_directory
    ):

        self.summary_csv = summary_csv

        self.correlation_directory = correlation_directory

        self.output_directory = output_directory

        self._check_inputs()

        os.makedirs(
            self.output_directory,
            exist_ok=True
        )

        print("\n========================================")
        print("Stage 5 - Step 17")
        print("Integer Peak Detection")
        print("========================================")

        print("\nLoading summary CSV...")

        self.summary = pd.read_csv(
            self.summary_csv
        )

        print("----------------------------------------")
        print(f"Total Windows : {len(self.summary)}")
        print("----------------------------------------")

        self.results = []


    def _check_inputs(self):
        """
        Verify required files/directories exist.
        """

        if not os.path.exists(
            self.summary_csv
        ):
            raise FileNotFoundError(

                f"\nMissing file:\n{self.summary_csv}"

            )

        if not os.path.isdir(
            self.correlation_directory
        ):
            raise FileNotFoundError(

                f"\nMissing directory:\n"
                f"{self.correlation_directory}"

            )

    def load_window(self, index):
        """
        Load one correlation surface and its metadata.

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """

        row = self.summary.iloc[index]

        window_id = int(row["window_id"])

        row_start = int(row["row_start"])
        row_end = int(row["row_end"])

        col_start = int(row["col_start"])
        col_end = int(row["col_end"])

        center_row = int(row["center_row"])
        center_col = int(row["center_col"])

        swath_coverage = float(row["swath_coverage"])
        cloud_percentage = float(row["cloud_percentage"])

        # ----------------------------------------
        # Build correlation surface filename
        # ----------------------------------------

        correlation_file = os.path.join(

            self.correlation_directory,

            f"correlation_surface_{window_id:05d}.npy"

        )

        if not os.path.exists(correlation_file):

            raise FileNotFoundError(

                f"Missing file:\n{correlation_file}"

            )

        # ----------------------------------------
        # Load correlation surface
        # ----------------------------------------

        correlation_surface = np.load(
            correlation_file
        )

        # ----------------------------------------
        # Validation
        # ----------------------------------------

        if correlation_surface.ndim != 2:

            raise ValueError(

                f"Window {window_id}: "
                "Correlation surface must be 2-D."

            )

        if not np.isfinite(correlation_surface).all():

            raise ValueError(

                f"Window {window_id}: "
                "Correlation surface contains NaN or Inf."

            )

        rows, cols = correlation_surface.shape

        return {

            "window_id": window_id,

            "row_start": row_start,
            "row_end": row_end,

            "col_start": col_start,
            "col_end": col_end,

            "center_row": center_row,
            "center_col": center_col,

            "swath_coverage": swath_coverage,
            "cloud_percentage": cloud_percentage,

            "rows": rows,
            "cols": cols,

            "correlation_surface": correlation_surface

        }


    def total_windows(self):
        """
        Total number of windows.
        """

        return len(self.summary)

    def find_integer_peak(self, correlation_surface):
        """
        Find the integer peak in the correlation surface.

        Parameters
        ----------
        correlation_surface : ndarray

        Returns
        -------
        peak_row : int

        peak_col : int

        peak_value : float
        """

        # ----------------------------------------
        # Find global maximum
        # ----------------------------------------

        peak_index = np.argmax(
            correlation_surface
        )

        peak_row, peak_col = np.unravel_index(

            peak_index,

            correlation_surface.shape

        )

        peak_value = float(

            correlation_surface[
                peak_row,
                peak_col
            ]

        )

        return (

            peak_row,

            peak_col,

            peak_value

        )


    def detect_peak(self, window):
        """
        Detect integer peak for one window.

        Parameters
        ----------
        window : dict

        Returns
        -------
        dict
        """

        peak_row, peak_col, peak_value = self.find_integer_peak(

            window["correlation_surface"]

        )

        return {

            "window_id": window["window_id"],

            "row_start": window["row_start"],
            "row_end": window["row_end"],

            "col_start": window["col_start"],
            "col_end": window["col_end"],

            "center_row": window["center_row"],
            "center_col": window["center_col"],

            "swath_coverage": window["swath_coverage"],
            "cloud_percentage": window["cloud_percentage"],

            "rows": window["rows"],
            "cols": window["cols"],

            "peak_row": peak_row,

            "peak_col": peak_col,

            "peak_value": peak_value

        }

    def compute_integer_shift(
        self,
        peak_row,
        peak_col,
        rows,
        cols
    ):
        """
        Convert FFT peak coordinates into
        integer translation.

        Parameters
        ----------
        peak_row : int

        peak_col : int

        rows : int

        cols : int

        Returns
        -------
        dy_integer : int

        dx_integer : int
        """

        # -------------------------------
        # Convert circular row shift
        # -------------------------------

        if peak_row > rows // 2:

            dy_integer = peak_row - rows

        else:

            dy_integer = peak_row

        # -------------------------------
        # Convert circular column shift
        # -------------------------------

        if peak_col > cols // 2:

            dx_integer = peak_col - cols

        else:

            dx_integer = peak_col

        return (

            dy_integer,

            dx_integer

        )


    def process_peak(self, window):
        """
        Detect peak and compute
        integer translation.

        Parameters
        ----------
        window : dict

        Returns
        -------
        dict
        """

        peak = self.detect_peak(window)

        dy_integer, dx_integer = self.compute_integer_shift(

            peak_row=peak["peak_row"],

            peak_col=peak["peak_col"],

            rows=peak["rows"],

            cols=peak["cols"]

        )

        peak["dy_integer"] = dy_integer

        peak["dx_integer"] = dx_integer

        return peak

    def process_all_windows(self):
        """
        Process all correlation surfaces.

        Workflow
        --------
        Load Correlation Surface
                ↓
        Find Integer Peak
                ↓
        Convert Circular Shift
                ↓
        Store Result

        Returns
        -------
        list
        """

        total = self.total_windows()

        print("\n----------------------------------------")
        print("Processing Correlation Surfaces")
        print("----------------------------------------")

        self.results = []

        for index in range(total):

            try:

                # ----------------------------------
                # Load one window
                # ----------------------------------

                window = self.load_window(index)

                # ----------------------------------
                # Detect peak
                # ----------------------------------

                result = self.process_peak(window)

                self.results.append(result)

                print(
                    f"[{index+1}/{total}] "
                    f"Window {result['window_id']} completed."
                )

            except Exception as error:

                print(
                    f"[{index+1}/{total}] "
                    f"Window skipped : {error}"
                )

                continue

        print("----------------------------------------")
        print(f"Processed : {len(self.results)}")
        print(f"Skipped   : {total-len(self.results)}")
        print("----------------------------------------")

        return self.results


    def print_statistics(self):
        """
        Print processing statistics.
        """

        if len(self.results) == 0:

            print("\nNo valid windows found.")

            return

        peak_values = [

            result["peak_value"]

            for result in self.results

        ]

        print("\n========================================")
        print("Peak Detection Statistics")
        print("========================================")

        print(f"Processed Windows : {len(self.results)}")

        print(
            f"Maximum Peak : {np.max(peak_values):.6f}"
        )

        print(
            f"Minimum Peak : {np.min(peak_values):.6f}"
        )

        print(
            f"Mean Peak    : {np.mean(peak_values):.6f}"
        )

        print("========================================")

    def save_results(self):
        """
        Save integer peak detection results.

        Output
        ------
        integer_peak_results.csv
        """

        if len(self.results) == 0:

            raise RuntimeError(
                "No results available to save."
            )

        output_file = os.path.join(

            self.output_directory,

            "integer_peak_results.csv"

        )

        output_rows = []

        for result in self.results:

            output_rows.append({

                "window_id":
                    result["window_id"],

                "row_start":
                    result["row_start"],

                "row_end":
                    result["row_end"],

                "col_start":
                    result["col_start"],

                "col_end":
                    result["col_end"],

                "center_row":
                    result["center_row"],

                "center_col":
                    result["center_col"],

                "swath_coverage":
                    result["swath_coverage"],

                "cloud_percentage":
                    result["cloud_percentage"],

                "peak_row":
                    result["peak_row"],

                "peak_col":
                    result["peak_col"],

                "peak_value":
                    result["peak_value"],

                "dy_integer":
                    result["dy_integer"],

                "dx_integer":
                    result["dx_integer"]

            })

        output_df = pd.DataFrame(output_rows)

        output_df.to_csv(

            output_file,

            index=False

        )

        print("\n----------------------------------------")
        print("Integer Peak Results Saved")
        print("----------------------------------------")
        print(f"Output File : {output_file}")
        print(f"Total Rows  : {len(output_df)}")
        print("----------------------------------------")

        return output_file


    def execute(self):
        """
        Complete Step 17.

        Workflow
        --------
        Load Correlation Surface
                ↓
        Detect Integer Peak
                ↓
        Convert Circular Shift
                ↓
        Save CSV

        Returns
        -------
        str
            Path to integer_peak_results.csv
        """

        self.process_all_windows()

        self.print_statistics()

        output_csv = self.save_results()

        gc.collect()

        return output_csv

if __name__ == "__main__":

    # =====================================================
    # Input Files
    # =====================================================

    summary_csv = (
        "stage5_phase_correlation/"
        "phase_correlation_summary.csv"
    )

    correlation_directory = (
        "stage5_phase_correlation/"
        "correlation_surface"
    )

    output_directory = (
        "stage5_peak_detection"
    )

    # =====================================================
    # Create PeakDetector Object
    # =====================================================

    detector = PeakDetector(

        summary_csv=summary_csv,

        correlation_directory=correlation_directory,

        output_directory=output_directory

    )

    # =====================================================
    # Run Step 17
    # =====================================================

    output_csv = detector.execute()

    # =====================================================
    # Display Results
    # =====================================================

    print("\n========================================")
    print("Stage 5 - Step 17 Completed")
    print("========================================")

    print("\nOutput")
    print("----------------------------------------")

    print(
        f"Integer Peak Results\n"
        f"{output_csv}"
    )

    print("----------------------------------------")