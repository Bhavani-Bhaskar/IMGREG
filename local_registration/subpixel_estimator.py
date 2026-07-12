"""
===============================================================
Stage 5 : Local Registration
Step 19 : Subpixel Bootstrap

Reference
---------
1. Foroosh et al. (2002)
2. Scheffler et al. (2017) - AROSICS

Purpose
-------
For every window that passed Step 18 (validated=True, integer
peak exactly at (0,0)), extract the fractional part of the
shift from the immediate neighbours of that peak, using
Foroosh's closed-form two-sample estimator:

    delta = c_plus  / (c_plus  + c0)   if c_plus  >= c_minus
    delta = -c_minus / (c_minus + c0)  otherwise

applied independently along rows (dy) and columns (dx).

Inputs
------
1. avhrr_data.tif
2. modis_data.tif
3. validated_integer_shifts.csv

Outputs
-------
subpixel_shifts.csv
final_surface/*.npy   (final correlation surface at the
                        validated integer shift, reused by
                        Step 20 - reliability)

NOTE
----
This module ONLY estimates the subpixel correction.
Reliability scoring starts in Step 20.
===============================================================
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import shift as nd_shift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_registration.core.phase_engine import PhaseCorrelationEngine


class SubpixelEstimator:
    """
    Stage 5 - Step 19

    Estimates the subpixel (fractional) shift correction for
    every window validated in Step 18.
    """

    def __init__(
        self,
        avhrr_path,
        modis_path,
        validated_csv,
        output_directory
    ):

        self.avhrr_path = avhrr_path
        self.modis_path = modis_path
        self.validated_csv = validated_csv
        self.output_directory = output_directory
        self.engine = PhaseCorrelationEngine()

        self._check_inputs()

        os.makedirs(self.output_directory, exist_ok=True)

        self.surface_dir = os.path.join(
            self.output_directory,
            "final_surface"
        )

        os.makedirs(self.surface_dir, exist_ok=True)

        print("\n========================================")
        print("Stage 5 - Step 19")
        print("Subpixel Bootstrap")
        print("========================================")

        print("\nLoading AVHRR...")

        avhrr_ds = gdal.Open(self.avhrr_path, gdal.GA_ReadOnly)

        if avhrr_ds is None:
            raise RuntimeError(f"Unable to open {self.avhrr_path}")

        self.avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

        print("Loading MODIS...")

        modis_ds = gdal.Open(self.modis_path, gdal.GA_ReadOnly)

        if modis_ds is None:
            raise RuntimeError(f"Unable to open {self.modis_path}")

        self.modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

        print("Reading validated shifts...")

        shifts = pd.read_csv(self.validated_csv)

        self.shifts = shifts[shifts["validated"] == True].reset_index(drop=True)

        print("----------------------------------------")
        print(f"AVHRR Shape        : {self.avhrr.shape}")
        print(f"MODIS Shape        : {self.modis.shape}")
        print(f"Validated Windows  : {len(self.shifts)} / {len(shifts)}")
        print("----------------------------------------")

        self.results = []

    def _check_inputs(self):

        required_files = [
            self.avhrr_path,
            self.modis_path,
            self.validated_csv
        ]

        for file in required_files:

            if not os.path.exists(file):
                raise FileNotFoundError(f"\nMissing input:\n{file}")

    def total_windows(self):

        return len(self.shifts)

    def get_window(self, index):
        """
        Load one validated window and re-apply its validated
        integer shift.

        Returns
        -------
        dict
        """

        row = self.shifts.iloc[index]

        window_id = int(row["window_id"])

        row_start = int(row["row_start"])
        row_end = int(row["row_end"])

        col_start = int(row["col_start"])
        col_end = int(row["col_end"])

        validated_dx = int(row["validated_dx"])
        validated_dy = int(row["validated_dy"])

        avhrr_window = self.avhrr[row_start:row_end, col_start:col_end]
        modis_window = self.modis[row_start:row_end, col_start:col_end]

        shifted_avhrr = nd_shift(
            avhrr_window,
            shift=(validated_dy, validated_dx),
            order=0,
            mode="constant",
            cval=np.nan,
            prefilter=False
        )

        return {

            "window_id": window_id,

            "row": row,

            "validated_dx": validated_dx,

            "validated_dy": validated_dy,

            "modis_window": modis_window,

            "shifted_avhrr": shifted_avhrr

        }

    def foroosh_delta(self, c0, c_plus, c_minus):
        """
        Foroosh's closed-form two-sample subpixel estimator,
        applied along a single axis.

        Parameters
        ----------
        c0 : float
            Correlation value at the integer peak.

        c_plus : float
            Correlation value at peak + 1.

        c_minus : float
            Correlation value at peak - 1.

        Returns
        -------
        float
            Subpixel correction in [-1, 1].
        """

        eps = 1e-12

        if c_plus >= c_minus:

            denom = c_plus + c0

            delta = c_plus / denom if abs(denom) > eps else 0.0

        else:

            denom = c_minus + c0

            delta = -c_minus / denom if abs(denom) > eps else 0.0

        return float(np.clip(delta, -1.0, 1.0))

    def estimate_subpixel(self, window):
        """
        Recompute the final correlation surface at the
        validated integer shift and extract the subpixel
        correction from the peak's immediate neighbours.

        Returns
        -------
        dict
        """

        surface = self.engine.compute(
            window["modis_window"],
            window["shifted_avhrr"]
        )

        rows, cols = surface.shape

        peak_index = np.argmax(surface)
        peak_row, peak_col = np.unravel_index(peak_index, surface.shape)

        if (peak_row, peak_col) != (0, 0):

            raise ValueError(
                f"Window {window['window_id']}: expected peak at (0,0), "
                f"found ({peak_row},{peak_col}). Not a validated window."
            )

        c0 = float(surface[0, 0])

        c_x_plus = float(surface[0, 1])
        c_x_minus = float(surface[0, cols - 1])

        c_y_plus = float(surface[1, 0])
        c_y_minus = float(surface[rows - 1, 0])

        delta_x = self.foroosh_delta(c0, c_x_plus, c_x_minus)
        delta_y = self.foroosh_delta(c0, c_y_plus, c_y_minus)

        final_dx = window["validated_dx"] + delta_x
        final_dy = window["validated_dy"] + delta_y

        return {

            "window_id": window["window_id"],

            "delta_x": delta_x,

            "delta_y": delta_y,

            "final_dx": final_dx,

            "final_dy": final_dy,

            "peak_value": c0,

            "surface": surface

        }

    def process_all_windows(self):

        print("\n----------------------------------------")
        print("Estimating Subpixel Shifts")
        print("----------------------------------------")

        total = self.total_windows()

        self.results = []

        for index in range(total):

            window = self.get_window(index)

            try:

                result = self.estimate_subpixel(window)

                self.results.append({**window["row"].to_dict(), **{

                    "delta_x": result["delta_x"],
                    "delta_y": result["delta_y"],
                    "final_dx": result["final_dx"],
                    "final_dy": result["final_dy"],
                    "subpixel_peak_value": result["peak_value"]

                }})

                surface_file = os.path.join(
                    self.surface_dir,
                    f"final_surface_{result['window_id']:05d}.npy"
                )

                np.save(surface_file, result["surface"])

                print(
                    f"[{index + 1}/{total}] "
                    f"Window {result['window_id']} "
                    f"dx={result['final_dx']:.3f} "
                    f"dy={result['final_dy']:.3f}"
                )

            except Exception as error:

                print(f"[{index + 1}/{total}] Skipped : {error}")

                continue

        print("----------------------------------------")
        print(f"Processed : {len(self.results)}")
        print(f"Skipped   : {total - len(self.results)}")
        print("----------------------------------------")

        return self.results

    def save_results(self):

        if len(self.results) == 0:
            raise RuntimeError("No subpixel results available to save.")

        output_file = os.path.join(self.output_directory, "subpixel_shifts.csv")

        output_df = pd.DataFrame(self.results)

        output_df.to_csv(output_file, index=False)

        print("\n----------------------------------------")
        print("Subpixel Results Saved")
        print("----------------------------------------")
        print(f"Output File : {output_file}")
        print(f"Total Rows  : {len(output_df)}")
        print("----------------------------------------")

        return output_file

    def print_statistics(self):

        if len(self.results) == 0:
            print("\nNo valid windows found.")
            return

        delta_x = [r["delta_x"] for r in self.results]
        delta_y = [r["delta_y"] for r in self.results]

        print("\n========================================")
        print("Subpixel Statistics")
        print("========================================")
        print(f"Processed Windows : {len(self.results)}")
        print(f"Mean |delta_x|    : {np.mean(np.abs(delta_x)):.4f}")
        print(f"Mean |delta_y|    : {np.mean(np.abs(delta_y)):.4f}")
        print(f"Max  |delta_x|    : {np.max(np.abs(delta_x)):.4f}")
        print(f"Max  |delta_y|    : {np.max(np.abs(delta_y)):.4f}")
        print("========================================")

    def execute(self):

        self.process_all_windows()

        self.print_statistics()

        output_csv = self.save_results()

        gc.collect()

        return output_csv

    def close(self):

        if hasattr(self, "avhrr"):
            del self.avhrr

        if hasattr(self, "modis"):
            del self.modis

        if hasattr(self, "shifts"):
            del self.shifts

        gc.collect()

        print("\nResources released successfully.")

    def get_output_summary(self):

        return {

            "subpixel_csv": os.path.join(self.output_directory, "subpixel_shifts.csv"),

            "final_surface_directory": self.surface_dir

        }

    def run(self):

        print("\n========================================")
        print("Running Subpixel Estimation")
        print("========================================")

        output_csv = self.execute()

        outputs = self.get_output_summary()

        self.close()

        print("\n========================================")
        print("Stage 5 - Step 19 Finished")
        print("========================================")

        print("\nOutputs")
        print("----------------------------------------")
        print(f"Subpixel CSV           : {outputs['subpixel_csv']}")
        print(f"Final Surface Directory: {outputs['final_surface_directory']}")
        print("----------------------------------------")

        return outputs

    def __len__(self):

        return len(self.shifts)

    def __repr__(self):

        return f"SubpixelEstimator(windows={len(self.shifts)})"


if __name__ == "__main__":

    avhrr_path = "/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_float32.tif"
    modis_path = "/home/bhaskar/Documents/ImageReg/2_outputs/05_modis_float32.tif"

    validated_csv = (
        "/home/bhaskar/Documents/ImageReg/stage5_phase_correlation/"
        "stage5_shift_validation/validated_integer_shifts.csv"
    )

    output_directory = (
        "/home/bhaskar/Documents/ImageReg/stage5_phase_correlation/"
        "stage5_subpixel_estimation"
    )

    estimator = SubpixelEstimator(
        avhrr_path=avhrr_path,
        modis_path=modis_path,
        validated_csv=validated_csv,
        output_directory=output_directory
    )

    outputs = estimator.run()

    print("\n========================================")
    print("Execution Completed Successfully")
    print("========================================")
