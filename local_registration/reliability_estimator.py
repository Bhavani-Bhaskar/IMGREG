"""
===============================================================
Stage 5 : Local Registration
Step 20 : Reliability (Peak Sharpness)

Reference
---------
1. Scheffler et al. (2017) - AROSICS, Section 2.3.2, Eq. 6

Purpose
-------
Score how sharply the correlation surface peaks at the
matched location versus the rest of the surface:

    R = 100 - 100 * (mu_remain + 3*sigma_remain) / mu_peak

where mu_peak is the mean |value| over a 3x3 neighbourhood
centred on the peak (which sits at (0,0) for every window
that reaches this step), and mu_remain / sigma_remain are the
mean/std of |value| over every other pixel in the surface.

AROSICS default: reject windows with R < 30.

Inputs
------
1. subpixel_shifts.csv
2. final_surface/*.npy  (produced by Step 19)

Outputs
-------
reliability_results.csv

NOTE
----
This module ONLY scores reliability. MSSIM validation
starts in Step 21.
===============================================================
"""

import os
import gc
import numpy as np
import pandas as pd


class ReliabilityEstimator:
    """
    Stage 5 - Step 20

    Computes the AROSICS peak-sharpness reliability score for
    every subpixel-refined window from Step 19.
    """

    def __init__(
        self,
        subpixel_csv,
        surface_directory,
        output_directory,
        reliability_threshold=30.0
    ):

        self.subpixel_csv = subpixel_csv
        self.surface_directory = surface_directory
        self.output_directory = output_directory
        self.reliability_threshold = reliability_threshold

        self._check_inputs()

        os.makedirs(self.output_directory, exist_ok=True)

        print("\n========================================")
        print("Stage 5 - Step 20")
        print("Reliability (Peak Sharpness)")
        print("========================================")

        print("\nLoading subpixel shifts...")

        self.shifts = pd.read_csv(self.subpixel_csv)

        print("----------------------------------------")
        print(f"Windows              : {len(self.shifts)}")
        print(f"Reliability Threshold: {self.reliability_threshold}")
        print("----------------------------------------")

        self.results = []

    def _check_inputs(self):

        if not os.path.exists(self.subpixel_csv):
            raise FileNotFoundError(f"\nMissing file:\n{self.subpixel_csv}")

        if not os.path.isdir(self.surface_directory):
            raise FileNotFoundError(f"\nMissing directory:\n{self.surface_directory}")

    def total_windows(self):

        return len(self.shifts)

    def load_surface(self, window_id):

        surface_file = os.path.join(
            self.surface_directory,
            f"final_surface_{window_id:05d}.npy"
        )

        if not os.path.exists(surface_file):
            raise FileNotFoundError(f"Missing file:\n{surface_file}")

        return np.load(surface_file)

    def peak_neighbourhood_mask(self, rows, cols):
        """
        Boolean mask selecting the 3x3 neighbourhood centred
        on (0, 0), wrapped circularly since the correlation
        surface is periodic.

        Returns
        -------
        ndarray (bool), shape (rows, cols)
        """

        mask = np.zeros((rows, cols), dtype=bool)

        for dr in (-1, 0, 1):

            for dc in (-1, 0, 1):

                mask[dr % rows, dc % cols] = True

        return mask

    def compute_reliability(self, surface):
        """
        AROSICS Eq. 6.

        Parameters
        ----------
        surface : ndarray

        Returns
        -------
        float
        """

        magnitude = np.abs(surface)

        rows, cols = surface.shape

        peak_mask = self.peak_neighbourhood_mask(rows, cols)

        mean_peak = float(magnitude[peak_mask].mean())

        remainder = magnitude[~peak_mask]

        mean_remain = float(remainder.mean())

        std_remain = float(remainder.std())

        eps = 1e-12

        if mean_peak < eps:
            return 0.0

        reliability = 100.0 - 100.0 * (mean_remain + 3.0 * std_remain) / mean_peak

        return float(reliability)

    def process_window(self, row):

        window_id = int(row["window_id"])

        surface = self.load_surface(window_id)

        reliability = self.compute_reliability(surface)

        accepted = reliability >= self.reliability_threshold

        result = row.to_dict()

        result["reliability"] = reliability
        result["reliability_accepted"] = accepted

        return result

    def process_all_windows(self):

        print("\n----------------------------------------")
        print("Scoring Reliability")
        print("----------------------------------------")

        total = self.total_windows()

        self.results = []

        for index in range(total):

            row = self.shifts.iloc[index]

            try:

                result = self.process_window(row)

                self.results.append(result)

                print(
                    f"[{index + 1}/{total}] "
                    f"Window {result['window_id']} "
                    f"R={result['reliability']:.2f} "
                    f"Accepted={result['reliability_accepted']}"
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
            raise RuntimeError("No reliability results available to save.")

        output_file = os.path.join(self.output_directory, "reliability_results.csv")

        output_df = pd.DataFrame(self.results)

        output_df.to_csv(output_file, index=False)

        print("\n----------------------------------------")
        print("Reliability Results Saved")
        print("----------------------------------------")
        print(f"Output File : {output_file}")
        print(f"Total Rows  : {len(output_df)}")
        print("----------------------------------------")

        return output_file

    def print_statistics(self):

        if len(self.results) == 0:
            print("\nNo valid windows found.")
            return

        reliability = [r["reliability"] for r in self.results]

        accepted = sum(r["reliability_accepted"] for r in self.results)

        print("\n========================================")
        print("Reliability Statistics")
        print("========================================")
        print(f"Processed Windows : {len(self.results)}")
        print(f"Accepted (R>={self.reliability_threshold:.0f}) : {accepted}")
        print(f"Rejected          : {len(self.results) - accepted}")
        print(f"Mean R            : {np.mean(reliability):.2f}")
        print(f"Min R             : {np.min(reliability):.2f}")
        print(f"Max R             : {np.max(reliability):.2f}")
        print("========================================")

    def execute(self):

        self.process_all_windows()

        self.print_statistics()

        output_csv = self.save_results()

        gc.collect()

        return output_csv

    def get_output_summary(self):

        return {

            "reliability_csv": os.path.join(
                self.output_directory, "reliability_results.csv"
            )

        }

    def run(self):

        print("\n========================================")
        print("Running Reliability Scoring")
        print("========================================")

        output_csv = self.execute()

        outputs = self.get_output_summary()

        print("\n========================================")
        print("Stage 5 - Step 20 Finished")
        print("========================================")

        print("\nOutputs")
        print("----------------------------------------")
        print(f"Reliability CSV : {outputs['reliability_csv']}")
        print("----------------------------------------")

        return outputs

    def __len__(self):

        return len(self.shifts)

    def __repr__(self):

        return f"ReliabilityEstimator(windows={len(self.shifts)})"


if __name__ == "__main__":

    subpixel_csv = (
        "/home/bhaskar/Documents/ImageReg/stage5_phase_correlation/"
        "stage5_subpixel_estimation/subpixel_shifts.csv"
    )

    surface_directory = (
        "/home/bhaskar/Documents/ImageReg/stage5_phase_correlation/"
        "stage5_subpixel_estimation/final_surface"
    )

    output_directory = (
        "/home/bhaskar/Documents/ImageReg/stage5_phase_correlation/"
        "stage5_reliability"
    )

    estimator = ReliabilityEstimator(
        subpixel_csv=subpixel_csv,
        surface_directory=surface_directory,
        output_directory=output_directory,
        reliability_threshold=30.0
    )

    outputs = estimator.run()

    print("\n========================================")
    print("Execution Completed Successfully")
    print("========================================")
