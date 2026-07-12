"""
===============================================================
Stage 5 : Local Registration
Step 16 : Cross-Power Spectrum Computation

Reference
---------
1. Foroosh et al. (2002)
2. Scheffler et al. (2017) - AROSICS

Inputs
------
1. avhrr_data.tif
2. modis_data.tif
3. accepted_window_pairs.csv

Outputs
-------
Correlation surface (.npy)
Cross-power spectrum (.npy)

NOTE
----
This module DOES NOT estimate shifts.
Peak detection starts in Stage 5 Step 17.
===============================================================
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_registration.core.phase_engine import PhaseCorrelationEngine

class PhaseCorrelation:
    """
    Stage 5 - Step 16

    Computes the normalized cross-power spectrum
    and correlation surface for each accepted window pair.
    """

    def __init__(
        self,
        avhrr_path,
        modis_path,
        csv_path,
        output_dir
    ):
        self.engine = PhaseCorrelationEngine()
        self.avhrr_path = avhrr_path
        self.modis_path = modis_path
        self.csv_path = csv_path
        self.output_dir = output_dir

        self.eps = 1e-12

        self._check_inputs()

        os.makedirs(self.output_dir, exist_ok=True)

        print("\n========================================")
        print("Stage 5 - Step 16")
        print("Cross-Power Spectrum")
        print("========================================")

        # ---------------------------------------
        # Load AVHRR
        # ---------------------------------------

        print("\nLoading AVHRR...")

        avhrr_ds = gdal.Open(
            self.avhrr_path,
            gdal.GA_ReadOnly
        )

        if avhrr_ds is None:
            raise RuntimeError(
                f"Unable to open {self.avhrr_path}"
            )

        self.avhrr = (
            avhrr_ds
            .GetRasterBand(1)
            .ReadAsArray()
            .astype(np.float32)
        )

        # ---------------------------------------
        # Load MODIS
        # ---------------------------------------

        print("Loading MODIS...")

        modis_ds = gdal.Open(
            self.modis_path,
            gdal.GA_ReadOnly
        )

        if modis_ds is None:
            raise RuntimeError(
                f"Unable to open {self.modis_path}"
            )

        self.modis = (
            modis_ds
            .GetRasterBand(1)
            .ReadAsArray()
            .astype(np.float32)
        )

        # ---------------------------------------
        # Read accepted windows
        # ---------------------------------------

        print("Reading accepted window pairs...")

        self.windows = pd.read_csv(
            self.csv_path
        )

        print("----------------------------------------")
        print(f"AVHRR Shape      : {self.avhrr.shape}")
        print(f"MODIS Shape      : {self.modis.shape}")
        print(f"Accepted Windows : {len(self.windows)}")
        print("----------------------------------------")

        # ---------------------------------------
        # Output folders
        # ---------------------------------------

        # self.cross_power_dir = os.path.join(
        #     self.output_dir,
        #     "cross_power"
        # )

        self.correlation_dir = os.path.join(
            self.output_dir,
            "correlation_surface"
        )

        # os.makedirs(
        #     self.cross_power_dir,
        #     exist_ok=True
        # )

        os.makedirs(
            self.correlation_dir,
            exist_ok=True
        )

        # Summary list for later CSV export
        self.summary = []

    def _check_inputs(self):
        """
        Check that all required input files exist.
        """

        required_files = [
            self.avhrr_path,
            self.modis_path,
            self.csv_path
        ]

        for file in required_files:

            if not os.path.exists(file):

                raise FileNotFoundError(
                    f"\nInput file not found:\n{file}"
                )
    def get_window_pair(self, index):
        """
        Extract one accepted window pair.

        Since preprocessing has already aligned the pixel
        grids, the same window coordinates are used for
        both AVHRR and MODIS.

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """

        row = self.windows.iloc[index]

        window_id = int(row["window_id"])

        row_start = int(row["row_start"])
        row_end = int(row["row_end"])

        col_start = int(row["col_start"])
        col_end = int(row["col_end"])

        center_row = int(row["center_row"])
        center_col = int(row["center_col"])

        swath_coverage = float(row["swath_coverage"])
        cloud_percentage = float(row["cloud_percentage"])

        # ------------------------------------------
        # Extract windows
        # ------------------------------------------

        avhrr_window = self.avhrr[
            row_start:row_end,
            col_start:col_end
        ]

        modis_window = self.modis[
            row_start:row_end,
            col_start:col_end
        ]

        # ------------------------------------------
        # Validate dimensions
        # ------------------------------------------

        expected_rows = row_end - row_start
        expected_cols = col_end - col_start

        if avhrr_window.shape != (expected_rows, expected_cols):

            raise ValueError(
                f"AVHRR window {window_id} has invalid size."
            )

        if modis_window.shape != (expected_rows, expected_cols):

            raise ValueError(
                f"MODIS window {window_id} has invalid size."
            )

        # ------------------------------------------
        # Reject invalid windows
        # ------------------------------------------

        if np.isnan(avhrr_window).all():

            raise ValueError(
                f"AVHRR window {window_id} contains only NaNs."
            )

        if np.isnan(modis_window).all():

            raise ValueError(
                f"MODIS window {window_id} contains only NaNs."
            )

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

            "avhrr_window": avhrr_window,
            "modis_window": modis_window

        }

    def total_windows(self):
        """
        Return the number of accepted windows.
        """

        return len(self.windows)

    def process_window(self, index):
        """
        Process one accepted window.

        Workflow
        --------
        Read window metadata
                ↓
        Extract AVHRR window
                ↓
        Extract MODIS window
                ↓
        Compute FFT
                ↓
        Compute Cross-Power Spectrum
                ↓
        Compute Correlation Surface

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """

        window = self.get_window_pair(index)

        reference_window = window["modis_window"]

        target_window = window["avhrr_window"]

        correlation_surface = self.engine.compute(
            reference_window,
            target_window
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

            #"cross_power": cross_power,

            "correlation_surface": correlation_surface

        }


    def process_all_windows(self):
        """
        Process every accepted window.

        Returns
        -------
        list
        """

        results = []

        total = self.total_windows()

        print("\n----------------------------------------")
        print("Processing Accepted Windows")
        print("----------------------------------------")

        for index in range(total):

            try:

                result = self.process_window(index)

                results.append(result)

                print(
                    f"[{index + 1}/{total}] "
                    f"Window {result['window_id']} completed."
                )

            except Exception as error:

                print(
                    f"[{index + 1}/{total}] "
                    f"Skipped : {error}"
                )

                continue

        print("----------------------------------------")
        print(f"Processed : {len(results)}")
        print(f"Skipped   : {total - len(results)}")
        print("----------------------------------------")

        return results

    def save_results(self, results):
        """
        Save cross-power spectrum and correlation surface
        for each accepted window.

        Parameters
        ----------
        results : list
            Output from process_all_windows().
        """

        print("\n----------------------------------------")
        print("Saving Results")
        print("----------------------------------------")

        for result in results:

            window_id = result["window_id"]

            # # ----------------------------------
            # # Save Cross-Power Spectrum
            # # ----------------------------------

            # cross_power_file = os.path.join(

            #     self.cross_power_dir,

            #     f"cross_power_{window_id:05d}.npy"

            # )

            # np.save(

            #     cross_power_file,

            #     result["cross_power"]

            # )

            # ----------------------------------
            # Save Correlation Surface
            # ----------------------------------

            correlation_file = os.path.join(

                self.correlation_dir,

                f"correlation_surface_{window_id:05d}.npy"

            )

            np.save(

                correlation_file,

                result["correlation_surface"]

            )

            # ----------------------------------
            # Save summary information
            # ----------------------------------

            self.summary.append({

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

                # "cross_power_file":
                #     cross_power_file,

                "correlation_surface_file":
                    correlation_file

            })

        # --------------------------------------
        # Save Summary CSV
        # --------------------------------------

        summary_df = pd.DataFrame(self.summary)

        summary_file = os.path.join(

            self.output_dir,

            "phase_correlation_summary.csv"

        )

        summary_df.to_csv(

            summary_file,

            index=False

        )

        print(f"\nSaved {len(results)} windows.")
        print(f"Summary : {summary_file}")

        return summary_file


    def clear_memory(self):
        """
        Release memory after Stage 5 Step 16.
        """

        gc.collect()

        print("\nMemory cleared.")

    def print_statistics(self, results):
        """
        Print Stage 5 statistics.

        Parameters
        ----------
        results : list
        """

        print("\n========================================")
        print("Stage 5 Statistics")
        print("========================================")

        total = self.total_windows()

        successful = len(results)

        failed = total - successful

        print(f"Accepted Windows : {total}")
        print(f"Processed        : {successful}")
        print(f"Failed           : {failed}")

        if successful > 0:

            cloud = [
                r["cloud_percentage"]
                for r in results
            ]

            swath = [
                r["swath_coverage"]
                for r in results
            ]

            print("----------------------------------------")
            print(
                f"Average Cloud (%) : "
                f"{np.mean(cloud):.2f}"
            )

            print(
                f"Average Swath (%) : "
                f"{np.mean(swath):.2f}"
            )

        print("========================================")


    def verify_saved_files(self):
        """
        Verify that output files exist.

        Returns
        -------
        bool
        """

        corr_files = os.listdir(
            self.correlation_dir
        )

        if len(corr_files) != len(self.summary):
            raise RuntimeError(
                "Missing correlation surface files."
            )

        print("\nOutput verification successful.")

        print(
            f"Correlation surfaces : {len(corr_files)}"
        )

        return True


    def execute(self):
        """
        Complete execution of Stage 5 Step 16.

        Workflow
        --------
        Load Windows
              ↓
        Process All Windows
              ↓
        Save Results
              ↓
        Verify Outputs
              ↓
        Print Statistics
              ↓
        Cleanup

        Returns
        -------
        str
            Path to summary CSV.
        """

        results = self.process_all_windows()

        summary_file = self.save_results(
            results
        )

        self.verify_saved_files()

        self.print_statistics(
            results
        )

        self.clear_memory()

        return summary_file

    def get_output_summary(self):
        """
        Return a summary dictionary for downstream modules.

        Returns
        -------
        dict
        """

        return {

            "correlation_surface_directory":
                self.correlation_dir,

            "summary_csv":
                os.path.join(
                    self.output_dir,
                    "phase_correlation_summary.csv"
                )

        }


    def close(self):
        """
        Release memory and GDAL resources.
        """

        if hasattr(self, "avhrr"):
            del self.avhrr

        if hasattr(self, "modis"):
            del self.modis

        if hasattr(self, "windows"):
            del self.windows

        gc.collect()

        print("\nResources released successfully.")


    def run(self):
        """
        Main execution function.

        Returns
        -------
        dict
            Output information for the next stage.
        """

        print("\n========================================")
        print("Running Phase Correlation")
        print("========================================")

        self.execute()

        outputs = self.get_output_summary()

        self.close()

        print("\n========================================")
        print("Stage 5 Step 16 Finished")
        print("========================================")

        print("\nOutputs")

        print("----------------------------------------")
        #print(f"Cross Power Directory : {outputs['cross_power_directory']}")
        print(f"Correlation Directory : {outputs['correlation_surface_directory']}")
        print(f"Summary CSV           : {outputs['summary_csv']}")
        print("----------------------------------------")

        return outputs


    def __len__(self):
        """
        Number of accepted windows.
        """

        return len(self.windows)


    def __repr__(self):

        return (
            f"PhaseCorrelation("
            f"windows={len(self.windows)}, "
            f"output='{self.output_dir}')"
        )

if __name__ == "__main__":

    # =====================================================
    # Input Files
    # =====================================================

    avhrr_path = "/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_float32.tif"

    modis_path = "/home/bhaskar/Documents/ImageReg/2_outputs/05_modis_float32.tif"

    csv_path = "/home/bhaskar/Documents/ImageReg/34_outputs/accepted_window_pairs.csv"

    output_dir = "stage5_phase_correlation"

    # =====================================================
    # Create PhaseCorrelation Object
    # =====================================================

    phase = PhaseCorrelation(

        avhrr_path=avhrr_path,

        modis_path=modis_path,

        csv_path=csv_path,

        output_dir=output_dir

    )

    # =====================================================
    # Run Stage 5 Step 16
    # =====================================================

    outputs = phase.run()

    # =====================================================
    # Print Output Information
    # =====================================================

    print("\n========================================")
    print("Execution Completed Successfully")
    print("========================================")

    print("\nOutput Files")
    print("----------------------------------------")

    # print(
    #     f"Cross-Power Directory\n"
    #     f"{outputs['cross_power_directory']}"
    # )

    # print()

    print(
        f"Correlation Surface Directory\n"
        f"{outputs['correlation_surface_directory']}"
    )

    print()

    print(
        f"Summary CSV\n"
        f"{outputs['summary_csv']}"
    )

    print("----------------------------------------")