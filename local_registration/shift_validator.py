"""
===============================================================
Stage 5 : Local Registration
Step 18 : Integer Shift Validation

Reference
---------
1. Foroosh et al. (2002)
2. Scheffler et al. (2017) - AROSICS

Purpose
-------
Validate the integer shift estimated in Step 17.

Workflow
--------
1. Read integer shift
2. Apply integer shift to AVHRR window
3. Recompute phase correlation
4. Check whether peak moves to (0,0)
5. Repeat up to 5 iterations if necessary

Inputs
------
1. avhrr_data.tif
2. modis_data.tif
3. integer_peak_results.csv

Outputs
-------
validated_integer_shifts.csv

NOTE
----
This module performs ONLY integer shift validation.
Subpixel estimation begins in Step 19.

SIGN CONVENTION (load-bearing, verified empirically - see
validate_window)
----------------------------------------------------------
engine.compute(modis, avhrr) returns a raw peak d meaning
"avhrr looks like modis shifted by +d". The correction that
aligns avhrr onto modis is shift(avhrr, -d), the negative.
validated_dx/validated_dy in this module's output (and
everything downstream: subpixel_shifts.csv's final_dx/dy,
mssim_results.csv) are the CORRECTION - i.e. directly usable
as scipy.ndimage.shift(avhrr_window, shift=(validated_dy,
validated_dx)) - not the raw peak. initial_dx/initial_dy keep
the raw Step 17 peak for reference and are NOT directly usable
as a shift argument.
===============================================================
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import shift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_registration.core.phase_engine import PhaseCorrelationEngine


class ShiftValidator:
    """
    Stage 5 - Step 18

    Validate integer shifts using the iterative
    AROSICS workflow.
    """

    def __init__(
        self,
        avhrr_path,
        modis_path,
        peak_csv,
        output_directory,
        max_iterations=5,
        min_peak_value=1e-6,
        min_valid_fraction=0.5
    ):

        self.avhrr_path = avhrr_path
        self.modis_path = modis_path
        self.peak_csv = peak_csv
        self.output_directory = output_directory
        self.max_iterations = max_iterations
        self.min_peak_value = min_peak_value
        self.min_valid_fraction = min_valid_fraction
        self.engine = PhaseCorrelationEngine()
        self.eps = 1e-12

        self._check_inputs()

        os.makedirs(
            self.output_directory,
            exist_ok=True
        )

        print("\n========================================")
        print("Stage 5 - Step 18")
        print("Integer Shift Validation")
        print("========================================")

        # -------------------------------------
        # Load AVHRR
        # -------------------------------------

        print("\nLoading AVHRR...")

        avhrr_ds = gdal.Open(
            self.avhrr_path,
            gdal.GA_ReadOnly
        )

        if avhrr_ds is None:

            raise RuntimeError(
                "Unable to open AVHRR image."
            )

        self.avhrr = (

            avhrr_ds
            .GetRasterBand(1)
            .ReadAsArray()
            .astype(np.float32)

        )

        # -------------------------------------
        # Load MODIS
        # -------------------------------------

        print("Loading MODIS...")

        modis_ds = gdal.Open(
            self.modis_path,
            gdal.GA_ReadOnly
        )

        if modis_ds is None:

            raise RuntimeError(
                "Unable to open MODIS image."
            )

        self.modis = (

            modis_ds
            .GetRasterBand(1)
            .ReadAsArray()
            .astype(np.float32)

        )

        # -------------------------------------
        # Load Integer Peak Results
        # -------------------------------------

        print("Reading integer peak results...")

        self.peaks = pd.read_csv(
            self.peak_csv
        )

        print("----------------------------------------")
        print(f"AVHRR Shape : {self.avhrr.shape}")
        print(f"MODIS Shape : {self.modis.shape}")
        print(f"Total Windows : {len(self.peaks)}")
        print(f"Maximum Iterations : {self.max_iterations}")
        print("----------------------------------------")

        self.results = []


    def _check_inputs(self):
        """
        Verify required input files exist.
        """

        required_files = [

            self.avhrr_path,

            self.modis_path,

            self.peak_csv

        ]

        for file in required_files:

            if not os.path.exists(file):

                raise FileNotFoundError(

                    f"\nMissing input:\n{file}"

                )
    def get_window(self, index):
        """
        Load one window and its integer shift.

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """

        row = self.peaks.iloc[index]

        window_id = int(row["window_id"])

        row_start = int(row["row_start"])
        row_end = int(row["row_end"])

        col_start = int(row["col_start"])
        col_end = int(row["col_end"])

        center_row = int(row["center_row"])
        center_col = int(row["center_col"])

        dy_integer = int(row["dy_integer"])
        dx_integer = int(row["dx_integer"])

        peak_row = int(row["peak_row"])
        peak_col = int(row["peak_col"])

        peak_value = float(row["peak_value"])

        swath_coverage = float(row["swath_coverage"])
        cloud_percentage = float(row["cloud_percentage"])

        # --------------------------------------
        # Extract windows
        # --------------------------------------

        avhrr_window = self.avhrr[
            row_start:row_end,
            col_start:col_end
        ]

        modis_window = self.modis[
            row_start:row_end,
            col_start:col_end
        ]

        # --------------------------------------
        # Validate window dimensions
        # --------------------------------------

        expected_rows = row_end - row_start
        expected_cols = col_end - col_start

        if avhrr_window.shape != (expected_rows, expected_cols):

            raise ValueError(

                f"Window {window_id}: "
                "Invalid AVHRR window."

            )

        if modis_window.shape != (expected_rows, expected_cols):

            raise ValueError(

                f"Window {window_id}: "
                "Invalid MODIS window."

            )

        return {

            "window_id": window_id,

            "row_start": row_start,
            "row_end": row_end,

            "col_start": col_start,
            "col_end": col_end,

            "center_row": center_row,
            "center_col": center_col,

            "peak_row": peak_row,
            "peak_col": peak_col,

            "peak_value": peak_value,

            "dy_integer": dy_integer,
            "dx_integer": dx_integer,

            "swath_coverage": swath_coverage,
            "cloud_percentage": cloud_percentage,

            "rows": expected_rows,
            "cols": expected_cols,

            "avhrr_window": avhrr_window,

            "modis_window": modis_window

        }


    def total_windows(self):
        """
        Return total number of windows.
        """

        return len(self.peaks)

    def apply_integer_shift(
        self,
        image,
        dy,
        dx
    ):
        """
        Apply the estimated integer shift to the
        target (AVHRR) window.

        Parameters
        ----------
        image : ndarray

        dy : int

        dx : int

        Returns
        -------
        ndarray
        """

        shifted = shift(

            image,

            shift=(dy, dx),

            order=0,

            mode="constant",

            cval=np.nan,

            prefilter=False

        )

        return shifted


    def valid_overlap_fraction(
        self,
        shifted_avhrr,
        modis_window
    ):
        """
        Fraction of pixels that are finite in both images after
        the integer shift has been applied.

        A shift whose magnitude approaches or exceeds the window
        size pushes most/all content out of frame (filled with
        NaN by apply_integer_shift), leaving nothing real to
        correlate against.
        """

        valid = (
            np.isfinite(shifted_avhrr) &
            np.isfinite(modis_window)
        )

        return float(np.mean(valid))


    # def compute_correlation_surface(
    #     self,
    #     reference_window,
    #     target_window
    # ):
    #     """
    #     Recompute the correlation surface after
    #     applying the integer shift.

    #     Returns
    #     -------
    #     ndarray
    #     """

    #     # ----------------------------------
    #     # Replace NaN / Inf
    #     # ----------------------------------

    #     reference = np.nan_to_num(
    #         reference_window,
    #         nan=0.0,
    #         posinf=0.0,
    #         neginf=0.0
    #     )

    #     target = np.nan_to_num(
    #         target_window,
    #         nan=0.0,
    #         posinf=0.0,
    #         neginf=0.0
    #     )

    #     # ----------------------------------
    #     # FFT
    #     # ----------------------------------

    #     F_ref = np.fft.fft2(reference)

    #     F_tar = np.fft.fft2(target)

    #     # ----------------------------------
    #     # Cross-power spectrum
    #     # ----------------------------------

    #     cross_power = F_tar * np.conjugate(F_ref)

    #     magnitude = np.abs(cross_power)

    #     cross_power = np.divide(
    #         cross_power,
    #         magnitude,
    #         out=np.zeros_like(cross_power),
    #         where=magnitude > self.eps
    #     )

    #     # ----------------------------------
    #     # Correlation surface
    #     # ----------------------------------

    #     correlation = np.fft.ifft2(
    #         cross_power
    #     )

    #     correlation = np.real(
    #         correlation
    #     )

    #     return correlation.astype(
    #         np.float32
    #     )


    def validate_once(
        self,
        window
    ):
        """
        Perform one validation iteration.

        Returns
        -------
        dict
        """

        shifted_avhrr = self.apply_integer_shift(

            window["avhrr_window"],

            window["dy_integer"],

            window["dx_integer"]

        )

        valid_fraction = self.valid_overlap_fraction(
            shifted_avhrr,
            window["modis_window"]
        )

        correlation_surface = self.engine.compute(
            window["modis_window"],
            shifted_avhrr
        )

        return {

            "window": window,

            "shifted_avhrr": shifted_avhrr,

            "correlation_surface": correlation_surface,

            "valid_fraction": valid_fraction

        }

    def find_peak(
        self,
        correlation_surface
    ):
        """
        Find the peak in the correlation surface.

        Parameters
        ----------
        correlation_surface : ndarray

        Returns
        -------
        peak_row : int

        peak_col : int

        peak_value : float
        """

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


    def convert_to_shift(
        self,
        peak_row,
        peak_col,
        rows,
        cols
    ):
        """
        Convert FFT peak coordinates into
        integer shift.

        Returns
        -------
        dy : int

        dx : int
        """

        if peak_row > rows // 2:

            dy = peak_row - rows

        else:

            dy = peak_row

        if peak_col > cols // 2:

            dx = peak_col - cols

        else:

            dx = peak_col

        return (

            dy,

            dx

        )


    def check_validation(
        self,
        validation_result
    ):
        """
        Check whether the shifted image is
        correctly aligned.

        Parameters
        ----------
        validation_result : dict

        Returns
        -------
        dict
        """

        surface = validation_result[
            "correlation_surface"
        ]

        valid_fraction = validation_result[
            "valid_fraction"
        ]

        rows, cols = surface.shape

        peak_row, peak_col, peak_value = self.find_peak(
            surface
        )

        remaining_dy, remaining_dx = self.convert_to_shift(

            peak_row,

            peak_col,

            rows,

            cols

        )

        shift_converged = (

            remaining_dx == 0 and

            remaining_dy == 0

        )

        # ----------------------------------------
        # A shift that pushes most of the window
        # out of frame collapses to an all-zero
        # correlation surface, whose peak trivially
        # sits at (0, 0). Reject that instead of
        # treating it as convergence.
        # ----------------------------------------

        reliable = (

            valid_fraction >= self.min_valid_fraction and

            peak_value >= self.min_peak_value

        )

        degenerate = shift_converged and not reliable

        validated = shift_converged and reliable

        return {

            "window": validation_result["window"],

            "peak_row": peak_row,

            "peak_col": peak_col,

            "peak_value": peak_value,

            "remaining_dx": remaining_dx,

            "remaining_dy": remaining_dy,

            "valid_fraction": valid_fraction,

            "degenerate": degenerate,

            "validated": validated

        }

    def validate_window(
        self,
        window
    ):
        """
        Validate one window using the AROSICS
        iterative integer-shift validation.

        Parameters
        ----------
        window : dict

        Returns
        -------
        dict
        """

        # ------------------------------------------------
        # Sign convention
        # ------------------------------------------------
        # engine.compute(modis, avhrr) returns the raw peak
        # d such that avhrr looks like modis shifted by +d
        # (verified empirically: building a synthetic avhrr
        # via scipy.ndimage.shift(modis, shift=(dy,dx)) and
        # running it through engine.compute recovers exactly
        # (dy,dx)). The CORRECTION that aligns avhrr back onto
        # modis is therefore shift(avhrr, -d), not shift(avhrr,
        # +d) - confirmed by direct MSE/NCC comparison on real
        # window data (shift=-d cut residual error by ~2-8x
        # relative to shift=+d, which instead compounds the
        # offset). total_dx/total_dy here track that correction
        # directly, starting at -(raw peak) and refined by
        # subtracting (not adding) each iteration's residual
        # raw peak.
        # ------------------------------------------------

        total_dx = -window["dx_integer"]
        total_dy = -window["dy_integer"]

        current_window = window.copy()

        iteration = 0

        validated = False

        peak_value = None
        peak_row = None
        peak_col = None
        valid_fraction = None
        degenerate = False

        while iteration < self.max_iterations:

            iteration += 1

            # ----------------------------------
            # Apply current shift
            # ----------------------------------

            current_window["dx_integer"] = total_dx
            current_window["dy_integer"] = total_dy

            validation = self.validate_once(
                current_window
            )

            result = self.check_validation(
                validation
            )

            peak_row = result["peak_row"]
            peak_col = result["peak_col"]
            peak_value = result["peak_value"]
            valid_fraction = result["valid_fraction"]
            degenerate = result["degenerate"]

            # ----------------------------------
            # Peak moved to origin?
            # ----------------------------------

            if result["validated"]:

                validated = True

                break

            # ----------------------------------
            # Shift has pushed the window out of
            # frame - further iterations cannot
            # recover, stop instead of wasting
            # iterations on a garbage surface.
            # ----------------------------------

            if degenerate:

                break

            # ----------------------------------
            # Update total correction estimate.
            # "remaining" is a raw peak (same sign
            # convention as the initial detection), so
            # subtract it to keep total_dx/dy as the
            # correction to apply, not the raw offset.
            # ----------------------------------

            total_dx -= result["remaining_dx"]

            total_dy -= result["remaining_dy"]

        return {

            "window_id":
                window["window_id"],

            "row_start":
                window["row_start"],

            "row_end":
                window["row_end"],

            "col_start":
                window["col_start"],

            "col_end":
                window["col_end"],

            "center_row":
                window["center_row"],

            "center_col":
                window["center_col"],

            "swath_coverage":
                window["swath_coverage"],

            "cloud_percentage":
                window["cloud_percentage"],

            "initial_dx":
                window["dx_integer"],

            "initial_dy":
                window["dy_integer"],

            "validated_dx":
                total_dx,

            "validated_dy":
                total_dy,

            "iterations":
                iteration,

            "validated":
                validated,

            "degenerate":
                degenerate,

            "valid_fraction":
                valid_fraction,

            "peak_row":
                peak_row,

            "peak_col":
                peak_col,

            "peak_value":
                peak_value

        }


    def validate_all_windows(self):
        """
        Validate all windows.

        Returns
        -------
        list
        """

        print("\n----------------------------------------")
        print("Validating Integer Shifts")
        print("----------------------------------------")

        self.results = []

        total = self.total_windows()

        for index in range(total):

            try:

                window = self.get_window(index)

                result = self.validate_window(
                    window
                )

                self.results.append(result)

                print(
                    f"[{index+1}/{total}] "
                    f"Window {result['window_id']} "
                    f"Iterations={result['iterations']} "
                    f"Validated={result['validated']}"
                )

            except Exception as error:

                print(
                    f"[{index+1}/{total}] "
                    f"Skipped : {error}"
                )

        print("----------------------------------------")
        print(f"Validated : {len(self.results)}")
        print("----------------------------------------")

        return self.results

    def save_results(self):
        """
        Save validated integer shifts.

        Output
        ------
        validated_integer_shifts.csv
        """

        if len(self.results) == 0:

            raise RuntimeError(
                "No validation results available."
            )

        output_file = os.path.join(

            self.output_directory,

            "validated_integer_shifts.csv"

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

                "initial_dx":
                    result["initial_dx"],

                "initial_dy":
                    result["initial_dy"],

                "validated_dx":
                    result["validated_dx"],

                "validated_dy":
                    result["validated_dy"],

                "iterations":
                    result["iterations"],

                "validated":
                    result["validated"],

                "degenerate":
                    result["degenerate"],

                "valid_fraction":
                    result["valid_fraction"],

                "peak_row":
                    result["peak_row"],

                "peak_col":
                    result["peak_col"],

                "peak_value":
                    result["peak_value"]

            })

        output_df = pd.DataFrame(output_rows)

        output_df.to_csv(

            output_file,

            index=False

        )

        print("\n----------------------------------------")
        print("Validation Results Saved")
        print("----------------------------------------")
        print(f"Output File : {output_file}")
        print(f"Total Rows  : {len(output_df)}")
        print("----------------------------------------")

        return output_file


    def print_statistics(self):
        """
        Print validation statistics.
        """

        total = len(self.results)

        validated = sum(

            result["validated"]

            for result in self.results

        )

        failed = total - validated

        degenerate = sum(

            result["degenerate"]

            for result in self.results

        )

        iterations = [

            result["iterations"]

            for result in self.results

        ]

        print("\n========================================")
        print("Validation Statistics")
        print("========================================")

        print(f"Total Windows     : {total}")
        print(f"Validated         : {validated}")
        print(f"Failed            : {failed}")
        print(f"Degenerate        : {degenerate}")

        if total > 0:

            print(
                f"Average Iterations : "
                f"{np.mean(iterations):.2f}"
            )

            print(
                f"Maximum Iterations : "
                f"{np.max(iterations)}"
            )

        print("========================================")


    def execute(self):
        """
        Execute Step 18.

        Workflow
        --------
        Load Window
              ↓
        Validate Integer Shift
              ↓
        Save Results
              ↓
        Print Statistics

        Returns
        -------
        str
            Path to validated_integer_shifts.csv
        """

        self.validate_all_windows()

        self.print_statistics()

        output_csv = self.save_results()

        gc.collect()

        return output_csv

    def close(self):
        """
        Release memory.
        """

        if hasattr(self, "avhrr"):
            del self.avhrr

        if hasattr(self, "modis"):
            del self.modis

        if hasattr(self, "peaks"):
            del self.peaks

        gc.collect()

        print("\nResources released successfully.")


    def get_output_summary(self):
        """
        Return output information for the next stage.

        Returns
        -------
        dict
        """

        return {

            "validated_shift_csv": os.path.join(

                self.output_directory,

                "validated_integer_shifts.csv"

            )

        }


    def run(self):
        """
        Execute the complete validation workflow.

        Returns
        -------
        dict
        """

        print("\n========================================")
        print("Running Integer Shift Validation")
        print("========================================")

        output_csv = self.execute()

        outputs = self.get_output_summary()

        self.close()

        print("\n========================================")
        print("Stage 5 - Step 18 Finished")
        print("========================================")

        print("\nOutputs")
        print("----------------------------------------")

        print(
            f"Validated Shift CSV :\n"
            f"{output_csv}"
        )

        print("----------------------------------------")

        return outputs


    def __len__(self):
        """
        Number of windows.
        """

        return len(self.peaks)


    def __repr__(self):

        return (

            f"ShiftValidator("
            f"windows={len(self.peaks)}, "
            f"max_iterations={self.max_iterations})"

        )

if __name__ == "__main__":

    # =====================================================
    # Input Files
    # =====================================================

    avhrr_path = (
        "/home/bhaskar/Documents/ImageReg/"
        "2_outputs/05_avhrr_float32.tif"
    )

    modis_path = (
        "/home/bhaskar/Documents/ImageReg/"
        "2_outputs/05_modis_float32.tif"
    )

    peak_csv = (
        "stage5_peak_detection/"
        "integer_peak_results.csv"
    )

    output_directory = (
        "stage5_shift_validation"
    )

    # =====================================================
    # Create Validator
    # =====================================================

    validator = ShiftValidator(

        avhrr_path=avhrr_path,

        modis_path=modis_path,

        peak_csv=peak_csv,

        output_directory=output_directory,

        max_iterations=5

    )

    # =====================================================
    # Execute
    # =====================================================

    outputs = validator.run()

    # =====================================================
    # Display Results
    # =====================================================

    print("\n========================================")
    print("Stage 5 - Step 18 Completed")
    print("========================================")

    print("\nOutput Files")
    print("----------------------------------------")

    print(
        f"Validated Shift CSV\n"
        f"{outputs['validated_shift_csv']}"
    )

    print("----------------------------------------")