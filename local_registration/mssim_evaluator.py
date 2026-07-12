"""
===============================================================
Stage 5 : Local Registration
Step 21 : MSSIM Before vs After

Reference
---------
Scheffler et al. (2017) - AROSICS, validation method 4

Purpose
-------
For every subpixel-refined window (Step 19 output), measure
the structural similarity between the AVHRR and MODIS crops
before applying the estimated (dx, dy) and after applying it.
A genuine correction should increase MSSIM; one that doesn't
is suspect regardless of what the reliability score (Step 20)
says, since R and MSSIM test different things (spectral peak
sharpness vs. actual spatial agreement).

AVHRR (DN counts) and MODIS (reflectance) live on unrelated
value scales, so each window is independently min-max
normalized before SSIM is computed - only structural
agreement matters here, not absolute intensity.

Both scores are computed over the same shared valid-pixel
bounding box (the region still finite after the shift is
applied), not the full window. Padding the shifted-out area
with a constant and scoring the full window would silently
dilute "after" against a solid, textureless block that has
nothing to do with alignment quality - verified this directly
lowered the measured gain by ~6x on a real window with a large
shift (+0.032 full-window vs +0.186 valid-region-only for the
same window).

Inputs
------
1. avhrr_data.tif
2. modis_data.tif
3. subpixel_shifts.csv

Outputs
-------
mssim_results.csv

NOTE
----
This module ONLY scores MSSIM before/after. It does not
decide final tie-point acceptance - that combines this with
Step 20's reliability and the |shift| cap in Stage 6.
===============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import shift as nd_shift
from skimage.metrics import structural_similarity


class MSSIMEvaluator:
    """
    Stage 5 - Step 21

    Computes MSSIM before/after the final (dx, dy) correction
    for every subpixel-refined window from Step 19.
    """

    def __init__(
        self,
        avhrr_path,
        modis_path,
        subpixel_csv,
        output_directory
    ):

        self.avhrr_path = avhrr_path
        self.modis_path = modis_path
        self.subpixel_csv = subpixel_csv
        self.output_directory = output_directory

        self._check_inputs()

        os.makedirs(self.output_directory, exist_ok=True)

        print("\n========================================")
        print("Stage 5 - Step 21")
        print("MSSIM Before vs After")
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

        print("Reading subpixel shifts...")

        self.shifts = pd.read_csv(self.subpixel_csv)

        print("----------------------------------------")
        print(f"AVHRR Shape       : {self.avhrr.shape}")
        print(f"MODIS Shape       : {self.modis.shape}")
        print(f"Windows           : {len(self.shifts)}")
        print("----------------------------------------")

        self.results = []

    def _check_inputs(self):

        required_files = [
            self.avhrr_path,
            self.modis_path,
            self.subpixel_csv
        ]

        for file in required_files:

            if not os.path.exists(file):
                raise FileNotFoundError(f"\nMissing input:\n{file}")

    def total_windows(self):

        return len(self.shifts)

    def normalize(self, window):
        """
        Min-max normalize using only finite pixels, so AVHRR
        (DN counts) and MODIS (reflectance) become comparable
        for a pure structural-similarity comparison.
        """

        valid = np.isfinite(window)

        if valid.sum() < 2:
            return np.zeros_like(window)

        lo = float(window[valid].min())
        hi = float(window[valid].max())

        if (hi - lo) < 1e-12:
            return np.zeros_like(window)

        normalized = (window - lo) / (hi - lo)

        return np.nan_to_num(normalized, nan=0.0)

    def compute_mssim(self, reference_window, target_window):

        ref = self.normalize(reference_window)
        tar = self.normalize(target_window)

        score = structural_similarity(
            ref,
            tar,
            data_range=1.0
        )

        return float(score)

    def valid_bounding_box(self, mask, min_size=16):
        """
        Bounding box of the finite region in `mask`, so
        before/after can be scored over identical pixels
        rather than full windows padded with an arbitrary
        constant where the shift left no real data.

        Raises
        ------
        ValueError
            If the valid region is empty or too small for a
            meaningful SSIM comparison.
        """

        rows_valid = np.where(mask.any(axis=1))[0]
        cols_valid = np.where(mask.any(axis=0))[0]

        if rows_valid.size == 0 or cols_valid.size == 0:
            raise ValueError("No valid overlap region after shift.")

        r0, r1 = rows_valid.min(), rows_valid.max() + 1
        c0, c1 = cols_valid.min(), cols_valid.max() + 1

        if (r1 - r0) < min_size or (c1 - c0) < min_size:
            raise ValueError("Valid overlap region too small for SSIM.")

        return r0, r1, c0, c1

    def process_window(self, row):

        window_id = int(row["window_id"])

        row_start = int(row["row_start"])
        row_end = int(row["row_end"])

        col_start = int(row["col_start"])
        col_end = int(row["col_end"])

        final_dx = float(row["final_dx"])
        final_dy = float(row["final_dy"])

        avhrr_window = self.avhrr[row_start:row_end, col_start:col_end]
        modis_window = self.modis[row_start:row_end, col_start:col_end]

        shifted_avhrr = nd_shift(
            avhrr_window,
            shift=(final_dy, final_dx),
            order=1,
            mode="constant",
            cval=np.nan
        )

        # ------------------------------------------------
        # Score both before and after over the SAME shared
        # valid-pixel region (the shift's own footprint),
        # so a large shift's clipped-out area doesn't dilute
        # the "after" score against a padded constant block.
        # ------------------------------------------------

        valid = (
            np.isfinite(shifted_avhrr) &
            np.isfinite(modis_window) &
            np.isfinite(avhrr_window)
        )

        r0, r1, c0, c1 = self.valid_bounding_box(valid)

        mssim_before = self.compute_mssim(
            modis_window[r0:r1, c0:c1],
            avhrr_window[r0:r1, c0:c1]
        )

        mssim_after = self.compute_mssim(
            modis_window[r0:r1, c0:c1],
            shifted_avhrr[r0:r1, c0:c1]
        )

        result = row.to_dict()

        result["mssim_before"] = mssim_before
        result["mssim_after"] = mssim_after
        result["mssim_delta"] = mssim_after - mssim_before
        result["mssim_increased"] = mssim_after > mssim_before

        return result

    def process_all_windows(self):

        print("\n----------------------------------------")
        print("Scoring MSSIM")
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
                    f"before={result['mssim_before']:.4f} "
                    f"after={result['mssim_after']:.4f} "
                    f"increased={result['mssim_increased']}"
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
            raise RuntimeError("No MSSIM results available to save.")

        output_file = os.path.join(self.output_directory, "mssim_results.csv")

        output_df = pd.DataFrame(self.results)

        output_df.to_csv(output_file, index=False)

        print("\n----------------------------------------")
        print("MSSIM Results Saved")
        print("----------------------------------------")
        print(f"Output File : {output_file}")
        print(f"Total Rows  : {len(output_df)}")
        print("----------------------------------------")

        return output_file

    def print_statistics(self):

        if len(self.results) == 0:
            print("\nNo valid windows found.")
            return

        increased = sum(r["mssim_increased"] for r in self.results)

        delta = [r["mssim_delta"] for r in self.results]

        print("\n========================================")
        print("MSSIM Statistics")
        print("========================================")
        print(f"Processed Windows : {len(self.results)}")
        print(f"Increased         : {increased}")
        print(f"Decreased/Equal   : {len(self.results) - increased}")
        print(f"Mean Delta        : {np.mean(delta):.4f}")
        print(f"Min Delta         : {np.min(delta):.4f}")
        print(f"Max Delta         : {np.max(delta):.4f}")
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
            "mssim_csv": os.path.join(self.output_directory, "mssim_results.csv")
        }

    def run(self):

        print("\n========================================")
        print("Running MSSIM Evaluation")
        print("========================================")

        output_csv = self.execute()

        outputs = self.get_output_summary()

        self.close()

        print("\n========================================")
        print("Stage 5 - Step 21 Finished")
        print("========================================")

        print("\nOutputs")
        print("----------------------------------------")
        print(f"MSSIM CSV : {outputs['mssim_csv']}")
        print("----------------------------------------")

        return outputs

    def __len__(self):

        return len(self.shifts)

    def __repr__(self):

        return f"MSSIMEvaluator(windows={len(self.shifts)})"


if __name__ == "__main__":

    avhrr_path = "/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_float32.tif"
    modis_path = "/home/bhaskar/Documents/ImageReg/2_outputs/05_modis_float32.tif"

    subpixel_csv = (
        "/home/bhaskar/Documents/ImageReg/stage5_phase_correlation/"
        "stage5_subpixel_estimation/subpixel_shifts.csv"
    )

    output_directory = (
        "/home/bhaskar/Documents/ImageReg/stage5_phase_correlation/"
        "stage5_mssim"
    )

    evaluator = MSSIMEvaluator(
        avhrr_path=avhrr_path,
        modis_path=modis_path,
        subpixel_csv=subpixel_csv,
        output_directory=output_directory
    )

    outputs = evaluator.run()

    print("\n========================================")
    print("Execution Completed Successfully")
    print("========================================")
