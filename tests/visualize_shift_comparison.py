"""
Visual audit of Stage 5 (steps 16-21) output.

For a representative sample of windows, plots three panels
side by side:

    original AVHRR crop | AVHRR shifted by (final_dx, final_dy) | MODIS crop

Each crop is independently min-max normalized (same treatment
Step 21 uses internally) purely for display contrast - it does
not affect any of the pipeline's numeric outputs.

The sample is chosen to span the quality spectrum rather than
being cherry-picked: highest reliability, lowest reliability,
best/worst real MSSIM change, a typical (median) window, and
any case where reliability and MSSIM disagree.

Output
------
gallery/shift_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.ndimage import shift as nd_shift

gdal.UseExceptions()

ROOT = os.path.join(os.path.dirname(__file__), "..")

AVHRR_FILE = os.path.join(ROOT, "2_outputs", "05_avhrr_float32.tif")
MODIS_FILE = os.path.join(ROOT, "2_outputs", "05_modis_float32.tif")

RELIABILITY_CSV = os.path.join(
    ROOT, "stage5_phase_correlation", "stage5_reliability", "reliability_results.csv"
)
MSSIM_CSV = os.path.join(
    ROOT, "stage5_phase_correlation", "stage5_mssim", "mssim_results.csv"
)

OUTPUT_FILE = os.path.join(ROOT, "gallery", "shift_comparison.png")


def normalize(window):

    valid = np.isfinite(window)

    if valid.sum() < 2:
        return np.zeros_like(window)

    lo = float(window[valid].min())
    hi = float(window[valid].max())

    if (hi - lo) < 1e-12:
        return np.zeros_like(window)

    return np.nan_to_num((window - lo) / (hi - lo), nan=0.0)


def select_sample(df):
    """
    Pick a representative set of windows, tagged with why
    each one was chosen. Avoids duplicates.
    """

    picks = {}

    picks[df.loc[df["reliability"].idxmax(), "window_id"]] = "highest R"
    picks[df.loc[df["reliability"].idxmin(), "window_id"]] = "lowest R"
    picks[df.loc[df["mssim_delta"].idxmax(), "window_id"]] = "best MSSIM gain"
    picks[df.loc[df["mssim_delta"].idxmin(), "window_id"]] = "worst MSSIM loss"

    median_r = df["reliability"].median()
    median_row = df.iloc[(df["reliability"] - median_r).abs().argsort().iloc[0]]
    picks.setdefault(median_row["window_id"], "median R (typical)")

    rejected = df[~df["reliability_accepted"]]

    if len(rejected) > 0:
        disagreement = rejected.loc[rejected["mssim_delta"].idxmax(), "window_id"]
        picks.setdefault(disagreement, "R rejected, but MSSIM improved a lot")

    return picks


def main():

    rel = pd.read_csv(RELIABILITY_CSV)
    mssim = pd.read_csv(MSSIM_CSV)[
        ["window_id", "mssim_before", "mssim_after", "mssim_delta", "mssim_increased"]
    ]

    df = rel.merge(mssim, on="window_id")

    picks = select_sample(df)

    avhrr_ds = gdal.Open(AVHRR_FILE)
    modis_ds = gdal.Open(MODIS_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    window_ids = list(picks.keys())

    fig, axes = plt.subplots(
        len(window_ids), 3,
        figsize=(9, 3 * len(window_ids))
    )

    for row_index, window_id in enumerate(window_ids):

        row = df[df["window_id"] == window_id].iloc[0]

        r0, r1 = int(row.row_start), int(row.row_end)
        c0, c1 = int(row.col_start), int(row.col_end)

        dx, dy = float(row.final_dx), float(row.final_dy)

        avhrr_crop = avhrr[r0:r1, c0:c1]
        modis_crop = modis[r0:r1, c0:c1]

        shifted_crop = nd_shift(
            avhrr_crop, shift=(dy, dx), order=1, mode="constant", cval=np.nan
        )

        panels = [
            (normalize(avhrr_crop), "AVHRR (original)"),
            (normalize(shifted_crop), "AVHRR (shifted)"),
            (normalize(modis_crop), "MODIS (reference)")
        ]

        for col_index, (image, label) in enumerate(panels):

            ax = axes[row_index, col_index]

            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_index == 0:
                ax.set_title(label, fontsize=10)

        reason = picks[window_id]

        axes[row_index, 0].set_ylabel(
            f"win {window_id}\n{reason}",
            fontsize=8
        )

        axes[row_index, 1].set_title(
            f"dx={dx:.2f} dy={dy:.2f}   R={row.reliability:.1f}"
            f"{' (accepted)' if row.reliability_accepted else ' (rejected)'}\n"
            f"MSSIM {row.mssim_before:.3f} -> {row.mssim_after:.3f} "
            f"({'+' if row.mssim_delta >= 0 else ''}{row.mssim_delta:.3f})",
            fontsize=8
        )

    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    plt.savefig(OUTPUT_FILE, dpi=130)

    print(f"Saved: {OUTPUT_FILE}")

    for window_id, reason in picks.items():
        print(f"  window {window_id}: {reason}")


if __name__ == "__main__":
    main()
