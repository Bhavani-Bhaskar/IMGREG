"""
main.py
-------

Preprocessing pipeline for MODIS–AVHRR image registration.
"""

import os
from configparser import ConfigParser

from preprocessing.projection import match_projection
from preprocessing.overlap import extract_common_overlap
from preprocessing.band_selection import select_band
from preprocessing.datatype import convert_to_float32
from preprocessing.geotransform import verify_geotransform
from preprocessing.valid_mask import create_valid_mask
from preprocessing.quality_check import quality_check

from window_selection.window_pair_builder import read_images
from window_selection.window_pair_builder import build_window_pairs
from window_selection.window_generator import generate_windows
from window_selection.window_filter import load_mask , filter_window_pairs
from window_selection.csv_writer import (
    save_accepted_pairs_csv,
    save_rejected_pairs_csv,
)

from local_registration.phase_correlation import PhaseCorrelation
from local_registration.peak_detector import PeakDetector
from local_registration.shift_validator import ShiftValidator
from local_registration.subpixel_estimator import SubpixelEstimator
from local_registration.reliability_estimator import ReliabilityEstimator
from local_registration.mssim_evaluator import MSSIMEvaluator

# ============================================================
# Read configuration
# ============================================================

config = ConfigParser()
config.read("config.txt")

debug = config.getboolean("General", "debug")

# ============================================================
# Pipeline Control
# ============================================================

PREPROCESSING = config.getboolean(
    "Pipeline",
    "preprocessing"
)

CLOUD_DETECTION = config.getboolean(
    "Pipeline",
    "cloud_detection"
)

LOCAL_REGISTRATION = config.getboolean(
    "Pipeline",
    "local_registration"
)

resampling = config["Resampling"]["resampling"]

reference_band = config.getint(
    "Registration Band",
    "reference_band"
)

target_band = config.getint(
    "Registration Band",
    "target_band"
)

kernel_size = config.getint(
    "Valid Mask",
    "kernel_size"
)

threshold = config.getfloat(
    "Valid Mask",
    "valid_threshold"
)

tolerance = config.getfloat(
    "GeoTransform",
    "geotransform_tolerance"
)


# ---------------------------------------------
# Stage 3 Parameters
# ---------------------------------------------

WINDOW_SIZE = config.getint(
    "Stage 3",
    "window_size"
)

STRIDE_RATIO = config.getfloat(
    "Stage 3",
    "stride_ratio"
)

STRIDE = int(WINDOW_SIZE * STRIDE_RATIO)

MINIMUM_SWATH_COVERAGE = config.getfloat(
    "Stage 3",
    "minimum_swath_coverage"
)

CLOUD_DN_THRESHOLD = config.getfloat(
    "Stage 3",
    "cloud_dn_threshold"
)

MAXIMUM_CLOUD_PERCENTAGE = config.getfloat(
    "Stage 3",
    "maximum_cloud_percentage"
)

MODIS_DARK_THRESHOLD = config.getfloat(
    "Stage 3",
    "modis_dark_threshold"
)

MAXIMUM_MODIS_DARK_PERCENTAGE = config.getfloat(
    "Stage 3",
    "maximum_modis_dark_percentage"
)

ACCEPTED_WINDOW_CSV = config.get(
    "Stage 3",
    "accepted_window_csv"
)

REJECTED_WINDOW_CSV = config.get(
    "Stage 3",
    "rejected_window_csv"
)


# ---------------------------------------------
# Stage 5 Parameters
# ---------------------------------------------

STAGE5_OUTPUT_ROOT = config.get(
    "Stage 5",
    "output_root"
)

MAX_ITERATIONS = config.getint(
    "Stage 5",
    "max_iterations"
)

MIN_PEAK_VALUE = config.getfloat(
    "Stage 5",
    "min_peak_value"
)

MIN_VALID_FRACTION = config.getfloat(
    "Stage 5",
    "min_valid_fraction"
)

RELIABILITY_THRESHOLD = config.getfloat(
    "Stage 5",
    "reliability_threshold"
)





# ============================================================
# Input files
# ============================================================

REFERENCE = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/modis_1km.tif"

TARGET = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/hrpt_M03_20250506_0420_33701_geo_b2.tif"

# ============================================================
# PREPROCESSING
# ============================================================

if PREPROCESSING:

    print("\n")
    print("=" * 60)
    print("RUNNING PREPROCESSING")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1
    # Projection
    # ---------------------------------------------------------

    target = match_projection(
        reference_file=REFERENCE,
        target_file=TARGET,
        output_file="2_outputs/01_projection.tif",
        resampling=resampling,
        debug=debug
    )

    # ---------------------------------------------------------
    # STEP 2
    # Overlap
    #
    # Also enforces the final pixel grid (explicit width/
    # height/outputBounds), which supersedes any earlier
    # resolution-matching step - so there is no separate
    # "match resolution" step here, it would be redundant.
    # ---------------------------------------------------------

    reference, target = extract_common_overlap(
        reference_file=REFERENCE,
        target_file=target,
        reference_output="2_outputs/03_modis_overlap.tif",
        target_output="2_outputs/03_avhrr_overlap.tif",
        resampling=resampling,
        debug=debug
    )

    # ---------------------------------------------------------
    # STEP 3
    # Band Selection
    # ---------------------------------------------------------

    reference = select_band(
        reference,
        "2_outputs/04_modis_band.tif",
        reference_band,
        debug
    )

    target = select_band(
        target,
        "2_outputs/04_avhrr_band.tif",
        target_band,
        debug
    )

    # ---------------------------------------------------------
    # STEP 4
    # Float32
    # ---------------------------------------------------------

    reference = convert_to_float32(
        reference,
        "2_outputs/05_modis_float32.tif",
        debug
    )

    target = convert_to_float32(
        target,
        "2_outputs/05_avhrr_float32.tif",
        debug
    )

    # ---------------------------------------------------------
    # STEP 5
    # Verification
    # ---------------------------------------------------------

    verify_geotransform(
        reference,
        target,
        tolerance=tolerance,
        debug=debug
    )

    # ---------------------------------------------------------
    # STEP 6
    # Valid Mask
    # ---------------------------------------------------------

    mask = create_valid_mask(
        input_file=target,
        output_file="2_outputs/07_avhrr_mask.tif",
        threshold=threshold,
        kernel_size=kernel_size,
        debug=debug
    )

    quality_check(
        reference_file=reference,
        target_file=target,
        mask_file=mask,
        tolerance=tolerance,
        debug=debug
    )

# ============================================================
# Finished
# ============================================================

    print("\n")
    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Reference Image : {reference}")
    print(f"Target Image    : {target}")
    print(f"Binary Mask     : {mask}")
    print("=" * 60)


else:

    print("\n")
    print("=" * 60)
    print("PREPROCESSING DISABLED")
    print("=" * 60)

    reference = "2_outputs/05_modis_float32.tif"

    target = "2_outputs/05_avhrr_float32.tif"

    mask = "2_outputs/07_avhrr_mask.tif"


# ==========================================================
# Stage 3 & Stage 4
# Generate and Filter Window Pairs
# ==========================================================

# Read images
avhrr_image, modis_image = read_images(
    target,
    reference,
    debug=debug
)

# Generate sliding windows
windows = generate_windows(
    image_shape=avhrr_image.shape,
    window_size=WINDOW_SIZE,
    stride=STRIDE
)

# Build AVHRR-MODIS window pairs
window_pairs = build_window_pairs(
    windows,
    avhrr_image,
    modis_image,
    debug=debug
)

# Read binary mask
mask = load_mask(
    mask,
    debug=debug
)

# Filter windows
accepted_pairs, rejected_pairs, statistics = filter_window_pairs(
    window_pairs=window_pairs,
    valid_mask=mask,
    minimum_swath_coverage=MINIMUM_SWATH_COVERAGE,
    maximum_cloud_percentage=MAXIMUM_CLOUD_PERCENTAGE,
    cloud_dn_threshold=CLOUD_DN_THRESHOLD,
    modis_dark_threshold=MODIS_DARK_THRESHOLD,
    maximum_modis_dark_percentage=MAXIMUM_MODIS_DARK_PERCENTAGE,
    cloud_detection=CLOUD_DETECTION,
    debug=debug
)

# Save accepted windows
save_accepted_pairs_csv(
    accepted_pairs,
    ACCEPTED_WINDOW_CSV,
    debug=debug
)

# Save rejected windows
save_rejected_pairs_csv(
    rejected_pairs,
    REJECTED_WINDOW_CSV,
    debug=debug
)


print("WINDOW SELECTION COMPLETED SUCCESSFULLY")

# ==========================================================
# Stage 5
# Local Registration (Steps 16-21)
# ==========================================================

if LOCAL_REGISTRATION:

    print("\n")
    print("=" * 60)
    print("RUNNING STAGE 5 - LOCAL REGISTRATION")
    print("=" * 60)

    peak_dir = os.path.join(STAGE5_OUTPUT_ROOT, "stage5_peak_detection")
    validation_dir = os.path.join(STAGE5_OUTPUT_ROOT, "stage5_shift_validation")
    subpixel_dir = os.path.join(STAGE5_OUTPUT_ROOT, "stage5_subpixel_estimation")
    reliability_dir = os.path.join(STAGE5_OUTPUT_ROOT, "stage5_reliability")
    mssim_dir = os.path.join(STAGE5_OUTPUT_ROOT, "stage5_mssim")

    # STEP 16 - Cross-Power Spectrum
    phase = PhaseCorrelation(
        avhrr_path=target,
        modis_path=reference,
        csv_path=ACCEPTED_WINDOW_CSV,
        output_dir=STAGE5_OUTPUT_ROOT
    )
    phase_outputs = phase.run()

    # STEP 17 - Integer Peak Detection
    peaks = PeakDetector(
        summary_csv=phase_outputs["summary_csv"],
        correlation_directory=phase_outputs["correlation_surface_directory"],
        output_directory=peak_dir
    )
    peak_csv = peaks.execute()

    # STEP 18 - Integer Shift Validation
    validator = ShiftValidator(
        avhrr_path=target,
        modis_path=reference,
        peak_csv=peak_csv,
        output_directory=validation_dir,
        max_iterations=MAX_ITERATIONS,
        min_peak_value=MIN_PEAK_VALUE,
        min_valid_fraction=MIN_VALID_FRACTION
    )
    validation_outputs = validator.run()

    # STEP 19 - Subpixel Bootstrap
    subpixel = SubpixelEstimator(
        avhrr_path=target,
        modis_path=reference,
        validated_csv=validation_outputs["validated_shift_csv"],
        output_directory=subpixel_dir
    )
    subpixel_outputs = subpixel.run()

    # STEP 20 - Reliability
    reliability = ReliabilityEstimator(
        subpixel_csv=subpixel_outputs["subpixel_csv"],
        surface_directory=subpixel_outputs["final_surface_directory"],
        output_directory=reliability_dir,
        reliability_threshold=RELIABILITY_THRESHOLD
    )
    reliability_outputs = reliability.run()

    # STEP 21 - MSSIM Before vs After
    mssim = MSSIMEvaluator(
        avhrr_path=target,
        modis_path=reference,
        subpixel_csv=subpixel_outputs["subpixel_csv"],
        output_directory=mssim_dir
    )
    mssim_outputs = mssim.run()

    print("\n")
    print("=" * 60)
    print("STAGE 5 COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Reliability CSV : {reliability_outputs['reliability_csv']}")
    print(f"MSSIM CSV       : {mssim_outputs['mssim_csv']}")
    print("=" * 60)

else:

    print("\n")
    print("=" * 60)
    print("STAGE 5 (LOCAL REGISTRATION) DISABLED")
    print("=" * 60)