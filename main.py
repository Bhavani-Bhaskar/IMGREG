"""
main.py
-------

Preprocessing pipeline for MODIS–AVHRR image registration.
"""

from configparser import ConfigParser

from preprocessing.projection import match_projection
from preprocessing.spatial_resolution import match_spatial_resolution
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

projection = config["Projection"]["target_projection"]

pixel_width = config.getfloat(
    "Spatial Resolution",
    "pixel_width"
)

pixel_height = config.getfloat(
    "Spatial Resolution",
    "pixel_height"
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

STRIDE = config.getint(
    "Stage 3",
    "stride"
)

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

ACCEPTED_WINDOW_CSV = config.get(
    "Stage 3",
    "accepted_window_csv"
)

REJECTED_WINDOW_CSV = config.get(
    "Stage 3",
    "rejected_window_csv"
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
    # Spatial Resolution
    # ---------------------------------------------------------

    target = match_spatial_resolution(
        input_file=target,
        output_file="2_outputs/02_resolution.tif",
        x_resolution=pixel_width,
        y_resolution=pixel_height,
        target_srs=projection,
        resampling=resampling,
        debug=debug
    )

    # ---------------------------------------------------------
    # STEP 3
    # Overlap
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
    # STEP 4
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
    # STEP 5
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
    # STEP 6
    # Verification
    # ---------------------------------------------------------

    verify_geotransform(
        reference,
        target,
        tolerance=tolerance,
        debug=debug
    )

    # ---------------------------------------------------------
    # STEP 7
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