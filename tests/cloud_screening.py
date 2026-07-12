
import numpy as np
import cv2
from osgeo import gdal


# =====================================================
# WINDOW GENERATION
# =====================================================

def generate_windows(image_shape,
                     window_size=256,
                     stride=128):

    rows, cols = image_shape

    windows = []

    for r in range(
            0,
            rows - window_size + 1,
            stride):

        for c in range(
                0,
                cols - window_size + 1,
                stride):

            windows.append({

                "row_start": r,
                "row_end": r + window_size,

                "col_start": c,
                "col_end": c + window_size,

                "center_row": r + window_size // 2,
                "center_col": c + window_size // 2
            })

    return windows


# =====================================================
# EXTRACT WINDOW
# =====================================================

def extract_window(image, win):

    return image[
        win["row_start"]:win["row_end"],
        win["col_start"]:win["col_end"]
    ]


# =====================================================
# CLOUD FRACTION (DN Threshold)
# =====================================================

def compute_cloud_fraction(
        window,
        cloud_threshold=350):
    """
    Compute cloud fraction using DN threshold.

    Parameters
    ----------
    window : ndarray

    cloud_threshold : int
        DN threshold

    Returns
    -------
    cloud_fraction
    cloud_mask
    """

    valid_mask = ~np.isnan(window)

    valid_pixels = np.sum(valid_mask)

    if valid_pixels == 0:
        return 0.0, np.zeros_like(window, dtype=bool)

    cloud_mask = (
        (window > cloud_threshold)
        & valid_mask
    )

    cloud_fraction = (
        np.sum(cloud_mask)
        / valid_pixels
    )

    return cloud_fraction, cloud_mask


# =====================================================
# ACCEPT / REJECT
# =====================================================

def get_status(
        cloud_fraction,
        reject_threshold=0.20):

    if cloud_fraction > reject_threshold:
        return "REJECT"
    else:
        return "ACCEPT"


# =====================================================
# NORMALIZE FOR DISPLAY
# =====================================================

def normalize_image(img):

    img = np.nan_to_num(img, nan=0)

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return img.astype(np.uint8)


# =====================================================
# LOAD IMAGE
# =====================================================

print("\nLoading AVHRR...")

input_file = "/data/student_project_imreg/codes/scripts/new/pipeline/2_outputs/05_avhrr_float32.tif"

ds = gdal.Open(input_file)

if ds is None:
    raise RuntimeError(f"Cannot open {input_file}")

band = ds.GetRasterBand(1)

avhrr_image = band.ReadAsArray().astype(np.float32)

print("\nImage Information")
print("-" * 50)
print("Shape:", avhrr_image.shape)
print("Dtype:", avhrr_image.dtype)
print("Total Pixels:", avhrr_image.size)
print("NaN Pixels:", np.isnan(avhrr_image).sum())

if np.isnan(avhrr_image).sum() < avhrr_image.size:
    print("Global Min :", np.nanmin(avhrr_image))
    print("Global Max :", np.nanmax(avhrr_image))
    print("Global Mean:", np.nanmean(avhrr_image))
else:
    raise RuntimeError("Entire image contains NaNs")


# =====================================================
# GENERATE WINDOWS
# =====================================================

windows = generate_windows(
    avhrr_image.shape,
    window_size=256,
    stride=128
)

print("\nTotal Windows =", len(windows))


# =====================================================
# USER INPUT
# =====================================================

window_id = int(
    input(
        f"\nEnter window number (0-{len(windows)-1}): "
    )
)

if window_id < 0 or window_id >= len(windows):
    raise ValueError("Invalid window number")


# =====================================================
# EXTRACT WINDOW
# =====================================================

win = windows[window_id]

window = extract_window(
    avhrr_image,
    win
)

print("\nWindow Information")
print("-" * 50)

nan_count = np.isnan(window).sum()

print("Window Shape :", window.shape)
print("Window Pixels:", window.size)
print("NaNs         :", nan_count)

valid_mask = ~np.isnan(window)

if np.sum(valid_mask) == 0:
    raise RuntimeError("Selected window contains only NaNs")

valid_pixels = window[valid_mask]

print("Valid Pixels :", valid_pixels.size)

print("\nStatistics")
print("Min  :", np.min(valid_pixels))
print("Max  :", np.max(valid_pixels))
print("Mean :", np.mean(valid_pixels))
print("Std  :", np.std(valid_pixels))

print("\nPercentiles")
print("P1   :", np.percentile(valid_pixels, 1))
print("P2   :", np.percentile(valid_pixels, 2))
print("P5   :", np.percentile(valid_pixels, 5))
print("P95  :", np.percentile(valid_pixels, 95))
print("P98  :", np.percentile(valid_pixels, 98))
print("P99  :", np.percentile(valid_pixels, 99))

print("\nDN Threshold Counts")
print("-" * 50)

thresholds = list(range(100, 1001, 50))

for th in thresholds:

    count = np.sum(valid_pixels > th)

    percent = 100 * count / valid_pixels.size

    print(
        f"DN > {th:4d} : "
        f"{count:6d} pixels "
        f"({percent:6.2f} %)"
    )


# =====================================================
# CLOUD SCREENING
# =====================================================

thresholds = list(range(100, 1001, 50))

print("\n")
print("=" * 60)
print("DN THRESHOLD ANALYSIS")
print("=" * 60)

best_mask = None
print("\nThreshold   Cloud%   Status")
print("--------------------------------")
for threshold in thresholds:

    cloud_fraction, cloud_mask = compute_cloud_fraction(
        window,
        cloud_threshold=threshold
    )

    status = get_status(
        cloud_fraction,
        reject_threshold=0.20
    )


    print(
        f"DN Threshold = {threshold:4d} | "
        f"Cloud = {cloud_fraction*100:6.2f}% | "
        f"{status}"
    )

    # Save one mask (change this if desired)
    if threshold == 350:
        best_mask = cloud_mask
        best_fraction = cloud_fraction
        best_threshold = threshold
        best_status = status


# =====================================================
# DISPLAY IMAGES
# =====================================================

# Normalize window for visualization
window_disp = normalize_image(window)

# Resize image
window_disp = cv2.resize(
    window_disp,
    (512, 512),
    interpolation=cv2.INTER_LINEAR
)

# Convert grayscale -> BGR
window_disp = cv2.cvtColor(
    window_disp,
    cv2.COLOR_GRAY2BGR
)

# Resize cloud mask
mask_disp = cv2.resize(
    best_mask.astype(np.uint8),
    (512, 512),
    interpolation=cv2.INTER_NEAREST
).astype(bool)

# Create overlay
cloud_disp = window_disp.copy()

# Paint cloud pixels red
cloud_disp[mask_disp] = (0, 0, 255)


# =====================================================
# LABELS
# =====================================================

cv2.putText(
    window_disp,
    f"Window {window_id}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    window_disp,
    f"Cloud={best_fraction*100:.1f}%",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    window_disp,
    f"Status={best_status}",
    (10, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    cloud_disp,
    f"Cloud Mask (>{best_threshold})",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


# =====================================================
# COMBINE
# =====================================================

combined = np.hstack(
    (
        window_disp,
        cloud_disp
    )
)


# =====================================================
# SAVE OUTPUT
# =====================================================

output_png = f"cloud_screening_window_{window_id}.png"

cv2.imwrite(
    output_png,
    combined
)


# =====================================================
# DISPLAY RESULT
# =====================================================

cv2.imshow("Cloud Screening", combined)

cv2.waitKey(0)

cv2.destroyAllWindows()



print("\nSaved:", output_png)

print("\nDone.")
