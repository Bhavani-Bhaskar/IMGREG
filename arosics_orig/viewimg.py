
from osgeo import gdal
import cv2
import numpy as np

def read_tiff(path):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    img = band.ReadAsArray().astype(np.float32)
    return img

def normalize_for_display(img):
    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    return img.astype(np.uint8)

def autoscale(img, max_width=1200, max_height=800):
    h, w = img.shape

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

# -------------------------
# File paths
# -------------------------

#avhrr_file = "/data/student_project_imreg/metop_work/hrpt_M03_20250506_0420_33701_geo_b2.tif"
#modis_file = "/data/student_project_imreg/metop_work/modis_1km.tif"

#avhrr_file="/data/student_project_imreg/codes/scripts/combined_bad_mask.tif"
#modis_file="/data/student_project_imreg/metop_work/hrpt_M03_20250506_0420_33701_geo_b2.tif"

avhrr_file="/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"
modis_file="/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km_oceanmask.tif"

# -------------------------
# Read images
# -------------------------

avhrr = read_tiff(avhrr_file)
modis = read_tiff(modis_file)

print("AVHRR Shape :", avhrr.shape)
print("MODIS Shape :", modis.shape)

# -------------------------
# Normalize for viewing
# -------------------------

avhrr_disp = normalize_for_display(avhrr)
modis_disp = normalize_for_display(modis)

# -------------------------
# Auto-scale
# -------------------------

avhrr_disp = autoscale(avhrr_disp)
modis_disp = autoscale(modis_disp)

# -------------------------
# Display
# -------------------------

cv2.namedWindow("AVHRR", cv2.WINDOW_NORMAL)
cv2.namedWindow("MODIS", cv2.WINDOW_NORMAL)

cv2.imshow("AVHRR", avhrr_disp)
cv2.imshow("MODIS", modis_disp)

print("Press any key to close...")

cv2.waitKey(0)
cv2.destroyAllWindows()