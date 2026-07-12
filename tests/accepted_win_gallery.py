import cv2
import math
import pandas as pd
import numpy as np
from osgeo import gdal

# =====================================================
# INPUTS
# =====================================================

AVHRR_FILE = "/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_float32.tif"
MODIS_FILE = "/home/bhaskar/Documents/ImageReg/2_outputs/05_modis_float32.tif"
CSV_FILE = "/home/bhaskar/Documents/ImageReg/34_outputs/accepted_window_pairs.csv"

THUMB_SIZE = 150
PAIR_WIDTH = THUMB_SIZE * 2
COLS = 5
ROWS = 4

WINDOWS_PER_PAGE = COLS * ROWS

# =====================================================
# READ AVHRR
# =====================================================

avhrr_ds = gdal.Open(AVHRR_FILE)

if avhrr_ds is None:
    raise RuntimeError("Cannot open AVHRR image.")

avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

avhrr = cv2.normalize(
    avhrr,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# =====================================================
# READ MODIS
# =====================================================

modis_ds = gdal.Open(MODIS_FILE)

if modis_ds is None:
    raise RuntimeError("Cannot open MODIS image.")

modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

modis = cv2.normalize(
    modis,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# =====================================================
# READ CSV
# =====================================================

df = pd.read_csv(CSV_FILE)

# =====================================================
# BUILD WINDOW LIST
# =====================================================

windows = []

for _, row in df.iterrows():

    r0 = int(row.row_start)
    r1 = int(row.row_end)

    c0 = int(row.col_start)
    c1 = int(row.col_end)

    avhrr_crop = avhrr[r0:r1, c0:c1]

    modis_crop = modis[r0:r1, c0:c1]

    windows.append({

        "id": int(row.window_id),

        "avhrr": avhrr_crop,

        "modis": modis_crop,

        "cloud": float(row.cloud_percentage),

        "swath": float(row.swath_coverage)

    })

# =====================================================
# PAGE DRAWER
# =====================================================

def build_page(page):

    canvas = np.full(

        (
            ROWS * (THUMB_SIZE + 55) + 40,
            COLS * PAIR_WIDTH
        ),

        35,

        dtype=np.uint8
    )

    start = page * WINDOWS_PER_PAGE
    end = min(start + WINDOWS_PER_PAGE, len(windows))

    index = start

    for r in range(ROWS):

        for c in range(COLS):

            if index >= end:
                break

            avhrr_thumb = cv2.resize(
                windows[index]["avhrr"],
                (THUMB_SIZE, THUMB_SIZE)
            )

            modis_thumb = cv2.resize(
                windows[index]["modis"],
                (THUMB_SIZE, THUMB_SIZE)
            )

            pair = np.hstack((avhrr_thumb, modis_thumb))

            y = 40 + r * (THUMB_SIZE + 55)
            x = c * PAIR_WIDTH

            canvas[
                y:y+THUMB_SIZE,
                x:x+PAIR_WIDTH
            ] = pair

            cv2.putText(
                canvas,
                "AVHRR",
                (x+20, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                255,
                1
            )

            cv2.putText(
                canvas,
                "MODIS",
                (x+THUMB_SIZE+20, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                255,
                1
            )

            info = (
                f"ID:{windows[index]['id']}  "
                f"C:{windows[index]['cloud']*100:.1f}%  "
                f"S:{windows[index]['swath']*100:.1f}%"
            )

            cv2.putText(
                canvas,
                info,
                (x+5, y+THUMB_SIZE+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                255,
                1
            )

            index += 1

    return canvas

# =====================================================
# VIEWER
# =====================================================

page = 0

total_pages = math.ceil(
    len(windows) / WINDOWS_PER_PAGE
)

cv2.namedWindow("Accepted Windows Gallery", cv2.WINDOW_NORMAL)

while True:

    page_img = build_page(page)

    cv2.putText(
        page_img,
        f"Page {page+1}/{total_pages}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2
    )

    cv2.putText(
        page_img,
        "N : Next   P : Previous   ESC : Exit",
        (250, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        255,
        2
    )

    cv2.imshow(
        "Accepted Windows Gallery",
        page_img
    )

    key = cv2.waitKeyEx(0)

    print("Key:", key)

    if key == 27:
        break

    elif key in (ord('n'), ord('N')):
        page = min(page + 1, total_pages - 1)

    elif key in (ord('p'), ord('P')):
        page = max(page - 1, 0)

    # Right Arrow
    elif key == 2555904:
        page = min(page + 1, total_pages - 1)

    # Left Arrow
    elif key == 2424832:
        page = max(page - 1, 0)

cv2.destroyAllWindows()