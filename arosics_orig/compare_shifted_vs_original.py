from osgeo import gdal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gdal.UseExceptions()

ORIG_PATH    = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2.tif"
SHIFTED_PATH = "/home/bhaskar/Documents/ImageReg/arosics_orig/05_avhrr_reflectance_ch2__shifted_to__modis_1km.tif"
REF_PATH     = "/home/bhaskar/Documents/ImageReg/arosics_orig/modis_1km.tif"

OUT_DIR = "/home/bhaskar/Documents/ImageReg/arosics_orig"

def read_band(path, band=1):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    arr = ds.GetRasterBand(band).ReadAsArray().astype("float64")
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, gt, ds.RasterXSize, ds.RasterYSize

def get_bounds(gt, xsize, ysize):
    x0, y0 = gt[0], gt[3]
    x1 = x0 + xsize * gt[1] + ysize * gt[2]
    y1 = y0 + xsize * gt[4] + ysize * gt[5]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

orig_arr, orig_gt, ow, oh = read_band(ORIG_PATH)
shift_arr, shift_gt, sw, sh = read_band(SHIFTED_PATH)

orig_bounds = get_bounds(orig_gt, ow, oh)
shift_bounds = get_bounds(shift_gt, sw, sh)

print("Original  bounds:", orig_bounds)
print("Shifted   bounds:", shift_bounds)
print("Original  shape: ", orig_arr.shape, " GSD:", orig_gt[1])
print("Shifted   shape: ", shift_arr.shape, " GSD:", shift_gt[1])

# --- 1. Full-scene side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
vmin = np.nanpercentile(orig_arr, 2)
vmax = np.nanpercentile(orig_arr, 98)
axes[0].imshow(orig_arr, cmap="gray", vmin=vmin, vmax=vmax)
axes[0].set_title(f"ORIGINAL AVHRR\nbounds={tuple(round(b,3) for b in orig_bounds)}")
axes[1].imshow(shift_arr, cmap="gray", vmin=vmin, vmax=vmax)
axes[1].set_title(f"SHIFTED AVHRR (corrected)\nbounds={tuple(round(b,3) for b in shift_bounds)}")
plt.suptitle("Full-scene comparison: original vs. shifted AVHRR")
plt.savefig(f"{OUT_DIR}/compare_full_scene3.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/compare_full_scene3.png")

# --- 2. Zoomed patch comparison, centered on the matching window AROSICS used ---
CENTER_LON, CENTER_LAT = 79.62435769648198, 20.420796564693784
HALF_DEG = 1.0

def crop_patch(arr, gt, xsize, ysize, lon0, lat0, half_deg):
    inv_gt = gdal.InvGeoTransform(gt)
    col0, row0 = gdal.ApplyGeoTransform(inv_gt, lon0 - half_deg, lat0 + half_deg)
    col1, row1 = gdal.ApplyGeoTransform(inv_gt, lon0 + half_deg, lat0 - half_deg)
    col0, col1 = sorted([int(round(col0)), int(round(col1))])
    row0, row1 = sorted([int(round(row0)), int(round(row1))])
    col0, row0 = max(col0, 0), max(row0, 0)
    col1, row1 = min(col1, xsize), min(row1, ysize)
    return arr[row0:row1, col0:col1]

ref_arr, ref_gt, rw, rh = read_band(REF_PATH)
ref_patch   = crop_patch(ref_arr,   ref_gt,   rw, rh, CENTER_LON, CENTER_LAT, HALF_DEG)
orig_patch  = crop_patch(orig_arr,  orig_gt,  ow, oh, CENTER_LON, CENTER_LAT, HALF_DEG)
shift_patch = crop_patch(shift_arr, shift_gt, sw, sh, CENTER_LON, CENTER_LAT, HALF_DEG)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
axes[0].imshow(ref_patch, cmap="gray")
axes[0].set_title("REFERENCE (MODIS)")
axes[1].imshow(orig_patch, cmap="gray")
axes[1].set_title("TARGET before shift (original AVHRR)")
axes[2].imshow(shift_patch, cmap="gray")
axes[2].set_title("TARGET after shift (corrected AVHRR)")
plt.suptitle(f"Zoomed patch @ ({CENTER_LON:.2f}E, {CENTER_LAT:.2f}N), ±{HALF_DEG} deg")
plt.savefig(f"{OUT_DIR}/compare_zoomed_patch3.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/compare_zoomed_patch3.png")

# --- 3. Checkerboard / blink overlay of reference vs. AVHRR before and after ---
def checkerboard(a, b, tile=20):
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    out = a.copy()
    ys, xs = np.indices((h, w))
    mask = ((ys // tile) + (xs // tile)) % 2 == 0
    out[mask] = b[mask]
    return out

def normalize(arr):
    lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

ref_n = normalize(ref_patch)
orig_n = normalize(orig_patch)
shift_n = normalize(shift_patch)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(checkerboard(ref_n, orig_n), cmap="gray")
axes[0].set_title("Checkerboard: REF vs ORIGINAL AVHRR\n(misalignment shows as broken edges at tile borders)")
axes[1].imshow(checkerboard(ref_n, shift_n), cmap="gray")
axes[1].set_title("Checkerboard: REF vs SHIFTED AVHRR\n(should look smoother if correction worked)")
plt.savefig(f"{OUT_DIR}/compare_checkerboard3.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/compare_checkerboard3.png")