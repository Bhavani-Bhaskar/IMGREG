"""Zoomed Karnataka/Kerala (SW coast) before/after alignment check."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from osgeo import gdal

gdal.UseExceptions()
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
d = os.path.join(_BASE, "inputs") + "/"   # inputs built by preprocessing.py
OUT = os.path.join(_BASE, "output", "report")

# SW coast crop (common grid 0.01deg, origin 64.0889,42.0742):
# Goa~row2677 -> Trivandrum~row3360 ; west coast cols ~900-1350
R0, R1, C0, C1 = 2600, 3480, 860, 1380

a = np.load(d + "a_arr.npy").astype(np.float32)
m = np.load(d + "m_arr.npy").astype(np.float32)
cd = gdal.Open(os.path.join(_BASE, "output", "avhrr_bowtie_corrected.tif"))
c = cd.GetRasterBand(1).ReadAsArray().astype(np.float32)
cd = None


def norm(x):
    v = np.isfinite(x) & (x != 0)
    if v.sum() < 10:
        return np.zeros_like(x)
    lo, hi = np.nanpercentile(x[v], 2), np.nanpercentile(x[v], 98)
    o = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    o[~np.isfinite(o)] = 0
    return o


def redcyan(av, mo):
    return np.dstack([av, mo, mo])


ac = norm(a[R0:R1, C0:C1]); mc = norm(m[R0:R1, C0:C1]); cc = norm(c[R0:R1, C0:C1])

fig, ax = plt.subplots(1, 2, figsize=(15, 12))
ax[0].imshow(redcyan(ac, mc)); ax[0].set_title("BEFORE - Karnataka/Kerala W coast\n(red=AVHRR, cyan=MODIS; fringe=shift)")
ax[1].imshow(redcyan(cc, mc)); ax[1].set_title("AFTER - shifted AVHRR\n(gray = aligned)")
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([])
plt.savefig(f"{OUT}/swcoast_redcyan.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}/swcoast_redcyan.png")

# composite: MODIS base with coast; before vs after overlaid
def comp(base, ov):
    ovv = np.isfinite(ov) & (ov > 0)
    o = base.copy(); o[ovv] = 0.5 * ov[ovv] + 0.5 * base[ovv]
    return o

fig, ax = plt.subplots(1, 2, figsize=(15, 12))
ax[0].imshow(comp(mc, ac), cmap="gray"); ax[0].set_title("BEFORE blended")
ax[1].imshow(comp(mc, cc), cmap="gray"); ax[1].set_title("AFTER blended")
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([])
plt.savefig(f"{OUT}/swcoast_blend.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}/swcoast_blend.png")
