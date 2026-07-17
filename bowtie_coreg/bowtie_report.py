"""
bowtie_report.py
----------------

Report deliverable: prove the AUTOMATICALLY-extracted tie points are valid, by
showing the AVHRR imagery shifted by each tie point's vector overlapping the
MODIS reference.

Validity is judged PER TIE POINT, independently of the global warp: for a tie
point with AVHRR source location `ta` and matched MODIS location `tb`
(off = tb - ta), the AVHRR feature currently at `ta` should move to `tb`. So:
    ncc_before = NCC( MODIS@tb , AVHRR@tb )   # uncorrected: same location
    ncc_after  = NCC( MODIS@tb , AVHRR@ta )   # corrected: AVHRR feature moved to tb
Both windows are pulled from real image locations (no NaN-empty-window trap).
A tie point is "valid" if ncc_after > ncc_before (its shift genuinely aligns
the AVHRR coastline onto MODIS).

Outputs (bowtie_output/report/):
  - tiepoint_vectors.png       bow-tie displacement field (arrows converging to nadir)
  - overlay_before_after.png   whole-scene red/cyan AVHRR-on-MODIS, before vs after
  - tiepoint_gallery.png       top points: MODIS | AVHRR before | AVHRR after
  - validity_summary.txt       per-point NCC before/after + match score

Run (after bowtie_pipeline import works):
    conda run -n geo python bowtie_report.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import bowtie_pipeline as bp

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
MANUAL_DIR = os.path.join(_BASE, "inputs")   # inputs built by preprocessing.py
REPORT_DIR = os.path.join(_BASE, "output", "report")
NCC_WIN = 90          # half-size of the validation window (px)
GALLERY_N = 12        # tie points shown in the gallery
GALLERY_HALF = 130    # half-size of gallery crops (px)


def ncc(a, b):
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < a.size * 0.3:
        return np.nan
    av = a[v] - a[v].mean(); bv = b[v] - b[v].mean()
    d = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av * bv).sum() / d) if d else np.nan


def crop(arr, cy, cx, half):
    H, W = arr.shape
    r0, r1 = max(cy - half, 0), min(cy + half, H)
    c0, c1 = max(cx - half, 0), min(cx + half, W)
    return arr[r0:r1, c0:c1]


def norm(a):
    a = a.astype(np.float32)
    v = np.isfinite(a) & (a != 0)
    if v.sum() < 10:
        return np.zeros_like(a)
    lo, hi = np.nanpercentile(a[v], 2), np.nanpercentile(a[v], 98)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def per_point_validation(ta, off, a_arr, m_arr):
    """NCC before/after for each tie point (see module docstring)."""
    rows = []
    for (sx, sy), (dx, dy) in zip(ta, off):
        ax, ay = int(round(sx)), int(round(sy))          # AVHRR source
        bx, by = int(round(sx + dx)), int(round(sy + dy))  # MODIS match (dest)
        m_ref = crop(m_arr, by, bx, NCC_WIN)
        a_before = crop(a_arr, by, bx, NCC_WIN)           # AVHRR at dest (uncorrected)
        a_after = crop(a_arr, ay, ax, NCC_WIN)            # AVHRR at source (corrected)
        h = min(m_ref.shape[0], a_before.shape[0], a_after.shape[0])
        w = min(m_ref.shape[1], a_before.shape[1], a_after.shape[1])
        if h < NCC_WIN or w < NCC_WIN:
            rows.append((np.nan, np.nan))
            continue
        nb = ncc(a_before[:h, :w], m_ref[:h, :w])
        na = ncc(a_after[:h, :w], m_ref[:h, :w])
        rows.append((nb, na))
    return np.array(rows)


def fig_vectors(ta, off, base_img, shape, path):
    H, W = shape
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    ax.imshow(norm(base_img), cmap="gray", origin="upper")
    # colour by dx sign: red = pull right (left of nadir), blue = pull left
    colors = np.where(off[:, 0] >= 0, "red", "deepskyblue")
    ax.quiver(ta[:, 0], ta[:, 1], off[:, 0], -off[:, 1], color=colors,
              angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.set_title(f"Extracted tie-point displacement field ({len(ta)} points)\n"
                 f"red = pull East, blue = pull West  (arrows converge toward nadir = bow-tie)")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def redcyan(avhrr, modis):
    """AVHRR in red, MODIS in cyan; aligned features -> gray, misaligned -> colour fringes."""
    a = norm(avhrr); m = norm(modis)
    return np.dstack([a, m, m])


def fig_overlay(a_before, a_after, m_arr, path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 13))
    axes[0].imshow(redcyan(a_before, m_arr))
    axes[0].set_title("BEFORE: original AVHRR (red) on MODIS (cyan)\ncolour fringes = misalignment")
    axes[1].imshow(redcyan(a_after, m_arr))
    axes[1].set_title("AFTER: tie-point-corrected AVHRR (red) on MODIS (cyan)\ngray = aligned")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def fig_gallery(ta, off, ncc_ba, a_arr, m_arr, path):
    shiftmag = np.hypot(off[:, 0], off[:, 1])
    valid = ncc_ba[:, 1] > ncc_ba[:, 0]
    order = np.argsort(-shiftmag)
    order = [i for i in order if valid[i]][:GALLERY_N]

    n = len(order)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for row, i in enumerate(order):
        sx, sy = ta[i]; dx, dy = off[i]
        ax_, ay_ = int(round(sx)), int(round(sy))
        bx, by = int(round(sx + dx)), int(round(sy + dy))
        m_ref = norm(crop(m_arr, by, bx, GALLERY_HALF))
        a_bef = norm(crop(a_arr, by, bx, GALLERY_HALF))
        a_aft = norm(crop(a_arr, ay_, ax_, GALLERY_HALF))
        for col, (img, title) in enumerate([
                (m_ref, "MODIS (reference)"),
                (a_bef, f"AVHRR before  NCC={ncc_ba[i,0]:.2f}"),
                (a_aft, f"AVHRR shifted  NCC={ncc_ba[i,1]:.2f}")]):
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=10)
            else:
                axes[row, col].set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(f"shift {shiftmag[i]:.0f}px", fontsize=9)
    plt.suptitle("Per-tie-point validity: AVHRR shifted onto MODIS "
                 "(col1 coastline should match col3)", y=1.001)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def main():
    global MANUAL_DIR, REPORT_DIR
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default=MANUAL_DIR)
    p.add_argument("--output", default=os.path.dirname(REPORT_DIR))
    args = p.parse_args()
    MANUAL_DIR = args.inputs
    REPORT_DIR = os.path.join(args.output, "report")
    os.makedirs(REPORT_DIR, exist_ok=True)

    arrays = {n: np.load(os.path.join(MANUAL_DIR, n + ".npy")) for n in
              ["a_arr", "s_arr", "m_arr", "s_land", "m_land", "cloud_mask"]}
    a_arr, m_arr = arrays["a_arr"], arrays["m_arr"]
    H, W = a_arr.shape

    print("Running full automatic tie-point extraction "
          "(match -> west-infill -> grow -> curate)...")
    ta_all, off_all, sc_all = bp.extract_tie_points(arrays, verbose=True)
    # sc<0 marks Gangetic zero-shift anchors: field-only, not real matches.
    real = sc_all >= 0
    ta, off, sc = ta_all[real], off_all[real], sc_all[real]
    print(f"Extracted {real.sum()} matched tie points "
          f"(+{(~real).sum()} Gangetic protection anchors)")

    print("\nPer-tie-point NCC validation (AVHRR shifted onto MODIS)...")
    ncc_ba = per_point_validation(ta, off, a_arr, m_arr)
    valid = ncc_ba[:, 1] > ncc_ba[:, 0]
    finite = np.isfinite(ncc_ba).all(axis=1)

    with open(os.path.join(REPORT_DIR, "validity_summary.txt"), "w") as f:
        f.write(f"Automatically extracted tie points: {len(ta)}\n")
        f.write(f"With finite NCC on both sides      : {finite.sum()}\n")
        f.write(f"VALID (NCC after > before)         : {valid.sum()} "
                f"({100*valid.sum()/len(ta):.0f}%)\n")
        f.write(f"Mean NCC before -> after           : "
                f"{np.nanmean(ncc_ba[:,0]):.3f} -> {np.nanmean(ncc_ba[:,1]):.3f}\n")
        f.write(f"Mean match score (coastline NCC)   : {sc.mean():.3f}\n")
        f.write(f"Shift magnitude px: min {np.hypot(off[:,0],off[:,1]).min():.0f} "
                f"mean {np.hypot(off[:,0],off[:,1]).mean():.0f} "
                f"max {np.hypot(off[:,0],off[:,1]).max():.0f}\n")
        f.write("\nrow  col   dx    dy   score  ncc_before ncc_after valid\n")
        for i in range(len(ta)):
            f.write(f"{ta[i,1]:5.0f} {ta[i,0]:5.0f} {off[i,0]:+5.0f} {off[i,1]:+5.0f} "
                    f"{sc[i]:.3f}   {ncc_ba[i,0]:+.3f}    {ncc_ba[i,1]:+.3f}   "
                    f"{'Y' if valid[i] else 'n'}\n")
    print(f"VALID tie points: {valid.sum()}/{len(ta)} "
          f"(mean NCC {np.nanmean(ncc_ba[:,0]):.3f} -> {np.nanmean(ncc_ba[:,1]):.3f})")

    # figures
    fig_vectors(ta, off, m_arr, (H, W), os.path.join(REPORT_DIR, "tiepoint_vectors.png"))

    print("\nBuilding TPS warp for whole-scene overlay...")
    dxf, dyf = bp.build_shift_field(ta_all, off_all, arrays)  # TPS + data-driven trust mask
    a_after = bp.warp_with_field(a_arr, dxf, dyf)
    fig_overlay(a_arr, a_after, m_arr, os.path.join(REPORT_DIR, "overlay_before_after.png"))

    fig_gallery(ta, off, ncc_ba, a_arr, m_arr, os.path.join(REPORT_DIR, "tiepoint_gallery.png"))

    # self-referential validation: cloud-masked NCC (orig vs corrected) on a grid,
    # overall and by 3x3 spatial tile (generalises to any granule / area)
    scene_ncc_report(a_arr, a_after, m_arr, ~arrays["cloud_mask"],
                     os.path.join(REPORT_DIR, "ncc_validation.txt"))

    print(f"\nReport written to {REPORT_DIR}/")


def scene_ncc_report(orig, corr, modis, clear, path, win=110, step=110):
    H, W = orig.shape
    rows = []
    for r in range(0, H - win, step):
        for c in range(0, W - win, step):
            cl = clear[r:r + win, c:c + win]
            mref = modis[r:r + win, c:c + win]
            b = ncc(orig[r:r + win, c:c + win], mref) if cl.mean() > 0.5 else np.nan
            # cloud-masked
            b = _ncc_masked(orig[r:r + win, c:c + win], mref, cl)
            a = _ncc_masked(corr[r:r + win, c:c + win], mref, cl)
            if np.isfinite(b) and np.isfinite(a):
                rows.append((r + win // 2, c + win // 2, b, a))
    rows = np.array(rows)
    lines = ["SELF-REFERENTIAL VALIDATION (cloud-masked NCC vs MODIS, original -> corrected)\n"]
    if len(rows):
        b, a = rows[:, 2], rows[:, 3]
        lines.append(f"checkpoint windows: {len(rows)}\n")
        lines.append(f"overall mean NCC: {b.mean():.3f} -> {a.mean():.3f}  "
                     f"(delta {a.mean()-b.mean():+.3f}, improved {100*(a>b).mean():.0f}%)\n\n")
        lines.append("by 3x3 spatial tile (row-band x col-band):\n")
        for ri, (r0, r1) in enumerate([(0, H//3), (H//3, 2*H//3), (2*H//3, H)]):
            for ci, (c0, c1) in enumerate([(0, W//3), (W//3, 2*W//3), (2*W//3, W)]):
                m = (rows[:, 0] >= r0) & (rows[:, 0] < r1) & (rows[:, 1] >= c0) & (rows[:, 1] < c1)
                if m.sum():
                    lines.append(f"  tile[{ri},{ci}] n={m.sum():3d}  "
                                 f"{rows[m,2].mean():.3f} -> {rows[m,3].mean():.3f}  "
                                 f"({rows[m,3].mean()-rows[m,2].mean():+.3f})\n")
    open(path, "w").writelines(lines)
    if len(rows):
        print(f"Scene NCC: {rows[:,2].mean():.3f} -> {rows[:,3].mean():.3f} "
              f"({100*(rows[:,3]>rows[:,2]).mean():.0f}% windows improved) -> {path}")


def _ncc_masked(a, b, mask):
    v = np.isfinite(a) & np.isfinite(b) & mask & (a != 0) & (b != 0)
    if v.sum() < a.size * 0.15:
        return np.nan
    av = a[v] - a[v].mean(); bv = b[v] - b[v].mean()
    d = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av * bv).sum() / d) if d else np.nan


if __name__ == "__main__":
    main()
