"""
bowtie_pipeline.py
------------------

Automatic AVHRR->MODIS geolocation correction.

The geolocation error is a BOW-TIE / panoramic across-track distortion - mean
dx ~= 0, with dx flipping sign across the nadir column (~1550), magnitude
growing toward the swath edges, and the swath shrinking at top & bottom. There
is NO global shift. So the pipeline matches binary land/water COASTLINE masks
locally (sensor/time invariant), with NO global pre-shift, plus terrain-
structure matching inland, and fits a Thin Plate Spline.

Inputs are the common-grid arrays in `inputs/` produced by preprocessing.py
directly from the raw granule bands (a_arr AVHRR-visible, s_arr SWIR, b4_arr
thermal, m_arr MODIS, s_land/m_land land masks, cloud_mask). The pipeline
depends only on those - no external/manual files. curated_ta/off.npy in
inputs/, if present, are used only for optional accuracy validation.

Run with the `geo` conda env:
    conda run -n geo python bowtie_pipeline.py
"""

import os
import numpy as np
import cv2
from osgeo import gdal
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import sobel, gaussian_filter

gdal.UseExceptions()

# Paths are resolved relative to this script so it runs from any working dir.
_BASE = os.path.dirname(os.path.abspath(__file__))   # bowtie_coreg/
_ROOT = os.path.dirname(_BASE)                        # project root
INPUTS_DIR = os.path.join(_BASE, "inputs")           # built by preprocessing.py from raw bands
OUTPUT_DIR = os.path.join(_BASE, "output")           # all generated outputs
# optional ground truth (hand-picked tie points) for validation only; if absent
# the pipeline still runs fully from raw bands and just skips validation.
GT_TA = os.path.join(INPUTS_DIR, "curated_ta.npy")
GT_OFF = os.path.join(INPUTS_DIR, "curated_off.npy")

# Stage 4 dense tile matching (based on the user's proven parameters, but a
# denser grid + a couple of extra tile sizes: coastline tiles are sparse, and
# the manual set had ~60 points across the scene, so we oversample candidates
# and let the automatic curation reject the extras).
TILE = 200
TILE_SIZES = (120, 160, 200, 240)  # multi-scale (incl. smaller 120px for more tie points)
STEP = 50                     # was 100: denser candidate grid
SEARCH = 170                  # max detectable shift +/-170px (true bow-tie max ~150px)
SCORE_THRESH = 0.55
FRAC_BOUNDS = (0.03, 0.97)
VALID_FRAC_MIN = 0.9          # AVHRR template tile must be this valid
MATCH_VALID_MIN = 0.85        # MODIS validity checked at the MATCHED location only
                              # (not over the whole search window - MODIS is only
                              # ~70% valid scene-wide, and gating the full search
                              # region blocked the entire east-of-nadir side)
CLOUD_GATE_MIN = 0.40         # was 0.80: too strict - it blocked all of Gujarat
                              # (only 45% clear). The distinctiveness + neighbour
                              # + LOO curation now guards against cloud false-matches.
PEAK_RATIO = 0.85             # reject if 2nd-best peak > this * best (ambiguous)
PEAK_SUPPRESS = 30            # px radius suppressed around the best peak

# Seed-and-grow: the confident seed matches cluster on the complex east coast;
# the west coast (a nearly straight N-S line -> along-shore matching ambiguity)
# and Gujarat (heavy cloud) get almost none. Fit a smooth bow-tie model from
# the seeds, PREDICT the shift elsewhere, and re-match with a small search
# window locked around the prediction. The tight window resolves the straight-
# coast ambiguity and lets the cloud gate relax safely.
GROW = True
GROW_ITERS = 3
GROW_STEP = 70
GROW_TILE = 200
GROW_PRED_TOL = 30            # search radius around the predicted shift (px)
GROW_SCORE_THRESH = 0.50      # can be lower: the prediction already constrains position
GROW_CLOUD_MIN = 0.45         # relaxed (prediction guards against cloud false-matches)
GROW_MODEL_ORDER = 2         # polynomial order for the bow-tie prediction model
GROW_ACCEPT_PX = 22          # accepted match must land within this of the prediction

# West-coast / Gujarat infill: these regions have weaker, cloudier, sometimes
# straighter coastlines, so the strict distinctiveness/score filters reject
# their (real but weak) matches. Run a dedicated permissive pass over the west
# columns and rely on neighbour-corroboration to keep only the mutually
# consistent ones (the correct far-west anchors agree with each other; the
# scattered wrong matches don't).
# Permissive infill: a relaxed-cloud, permissive-distinctiveness coastline pass
# over the WHOLE scene (data-driven, not restricted to any column). Neighbour
# corroboration downstream keeps only mutually-consistent matches, so the extra
# candidates are safe. (Was 'west infill' - generalised for any granule.)
WEST_INFILL = True
WEST_COL_MAX = None          # None = whole scene (no hardcoded column limit)
WEST_CLOUD_MIN = 0.35
WEST_SCORE_THRESH = 0.52
WEST_PEAK_RATIO = 0.93       # permissive distinctiveness for the infill pass

# Straight-coast (Karnataka/Kerala) matching. That SW coast is a near-straight
# N-S line: 2D matching is ambiguous ALONG the coast (aperture problem) and
# produced no surviving points. Fix: measure ONLY the reliable across-shore
# (dx, east-west) component with a wide-x / narrow-y search locked around the
# bow-tie model's predicted position, and let dy stay at the (small, smooth)
# predicted value. Gives a clean dx gradient down the coast.
# Straight-coast dx-match was hardcoded to the 33701 Karnataka/Kerala coast, so
# it is DISABLED in the general pipeline (the model-constrained 'grow' pass
# provides comparable coverage without a hardcoded region).
STRAIGHT_COAST = False
SC_REGION = (860, 1400, 2400, 3380)  # col_min, col_max, row_min, row_max (unused)
SC_TILE = 200
SC_STEP = 60
SC_SEARCH_X = 75             # wide search across-shore (reliable dx)
SC_SEARCH_Y = 8              # narrow search along-shore (dy from prediction)
SC_SCORE_THRESH = 0.50
SC_CLOUD_MIN = 0.50

# Interior / North (Himalaya + Gangetic plain) matching. Coastlines only exist
# at the shore, so the north (which sits north of every coastline point) is
# only reached by extrapolation and stays uncorrected. Match terrain STRUCTURE
# (gradient magnitude - ridges, snow lines, rivers) there, cross-sensor. Uses a
# distinctiveness gate + the real-image NCC gate so only genuine texture
# matches survive. (The interior BETWEEN the coasts is already corrected by the
# smooth field, so this mainly helps the north.)
INTERIOR_MATCH = True
IM_TILE = 200
IM_STEP = 90
IM_SEARCH = 90
IM_SCORE_THRESH = 0.40
IM_PEAK_RATIO = 0.85         # distinctiveness for structure matches
IM_STD_MIN = 4.0            # min template texture (structure std)
IM_CLOUD_MIN = 0.85
IM_GAUSS = 1.5

# DATA-DRIVEN AUTO-PROTECTION (replaces all hardcoded region boxes). Any place
# the correction can't be trusted is pinned to its original geolocation with a
# zero-shift anchor. A coarse-grid cell is "protected" if it is:
#   - CLOUDY      : local clear fraction < AP_CLEAR_MIN  (can't match through cloud)
#   - SWATH EDGE  : local valid fraction < AP_VALID_MIN  (too little AVHRR data)
#   - EXTRAPOLATED: farther than AP_FAR_PX from any verified tie point
# This generalises to any granule: on a cloud-free pass the same region is NOT
# protected and gets corrected normally.
AUTO_PROTECT = True
AP_WIN = 80            # half-window for local cloud/valid stats
AP_CLEAR_MIN = 0.55    # local clear-sky fraction below this -> not trusted (cloudy)
AP_VALID_MIN = 0.55    # local AVHRR-valid fraction below this -> not trusted (edge)
AP_FEATHER = 45        # px; feather the trust mask so the warp has no hard seam
AP_DROP_CLEAR = 0.45   # drop matched points whose local clear fraction < this
AP_DROP_VALID = 0.45   # drop matched points whose local valid fraction < this
GANGETIC_PROTECT = True  # kept as the on/off flag for auto-protection

# Cloud-masked region infill for specific hard boxes that stay misaligned:
# Gujarat (heavy cloud -> few matches) and the Bangladesh/Ganges delta (at the
# AVHRR swath edge -> valid-fraction gate rejects it). Clouds are removed from
# the AVHRR land mask (thermal mask) so the visible coastline drives matching,
# smaller tiles + a relaxed valid gate give denser coverage, and the shift is
# locked around the bow-tie model prediction. Boxes are (col0,col1,row0,row1).
REGION_INFILL = False
RI_BOXES = [(480, 860, 1550, 2250),     # Gujarat / Kathiawar
            (2350, 2850, 1650, 2260)]   # Bangladesh / Ganges delta
RI_TILES = (110, 150)                   # smaller windows -> more tie points
RI_STEP = 45
RI_SEARCH = 50
RI_SCORE_THRESH = 0.42
RI_VALID_MIN = 0.5                      # relaxed: these boxes sit at the swath edge
RI_DEDUP = 45

# Auto-curation (replaces manual QGIS picking).
NEIGHBOR_K = 4            # nearest neighbours for corroboration
NEIGHBOR_AGREE_PX = 25    # a neighbour "agrees" if its shift is within this
LOO_REJECT_PX = 25        # iteratively drop the worst point while LOO resid > this
CURATION_SMOOTHING = 15.0
DEDUP_RADIUS = 40         # merge multi-scale duplicate matches within this radius

# TPS warp (the user's Stage 6 parameters).
TPS_SMOOTHING = 15.0
TPS_GRID_STEP = 40
TPS_CLIP_MARGIN = 25


# ============================================================
# Load read-only manual-pipeline arrays
# ============================================================

def load_arrays():
    if not os.path.exists(os.path.join(INPUTS_DIR, "a_arr.npy")):
        raise SystemExit(
            f"Inputs not found in {INPUTS_DIR}.\n"
            f"Run the preprocessing stage first:\n"
            f"    conda run -n geo python {os.path.join(_BASE, 'preprocessing.py')}")

    def L(name):
        return np.load(os.path.join(INPUTS_DIR, name + ".npy"))
    arrays = {n: L(n) for n in
              ["a_arr", "s_arr", "m_arr", "b4_arr", "s_land", "m_land", "cloud_mask"]}

    grid = np.load(os.path.join(INPUTS_DIR, "grid.npz"), allow_pickle=True)
    geotransform = tuple(float(x) for x in grid["geotransform"])
    projection = str(grid["projection"])
    print(f"Loaded common-grid arrays {arrays['a_arr'].shape}, "
          f"grid origin ({geotransform[0]:.4f}, {geotransform[3]:.4f}) @ {geotransform[1]} deg")
    return arrays, geotransform, projection


def save_geotiff(array, geotransform, projection, path, dtype=gdal.GDT_Float32):
    driver = gdal.GetDriverByName("GTiff")
    ysize, xsize = array.shape
    out = driver.Create(path, xsize, ysize, 1, dtype)
    out.SetGeoTransform(geotransform)
    out.SetProjection(projection)
    out.GetRasterBand(1).WriteArray(array)
    out.FlushCache()
    out = None


# ============================================================
# Stage 4: dense tile matching on binary coastline masks
# (NCC coastline tile matching)
# ============================================================

def _match_one_scale(templ_img, templ_valid, ref_img, ref_valid, clear, tile):
    H, W = templ_img.shape
    tie_a, tie_b, scores = [], [], []
    for r in range(SEARCH + tile, H - SEARCH - tile, STEP):
        for c in range(SEARCH + tile, W - SEARCH - tile, STEP):
            if templ_valid[r:r + tile, c:c + tile].mean() < VALID_FRAC_MIN:
                continue
            if clear[r:r + tile, c:c + tile].mean() < CLOUD_GATE_MIN:
                continue
            t = templ_img[r:r + tile, c:c + tile]
            frac = (t > 127).mean()               # binary-mask land fraction
            if frac < FRAC_BOUNDS[0] or frac > FRAC_BOUNDS[1]:
                continue                           # near-uniform tile: no boundary to match
            sr0, sr1 = r - SEARCH, r + tile + SEARCH
            sc0, sc1 = c - SEARCH, c + tile + SEARCH
            res = cv2.matchTemplate(ref_img[sr0:sr1, sc0:sc1], t, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < SCORE_THRESH:
                continue
            dx = maxloc[0] - SEARCH
            dy = maxloc[1] - SEARCH
            if abs(dx) >= SEARCH - 8 or abs(dy) >= SEARCH - 8:
                continue                           # match at search boundary => likely false
            # peak distinctiveness: suppress a window around the best peak and
            # find the next-best. If the runner-up is nearly as strong, the
            # coastline pattern is ambiguous (repeated feature) and the match is
            # unreliable - the systematic failure mode that fools LOO/neighbour
            # checks because whole clusters lock onto the same wrong feature.
            res2 = res.copy()
            y0 = max(maxloc[1] - PEAK_SUPPRESS, 0); y1 = maxloc[1] + PEAK_SUPPRESS
            x0 = max(maxloc[0] - PEAK_SUPPRESS, 0); x1 = maxloc[0] + PEAK_SUPPRESS
            res2[y0:y1, x0:x1] = -1
            _, second, _, _ = cv2.minMaxLoc(res2)
            if second > PEAK_RATIO * maxval:
                continue
            # MODIS validity at the matched location only (not whole search window)
            mr0, mc0 = sr0 + maxloc[1], sc0 + maxloc[0]
            if ref_valid[mr0:mr0 + tile, mc0:mc0 + tile].mean() < MATCH_VALID_MIN:
                continue
            tie_a.append((c + tile / 2, r + tile / 2))
            tie_b.append((c + tile / 2 + dx, r + tile / 2 + dy))
            scores.append(maxval)
    return tie_a, tie_b, scores


def dense_tile_match(templ_img, templ_valid, ref_img, ref_valid, clear):
    """Multi-scale dense tile matching; keep the highest-scoring match per
    local neighbourhood so overlapping/multi-scale duplicates don't over-weight
    one spot."""
    all_a, all_b, all_s = [], [], []
    for tile in TILE_SIZES:
        a, b, s = _match_one_scale(templ_img, templ_valid, ref_img, ref_valid, clear, tile)
        all_a += a; all_b += b; all_s += s
    if not all_a:
        return np.array([]), np.array([]), np.array([])

    a = np.array(all_a); b = np.array(all_b); s = np.array(all_s)

    # greedy dedup: take highest score first, suppress others within DEDUP_RADIUS
    order = np.argsort(-s)
    kept = []
    taken = np.zeros(len(s), dtype=bool)
    for i in order:
        if taken[i]:
            continue
        kept.append(i)
        d = np.hypot(a[:, 0] - a[i, 0], a[:, 1] - a[i, 1])
        taken |= d < DEDUP_RADIUS
    kept = np.array(kept)
    return a[kept], b[kept], s[kept]


def west_infill_match(templ_img, templ_valid, ref_img, ref_valid, clear):
    """Permissive matching pass restricted to the western columns, to recover
    weak/cloudy far-west coastline anchors that the strict main pass drops.
    Kept honest by the downstream neighbour-corroboration + LOO curation."""
    H, W = templ_img.shape
    ta, tb, sc = [], [], []
    for tile in TILE_SIZES:
        half = tile // 2
        for r in range(SEARCH + tile, H - SEARCH - tile, STEP):
            col_max = W - SEARCH - tile if WEST_COL_MAX is None else min(WEST_COL_MAX, W - SEARCH - tile)
            for c in range(SEARCH + tile, col_max, STEP):
                if templ_valid[r:r + tile, c:c + tile].mean() < VALID_FRAC_MIN:
                    continue
                if clear[r:r + tile, c:c + tile].mean() < WEST_CLOUD_MIN:
                    continue
                t = templ_img[r:r + tile, c:c + tile]
                frac = (t > 127).mean()
                if frac < FRAC_BOUNDS[0] or frac > FRAC_BOUNDS[1]:
                    continue
                sr0, sr1 = r - SEARCH, r + tile + SEARCH
                sc0, sc1 = c - SEARCH, c + tile + SEARCH
                res = cv2.matchTemplate(ref_img[sr0:sr1, sc0:sc1], t, cv2.TM_CCOEFF_NORMED)
                _, maxval, _, maxloc = cv2.minMaxLoc(res)
                if maxval < WEST_SCORE_THRESH:
                    continue
                dx, dy = maxloc[0] - SEARCH, maxloc[1] - SEARCH
                if abs(dx) >= SEARCH - 8 or abs(dy) >= SEARCH - 8:
                    continue
                res2 = res.copy()
                y0 = max(maxloc[1] - PEAK_SUPPRESS, 0); x0 = max(maxloc[0] - PEAK_SUPPRESS, 0)
                res2[y0:maxloc[1] + PEAK_SUPPRESS, x0:maxloc[0] + PEAK_SUPPRESS] = -1
                _, second, _, _ = cv2.minMaxLoc(res2)
                if second > WEST_PEAK_RATIO * maxval:
                    continue
                mr0, mc0 = sr0 + maxloc[1], sc0 + maxloc[0]
                if ref_valid[mr0:mr0 + tile, mc0:mc0 + tile].mean() < MATCH_VALID_MIN:
                    continue
                ta.append((c + half, r + half))
                tb.append((c + half + dx, r + half + dy))
                sc.append(maxval)
    return np.array(ta), np.array(tb), np.array(sc)


# ============================================================
# Seed-and-grow: extend coverage into the west coast / Gujarat
# ============================================================

def _poly_design(col, row, order):
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((col ** i) * (row ** j))
    return np.column_stack(terms)


def fit_bowtie_model(ta, off, order=GROW_MODEL_ORDER):
    """Low-order polynomial dx,dy = f(col,row). Extrapolates smoothly (unlike
    TPS), so it can predict shifts in regions with no tie points yet."""
    A = _poly_design(ta[:, 0].astype(float), ta[:, 1].astype(float), order)
    cx = np.linalg.lstsq(A, off[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A, off[:, 1], rcond=None)[0]

    def predict(cols, rows):
        Ap = _poly_design(np.asarray(cols, float), np.asarray(rows, float), order)
        return Ap @ cx, Ap @ cy
    return predict


def auto_protect_drop(ta, off, sc, arrays, verbose=True):
    """Drop matched points that fall in strongly cloudy / off-swath cells (their
    matches are unreliable). Data-driven, no hardcoded regions."""
    s_valid = arrays["s_arr"] > 0
    clear = ~arrays["cloud_mask"]
    w = AP_WIN

    def local(mask, r, c):
        return mask[max(r - w, 0):r + w, max(c - w, 0):c + w].mean()

    keep = np.ones(len(ta), dtype=bool)
    for i, (c, r) in enumerate(ta):
        ri, ci = int(r), int(c)
        if local(clear, ri, ci) < AP_DROP_CLEAR or local(s_valid, ri, ci) < AP_DROP_VALID:
            keep[i] = False
    if verbose:
        print(f"Auto-protect: dropped {int((~keep).sum())} cloud/edge matched points")
    return ta[keep], off[keep], sc[keep]


def build_trust_mask(ta, arrays):
    """
    Per-pixel weight (0..1) for HOW MUCH of the fitted shift field to apply.
    1 where the correction is trustworthy (clear, on-swath, INSIDE the convex
    hull of tie points); tapered to 0 in cloudy / swath-edge / extrapolated
    regions so those keep their original geolocation. Feathered so the warp has
    no hard seams. Data-driven -> generalises to any granule.
    """
    from scipy.ndimage import uniform_filter
    s_valid = arrays["s_arr"] > 0
    clear = ~arrays["cloud_mask"]
    H, W = s_valid.shape

    clear_f = uniform_filter(clear.astype(np.float32), 2 * AP_WIN)
    valid_f = uniform_filter(s_valid.astype(np.float32), 2 * AP_WIN)
    trust = (clear_f >= AP_CLEAR_MIN) & (valid_f >= AP_VALID_MIN) & s_valid

    if len(ta) >= 4:
        from scipy.spatial import Delaunay
        try:
            hull = Delaunay(ta)
            step = 40
            ys, xs = np.mgrid[0:H:step, 0:W:step]
            insideC = (hull.find_simplex(np.column_stack([xs.ravel(), ys.ravel()])) >= 0)
            inside = cv2.resize(insideC.reshape(xs.shape).astype(np.float32), (W, H),
                                interpolation=cv2.INTER_LINEAR) > 0.5
            trust &= inside
        except Exception:
            pass

    return gaussian_filter(trust.astype(np.float32), AP_FEATHER)


def gradient_structure(arr, valid):
    """Gradient-magnitude structure image (edges: ridges, rivers, snow lines).
    Invalid pixels filled with the median so no false edges form at borders."""
    f = arr.astype(np.float32).copy()
    f[~valid] = np.nanmedian(arr[valid])
    f = gaussian_filter(f, IM_GAUSS)
    return np.hypot(sobel(f, 0), sobel(f, 1))


def interior_structure_match(a_struct, m_struct, a_arr, m_arr, a_valid, m_valid, clear):
    """Cross-sensor terrain-structure matching for the interior/North. Wide
    search + distinctiveness + real-image NCC gate. Returns tie points where
    genuine terrain texture (mostly the Himalayan ridge and north) matches."""
    H, W = a_struct.shape
    tile, half = IM_TILE, IM_TILE // 2
    ta, tb, sc = [], [], []
    for r in range(IM_SEARCH + tile, H - IM_SEARCH - tile, IM_STEP):
        for c in range(IM_SEARCH + tile, W - IM_SEARCH - tile, IM_STEP):
            if a_valid[r:r + tile, c:c + tile].mean() < VALID_FRAC_MIN:
                continue
            if clear[r:r + tile, c:c + tile].mean() < IM_CLOUD_MIN:
                continue
            t = a_struct[r:r + tile, c:c + tile]
            if t.std() < IM_STD_MIN:
                continue                          # too little texture to match
            sr0, sr1 = r - IM_SEARCH, r + tile + IM_SEARCH
            sc0, sc1 = c - IM_SEARCH, c + tile + IM_SEARCH
            res = cv2.matchTemplate(m_struct[sr0:sr1, sc0:sc1], t, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < IM_SCORE_THRESH:
                continue
            r2 = res.copy()
            r2[max(maxloc[1] - PEAK_SUPPRESS, 0):maxloc[1] + PEAK_SUPPRESS,
               max(maxloc[0] - PEAK_SUPPRESS, 0):maxloc[0] + PEAK_SUPPRESS] = -1
            _, second, _, _ = cv2.minMaxLoc(r2)
            if second > IM_PEAK_RATIO * maxval:
                continue
            dx, dy = maxloc[0] - IM_SEARCH, maxloc[1] - IM_SEARCH
            if abs(dx) >= IM_SEARCH - 8 or abs(dy) >= IM_SEARCH - 8:
                continue
            mr0, mc0 = sr0 + maxloc[1], sc0 + maxloc[0]
            if m_valid[mr0:mr0 + tile, mc0:mc0 + tile].mean() < MATCH_VALID_MIN:
                continue
            ta.append((c + half, r + half))
            tb.append((c + half + dx, r + half + dy))
            sc.append(maxval)
    return np.array(ta), np.array(tb), np.array(sc)


def region_infill(model, s_land, m_land, cloud, s_valid, m_valid):
    """Cloud-masked, model-constrained coastline matching inside the RI_BOXES
    (Gujarat, Bangladesh delta). AVHRR clouds are removed from the land mask so
    the visible coastline drives matching; smaller tiles + relaxed valid gate
    give denser coverage; the search is locked around the model's prediction."""
    clear = ~cloud
    H, W = s_land.shape
    s_nc = cv2.GaussianBlur(((s_land & clear).astype(np.uint8) * 255), (5, 5), 1.0)
    m_u8 = cv2.GaussianBlur((m_land.astype(np.uint8) * 255), (5, 5), 1.0)

    ta, tb, sc = [], [], []
    for (c0, c1, r0, r1) in RI_BOXES:
        for tile in RI_TILES:
            half = tile // 2
            for r in range(r0, min(r1, H - tile), RI_STEP):
                for c in range(c0, min(c1, W - tile), RI_STEP):
                    if s_valid[r:r + tile, c:c + tile].mean() < RI_VALID_MIN:
                        continue
                    t = s_nc[r:r + tile, c:c + tile]
                    frac = (t > 127).mean()
                    if frac < 0.04 or frac > 0.96:
                        continue
                    pdx, pdy = model([c + half], [r + half])
                    pdx, pdy = float(pdx[0]), float(pdy[0])
                    sr0 = r + int(round(pdy)) - RI_SEARCH
                    sc0 = c + int(round(pdx)) - RI_SEARCH
                    sr1 = sr0 + tile + 2 * RI_SEARCH
                    sc1 = sc0 + tile + 2 * RI_SEARCH
                    if sr0 < 0 or sc0 < 0 or sr1 > H or sc1 > W:
                        continue
                    res = cv2.matchTemplate(m_u8[sr0:sr1, sc0:sc1], t, cv2.TM_CCOEFF_NORMED)
                    _, maxval, _, maxloc = cv2.minMaxLoc(res)
                    if maxval < RI_SCORE_THRESH:
                        continue
                    mr0, mc0 = sr0 + maxloc[1], sc0 + maxloc[0]
                    if m_valid[mr0:mr0 + tile, mc0:mc0 + tile].mean() < MATCH_VALID_MIN:
                        continue
                    ta.append((c + half, r + half))
                    tb.append((c + half + (mc0 - c), r + half + (mr0 - r)))
                    sc.append(maxval)
    if not ta:
        return np.array([]), np.array([]), np.array([])
    a, b, s = np.array(ta), np.array(tb), np.array(sc)
    return dedup_points(a, b, s, radius=RI_DEDUP)


def straight_coast_match(model, templ_img, templ_valid, ref_img, ref_valid, clear):
    """Across-shore (dx) matching for the straight SW coast. Search is wide in
    x (measures dx) and narrow in y (dy taken from the model prediction), which
    resolves the along-shore aperture ambiguity that otherwise yields no good
    points there. `model` is a fitted bow-tie predictor (fit_bowtie_model)."""
    c0r, c1r, r0r, r1r = SC_REGION
    H, W = templ_img.shape
    tile, half = SC_TILE, SC_TILE // 2
    ta, tb, sc = [], [], []
    for r in range(r0r, min(r1r, H - tile), SC_STEP):
        for c in range(c0r, min(c1r, W - tile), SC_STEP):
            if templ_valid[r:r + tile, c:c + tile].mean() < VALID_FRAC_MIN:
                continue
            if clear[r:r + tile, c:c + tile].mean() < SC_CLOUD_MIN:
                continue
            t = templ_img[r:r + tile, c:c + tile]
            frac = (t > 127).mean()
            if frac < FRAC_BOUNDS[0] or frac > FRAC_BOUNDS[1]:
                continue
            pdx, pdy = model([c + half], [r + half])
            pdx, pdy = float(pdx[0]), float(pdy[0])
            sr0 = r + int(round(pdy)) - SC_SEARCH_Y
            sc0 = c + int(round(pdx)) - SC_SEARCH_X
            sr1 = sr0 + tile + 2 * SC_SEARCH_Y
            sc1 = sc0 + tile + 2 * SC_SEARCH_X
            if sr0 < 0 or sc0 < 0 or sr1 > H or sc1 > W:
                continue
            res = cv2.matchTemplate(ref_img[sr0:sr1, sc0:sc1], t, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < SC_SCORE_THRESH:
                continue
            mr0, mc0 = sr0 + maxloc[1], sc0 + maxloc[0]
            if ref_valid[mr0:mr0 + tile, mc0:mc0 + tile].mean() < MATCH_VALID_MIN:
                continue
            ta.append((c + half, r + half))
            tb.append((c + half + (mc0 - c), r + half + (mr0 - r)))
            sc.append(maxval)
    return np.array(ta), np.array(tb), np.array(sc)


def grow_matches(seed_a, seed_b, seed_s, templ_img, templ_valid, ref_img,
                 ref_valid, clear):
    """
    Iteratively grow the tie-point set using the smoothness prior:
      1. fit a bow-tie model from current points,
      2. sweep a grid; at each tile predict the shift and run matchTemplate in a
         SMALL window (+/-GROW_PRED_TOL) around the predicted MODIS location,
      3. accept the match if it scores well, lands within GROW_ACCEPT_PX of the
         prediction, and its matched MODIS region is valid.
    The tight predicted window is what makes straight-coast (west) and cloudy
    (Gujarat) tiles matchable without spurious long-range slips.
    """
    ta = list(seed_a); tb = list(seed_b); sc = list(seed_s)
    H, W = templ_img.shape
    tile = GROW_TILE
    half = tile // 2

    for it in range(GROW_ITERS):
        predict = fit_bowtie_model(np.array(ta), np.array(tb) - np.array(ta))
        existing = np.array(ta)
        added = 0
        for r in range(half, H - half, GROW_STEP):
            for c in range(half, W - half, GROW_STEP):
                cyx = c; cyy = r
                # skip if we already have a point very close
                if len(existing) and np.min(np.hypot(existing[:, 0] - cyx,
                                                      existing[:, 1] - cyy)) < GROW_STEP:
                    continue
                r0, c0 = r - half, c - half
                if templ_valid[r0:r0 + tile, c0:c0 + tile].mean() < VALID_FRAC_MIN:
                    continue
                if clear[r0:r0 + tile, c0:c0 + tile].mean() < GROW_CLOUD_MIN:
                    continue
                t = templ_img[r0:r0 + tile, c0:c0 + tile]
                frac = (t > 127).mean()
                if frac < FRAC_BOUNDS[0] or frac > FRAC_BOUNDS[1]:
                    continue
                pdx, pdy = predict([cyx], [cyy])
                pdx, pdy = float(pdx[0]), float(pdy[0])
                # search window in the reference, locked around the prediction
                sr0 = r0 + int(round(pdy)) - GROW_PRED_TOL
                sc0 = c0 + int(round(pdx)) - GROW_PRED_TOL
                sr1 = sr0 + tile + 2 * GROW_PRED_TOL
                sc1 = sc0 + tile + 2 * GROW_PRED_TOL
                if sr0 < 0 or sc0 < 0 or sr1 > H or sc1 > W:
                    continue
                res = cv2.matchTemplate(ref_img[sr0:sr1, sc0:sc1], t, cv2.TM_CCOEFF_NORMED)
                _, maxval, _, maxloc = cv2.minMaxLoc(res)
                if maxval < GROW_SCORE_THRESH:
                    continue
                mr0, mc0 = sr0 + maxloc[1], sc0 + maxloc[0]
                dy = mr0 - r0
                dx = mc0 - c0
                if np.hypot(dx - pdx, dy - pdy) > GROW_ACCEPT_PX:
                    continue
                if ref_valid[mr0:mr0 + tile, mc0:mc0 + tile].mean() < MATCH_VALID_MIN:
                    continue
                ta.append((cyx, cyy))
                tb.append((cyx + dx, cyy + dy))
                sc.append(maxval)
                existing = np.array(ta)
                added += 1
        print(f"  grow iter {it+1}: +{added} points (total {len(ta)})")
        if added == 0:
            break

    return np.array(ta), np.array(tb), np.array(sc)


# ============================================================
# Stage 5: automatic bow-tie-consistency curation
# (replaces manual QGIS picking; uses the manual Stage 5 logic:
#  neighbour corroboration + leave-one-out TPS residual)
# ============================================================

def loo_residuals(ta, off, smoothing):
    """Leave-one-out TPS prediction error per point (px)."""
    n = len(ta)
    resid = np.full(n, np.inf)
    if n < 6:
        return resid
    idx = np.arange(n)
    for i in idx:
        keep = idx != i
        try:
            rx = RBFInterpolator(ta[keep], off[keep, 0],
                                 kernel="thin_plate_spline", smoothing=smoothing)
            ry = RBFInterpolator(ta[keep], off[keep, 1],
                                 kernel="thin_plate_spline", smoothing=smoothing)
            px = rx(ta[i:i + 1])[0]
            py = ry(ta[i:i + 1])[0]
            resid[i] = np.hypot(px - off[i, 0], py - off[i, 1])
        except Exception:
            resid[i] = np.inf
    return resid


def neighbour_corroborated(ta, off):
    """Keep a point only if >=1 of its K nearest neighbours has a similar shift."""
    n = len(ta)
    if n <= NEIGHBOR_K:
        return np.ones(n, dtype=bool)
    tree = cKDTree(ta)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        _, nbr = tree.query(ta[i], k=NEIGHBOR_K + 1)
        nbr = [j for j in nbr if j != i]
        d = np.hypot(off[nbr, 0] - off[i, 0], off[nbr, 1] - off[i, 1])
        ok[i] = np.any(d <= NEIGHBOR_AGREE_PX)
    return ok


def auto_curate(tie_a, tie_b, scores):
    off = tie_b - tie_a

    # 1. neighbour corroboration (isolated high-score points are unreliable)
    ok = neighbour_corroborated(tie_a, off)
    ta, off, sc = tie_a[ok], off[ok], scores[ok]
    print(f"  neighbour corroboration: {ok.sum()}/{len(ok)} kept")

    # 2. iterative LOO rejection: drop the single worst point while its
    #    leave-one-out TPS error exceeds LOO_REJECT_PX, refit, repeat.
    while len(ta) >= 6:
        resid = loo_residuals(ta, off, CURATION_SMOOTHING)
        worst = int(np.argmax(resid))
        if resid[worst] <= LOO_REJECT_PX:
            break
        keep = np.arange(len(ta)) != worst
        ta, off, sc = ta[keep], off[keep], sc[keep]

    resid = loo_residuals(ta, off, CURATION_SMOOTHING)
    print(f"  iterative LOO curation: {len(ta)} points kept, "
          f"LOO RMSE={np.sqrt(np.mean(resid**2)):.1f}px median={np.median(resid):.1f}px")
    return ta, off, sc, resid


# ============================================================
# Stage 6: TPS warp (ported from stage6_tps_warp.py)
# ============================================================

def fit_tps_field(ta, off, W, H):
    rbf_dx = RBFInterpolator(ta, off[:, 0], kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)
    rbf_dy = RBFInterpolator(ta, off[:, 1], kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)

    gx = np.linspace(0, W - 1, W // TPS_GRID_STEP + 2)
    gy = np.linspace(0, H - 1, H // TPS_GRID_STEP + 2)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)

    dx_grid = rbf_dx(grid_pts).reshape(GX.shape).astype(np.float32)
    dy_grid = rbf_dy(grid_pts).reshape(GX.shape).astype(np.float32)

    dx_grid = np.clip(dx_grid, off[:, 0].min() - TPS_CLIP_MARGIN, off[:, 0].max() + TPS_CLIP_MARGIN)
    dy_grid = np.clip(dy_grid, off[:, 1].min() - TPS_CLIP_MARGIN, off[:, 1].max() + TPS_CLIP_MARGIN)

    dxf = cv2.resize(dx_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    dyf = cv2.resize(dy_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    return dxf, dyf


def build_shift_field(ta, off, arrays):
    """Fit the TPS field from the tie points and (if enabled) multiply by the
    data-driven trust mask so cloudy / swath-edge / extrapolated regions keep
    their original geolocation. Used by both the pipeline and the report."""
    H, W = arrays["a_arr"].shape
    dxf, dyf = fit_tps_field(ta, off, W, H)
    if AUTO_PROTECT:
        trust = build_trust_mask(ta, arrays)
        dxf = dxf * trust
        dyf = dyf * trust
    return dxf, dyf


def warp_with_field(arr, dxf, dyf):
    H, W = arr.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    src_x = (xs - dxf).astype(np.float32)   # inverse mapping: output(x)=input(x-dx)
    src_y = (ys - dyf).astype(np.float32)
    return cv2.remap(arr.astype(np.float32), src_x, src_y,
                     interpolation=cv2.INTER_LINEAR, borderValue=0)


# ============================================================
# Validation against the manual ground truth
# ============================================================

def validate(ta, off, dxf, dyf, warped, arrays, geotransform, projection):
    if not (os.path.exists(GT_TA) and os.path.exists(GT_OFF)):
        print("(no ground-truth tie points in inputs/ - skipping validation)")
        return
    cur_ta = np.load(GT_TA)
    cur_off = np.load(GT_OFF)

    # (1) does MY field predict THEIR curated displacements?
    pred_dx = dxf[np.clip(cur_ta[:, 1].astype(int), 0, dxf.shape[0] - 1),
                  np.clip(cur_ta[:, 0].astype(int), 0, dxf.shape[1] - 1)]
    pred_dy = dyf[np.clip(cur_ta[:, 1].astype(int), 0, dyf.shape[0] - 1),
                  np.clip(cur_ta[:, 0].astype(int), 0, dyf.shape[1] - 1)]
    err = np.hypot(pred_dx - cur_off[:, 0], pred_dy - cur_off[:, 1])
    print(f"(1) My TPS field vs their 66 curated displacements: "
          f"RMSE={np.sqrt(np.mean(err**2)):.1f}px median={np.median(err):.1f}px max={err.max():.1f}px")

    # (2) my auto tie points vs their nearest curated point
    tree = cKDTree(cur_ta)
    d, idx = tree.query(ta)
    near = d < 120
    if near.sum():
        shift_err = np.hypot(off[near, 0] - cur_off[idx[near], 0],
                             off[near, 1] - cur_off[idx[near], 1])
        print(f"(2) My auto points near a curated point ({near.sum()} of {len(ta)}): "
              f"shift agreement RMSE={np.sqrt(np.mean(shift_err**2)):.1f}px")

    # (3) bow-tie sanity: my field's dx sign should flip across the nadir (~col 1550)
    print(f"(3) Bow-tie check - field mean dx by column band:")
    for lo, hi in [(700, 1000), (1000, 1300), (1300, 1600), (1600, 1900), (1900, 2600)]:
        print(f"    col[{lo}-{hi}]: mean dx = {dxf[:, lo:hi].mean():+.1f}")


# ============================================================
# Main
# ============================================================

def median_consistency_mask(ta, off, k=6, thresh=40):
    """Reject a point whose shift deviates > `thresh` from the MEDIAN shift of
    its k nearest neighbours. Stricter than 'at least one neighbour agrees'
    (which lets a wrong point survive if any single neighbour happens to match),
    so it catches lone over-shifting matches like the Karnataka coast point."""
    if len(ta) <= k:
        return np.ones(len(ta), dtype=bool)
    tree = cKDTree(ta)
    keep = np.ones(len(ta), dtype=bool)
    for i in range(len(ta)):
        _, nb = tree.query(ta[i], k=k + 1)
        nb = [j for j in np.atleast_1d(nb) if j != i]
        med = np.median(off[nb], axis=0)
        keep[i] = np.hypot(off[i, 0] - med[0], off[i, 1] - med[1]) <= thresh
    return keep


def dedup_points(ta, tb, sc, radius=40):
    """Greedy dedup by AVHRR location (keep highest score within `radius`).
    Applied to the MERGED main+west-infill+grow set so a point can't corroborate
    itself via a duplicate - that self-corroboration was letting single wrong
    matches (e.g. an over-shifting Karnataka point) survive the neighbour check."""
    if len(ta) == 0:
        return ta, tb, sc
    order = np.argsort(-sc)
    taken = np.zeros(len(sc), dtype=bool)
    keep = []
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        taken |= np.hypot(ta[:, 0] - ta[i, 0], ta[:, 1] - ta[i, 1]) < radius
    keep = np.array(keep)
    return ta[keep], tb[keep], sc[keep]


def _ncc(a, b, mask=None):
    v = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        v &= mask                       # exclude cloud pixels from the correlation
    if v.sum() < a.size * 0.2:
        return np.nan
    av = a[v] - a[v].mean(); bv = b[v] - b[v].mean()
    dd = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av * bv).sum() / dd) if dd else np.nan


def ncc_valid_mask(ta, off, a_arr, m_arr, win=90, clear=None):
    """
    Keep only tie points whose shift actually improves alignment on the REAL
    imagery (AVHRR visible vs MODIS), independent of any field model:
      before = NCC(MODIS@dest, AVHRR@dest);  after = NCC(MODIS@dest, AVHRR@src).
    A wrong match (e.g. the Karnataka straight-coast slips) fails to improve and
    is dropped - this is what removes points that visibly degrade the overlay,
    which the internal neighbour/LOO checks can't catch when wrong points
    cluster. Matching used SWIR, this check uses the visible band, so it's a
    semi-independent cross-check.
    """
    H, W = a_arr.shape
    keep = np.zeros(len(ta), dtype=bool)
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(ta, off)):
        ax, ay = int(round(sx)), int(round(sy))
        bx, by = int(round(sx + dx)), int(round(sy + dy))
        if not (win <= ay < H - win and win <= ax < W - win and
                win <= by < H - win and win <= bx < W - win):
            continue
        m_ref = m_arr[by - win:by + win, bx - win:bx + win]
        cl = None
        if clear is not None:
            cl = clear[by - win:by + win, bx - win:bx + win] & clear[ay - win:ay + win, ax - win:ax + win]
        before = _ncc(a_arr[by - win:by + win, bx - win:bx + win], m_ref, cl)
        after = _ncc(a_arr[ay - win:ay + win, ax - win:ax + win], m_ref, cl)
        keep[i] = np.isfinite(after) and np.isfinite(before) and after > before
    return keep


def extract_tie_points(arrays, verbose=True):
    """Full automatic tie-point extraction (matching -> west-infill -> grow ->
    curation). Shared by main() and the report/visualization scripts so they
    all use the identical tie-point set."""
    m_arr = arrays["m_arr"]
    clear = ~arrays["cloud_mask"]
    s_valid = arrays["s_arr"] > 0
    m_valid = ~np.isnan(m_arr) & (m_arr > 0)
    s_u8 = cv2.GaussianBlur((arrays["s_land"].astype(np.uint8) * 255), (5, 5), 1.0)
    m_u8 = cv2.GaussianBlur((arrays["m_land"].astype(np.uint8) * 255), (5, 5), 1.0)

    tie_a, tie_b, scores = dense_tile_match(s_u8, s_valid, m_u8, m_valid, clear)
    if verbose:
        print(f"Dense matching: {len(tie_a)} candidates")

    if WEST_INFILL:
        wa, wb, ws = west_infill_match(s_u8, s_valid, m_u8, m_valid, clear)
        if verbose:
            print(f"Permissive infill: +{len(wa)} candidates (whole scene)")
        if len(wa):
            tie_a = np.vstack([tie_a, wa]); tie_b = np.vstack([tie_b, wb])
            scores = np.concatenate([scores, ws])

    if GROW:
        seed_a, seed_off, seed_sc, _ = auto_curate(tie_a, tie_b, scores)
        seed_b = seed_a + seed_off
        tie_a, tie_b, scores = grow_matches(seed_a, seed_b, seed_sc,
                                            s_u8, s_valid, m_u8, m_valid, clear)
        if verbose:
            print(f"Seed-and-grow: {len(tie_a)} candidates")

    if STRAIGHT_COAST:
        # fit a bow-tie model from the confident (curated) points, then add
        # across-shore dx matches along the straight SW coast
        ca, coff2, _, _ = auto_curate(tie_a, tie_b, scores)
        model = fit_bowtie_model(ca, coff2)
        sa, sb, ss = straight_coast_match(model, s_u8, s_valid, m_u8, m_valid, clear)
        if verbose:
            print(f"Straight-coast (SW) dx-match: +{len(sa)} candidates")
        if len(sa):
            tie_a = np.vstack([tie_a, sa]); tie_b = np.vstack([tie_b, sb])
            scores = np.concatenate([scores, ss])

    if INTERIOR_MATCH:
        a_valid = arrays["a_arr"] > 0
        a_struct = gradient_structure(arrays["a_arr"], a_valid)
        m_struct = gradient_structure(m_arr, m_valid)
        ia, ib, isc = interior_structure_match(a_struct, m_struct, arrays["a_arr"].astype(np.float32),
                                               m_arr.astype(np.float32), a_valid, m_valid, clear)
        # keep only interior matches that pass the real-image NCC gate
        if len(ia):
            keep = ncc_valid_mask(ia, ib - ia, arrays["a_arr"].astype(np.float32), m_arr.astype(np.float32))
            ia, ib, isc = ia[keep], ib[keep], isc[keep]
        if verbose:
            print(f"Interior/North structure match: +{len(ia)} NCC-valid candidates")
        if len(ia):
            tie_a = np.vstack([tie_a, ia]); tie_b = np.vstack([tie_b, ib])
            scores = np.concatenate([scores, isc])

    if REGION_INFILL:
        ca, coff2, _, _ = auto_curate(tie_a, tie_b, scores)
        model = fit_bowtie_model(ca, coff2)
        ra, rb, rsc = region_infill(model, arrays["s_land"], arrays["m_land"],
                                    arrays["cloud_mask"], s_valid, m_valid)
        # cloud-masked NCC gate so cloudy-but-real matches (Gujarat) survive
        if len(ra):
            keep = ncc_valid_mask(ra, rb - ra, arrays["a_arr"].astype(np.float32),
                                  arrays["m_arr"].astype(np.float32), clear=clear)
            ra, rb, rsc = ra[keep], rb[keep], rsc[keep]
        if verbose:
            print(f"Region infill (Gujarat + Bangladesh delta): +{len(ra)} NCC-valid candidates")
        if len(ra):
            tie_a = np.vstack([tie_a, ra]); tie_b = np.vstack([tie_b, rb])
            scores = np.concatenate([scores, rsc])

    # dedup the merged main+west-infill+grow set so no point self-corroborates
    tie_a, tie_b, scores = dedup_points(tie_a, tie_b, scores)
    if verbose:
        print(f"After dedup: {len(tie_a)} candidates")

    ta, off, sc, resid = auto_curate(tie_a, tie_b, scores)

    # local-median consistency: drop lone points whose shift disagrees with the
    # median of their neighbours (catches over-shifting matches the weaker
    # 'one neighbour agrees' rule keeps)
    keep = median_consistency_mask(ta, off)
    if verbose:
        print(f"Local-median consistency: {keep.sum()}/{len(ta)} points kept")
    ta, off, sc = ta[keep], off[keep], sc[keep]

    # final data-driven filter: keep only points that improve NCC on the real
    # imagery (removes wrong-but-self-consistent matches like the Karnataka slips)
    keep = ncc_valid_mask(ta, off, arrays["a_arr"].astype(np.float32),
                          arrays["m_arr"].astype(np.float32), clear=clear)
    if verbose:
        print(f"NCC validity filter: {keep.sum()}/{len(ta)} points improve real alignment")
    ta, off, sc = ta[keep], off[keep], sc[keep]

    if AUTO_PROTECT:
        ta, off, sc = auto_protect_drop(ta, off, sc, arrays, verbose=verbose)

    return ta, off, sc


def main():
    global INPUTS_DIR, OUTPUT_DIR, GT_TA, GT_OFF
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default=INPUTS_DIR)
    p.add_argument("--output", default=OUTPUT_DIR)
    args = p.parse_args()
    INPUTS_DIR, OUTPUT_DIR = args.inputs, args.output
    GT_TA = os.path.join(INPUTS_DIR, "curated_ta.npy")
    GT_OFF = os.path.join(INPUTS_DIR, "curated_off.npy")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    arrays, geotransform, projection = load_arrays()

    print("\n" + "=" * 60)
    print("TIE-POINT EXTRACTION (match -> west-infill -> grow -> curate)")
    print("=" * 60)
    ta, off, sc = extract_tie_points(arrays)
    np.save(os.path.join(OUTPUT_DIR, "auto_ta.npy"), ta)
    np.save(os.path.join(OUTPUT_DIR, "auto_off.npy"), off)
    np.save(os.path.join(OUTPUT_DIR, "auto_sc.npy"), sc)  # sc<0 marks Gangetic zero-anchors
    print(f"dx: mean={off[:,0].mean():+.1f} (expect ~0)  "
          f"min={off[:,0].min():+.1f} max={off[:,0].max():+.1f}")
    print(f"dy: mean={off[:,1].mean():+.1f}  min={off[:,1].min():+.1f} max={off[:,1].max():+.1f}")

    print("\n" + "=" * 60)
    print("STAGE 6: TPS WARP")
    print("=" * 60)
    dxf, dyf = build_shift_field(ta, off, arrays)
    warped = warp_with_field(arrays["a_arr"], dxf, dyf)
    save_geotiff(dxf, geotransform, projection, os.path.join(OUTPUT_DIR, "shift_field_dx.tif"))
    save_geotiff(dyf, geotransform, projection, os.path.join(OUTPUT_DIR, "shift_field_dy.tif"))
    save_geotiff(warped, geotransform, projection, os.path.join(OUTPUT_DIR, "avhrr_bowtie_corrected.tif"))
    print(f"Wrote {OUTPUT_DIR}/avhrr_bowtie_corrected.tif")

    print("\n" + "=" * 60)
    print("VALIDATION AGAINST MANUAL GROUND TRUTH")
    print("=" * 60)
    validate(ta, off, dxf, dyf, warped, arrays, geotransform, projection)


if __name__ == "__main__":
    main()
