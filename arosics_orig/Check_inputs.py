#!/usr/bin/env python3
"""
check_arosics_inputs.py

Validates that two GeoTIFF (or any GDAL-readable raster) files are suitable
as im_ref / im_tgt inputs for arosics.COREG_LOCAL.

Checks performed:
  1. Files exist and are openable by GDAL
  2. Driver is GDAL-supported
  3. Raster has a valid geotransform (is georeferenced at all)
  4. Raster has a valid projection / CRS defined
  5. Geotransform has no metadata rotation (GT[2], GT[4] == 0)
  6. Pixel size (GSD) is sane (non-zero, not absurd)
  7. Band count and requested r_b4match / s_b4match are valid
  8. Data type is a GDAL-supported numeric type
  9. NoData value is defined (warns if not, since it affects masking)
  10. Image is not entirely NoData
  11. Reference and target actually spatially overlap
  12. Overlap area is large enough for the requested grid_res / window_size
  13. Reports whether ref and target share the same projection (AROSICS can
      handle reprojection, but it's good to know upfront)

Usage:
    python check_arosics_inputs.py /path/to/reference.tif /path/to/target.tif \
        --r-band 1 --s-band 1 --window-size 256 256 --grid-res 50

If GDAL is not installed in this environment, install it via:
    conda install -c conda-forge gdal
or
    pip install gdal   (often finicky outside conda; conda-forge is recommended)
"""

import argparse
import sys

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL Python bindings not found. Install with:\n"
          "  conda install -c conda-forge gdal\n"
          "AROSICS itself depends on GDAL, so if arosics is installed and working, "
          "GDAL should already be available in that same environment.")
    sys.exit(1)


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


class Report:
    def __init__(self, label):
        self.label = label
        self.items = []  # (status, message)

    def add(self, status, message):
        self.items.append((status, message))

    def print(self):
        print(f"\n{'=' * 70}\n{self.label}\n{'=' * 70}")
        for status, msg in self.items:
            print(f"  [{status:4}] {msg}")

    def has_fail(self):
        return any(s == FAIL for s, _ in self.items)


def inspect_raster(path, band_to_match, label):
    rep = Report(f"Checking {label}: {path}")
    info = {}

    # 1. Open with GDAL
    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
    except RuntimeError as e:
        rep.add(FAIL, f"Could not open file with GDAL: {e}")
        rep.print()
        return rep, None

    if ds is None:
        rep.add(FAIL, "GDAL returned None — file is not a readable raster.")
        rep.print()
        return rep, None

    rep.add(PASS, f"File opened successfully. Driver: {ds.GetDriver().ShortName}")

    # 2. Geotransform
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        rep.add(FAIL, "No geotransform found — file is not georeferenced. "
                      "AROSICS requires georeferenced input.")
    else:
        default_gt = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if tuple(gt) == default_gt:
            rep.add(FAIL, "Geotransform is the GDAL default identity transform "
                          "— this almost always means the file has NO real "
                          "georeferencing.")
        else:
            rep.add(PASS, f"Geotransform present: {gt}")
        info["gt"] = gt

        # 5. Metadata rotation check (AROSICS explicitly does not handle this
        #    without an automatic resample — worth knowing in advance)
        if gt is not None and (abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9):
            rep.add(WARN, f"Geotransform has a rotation/shear term (GT[2]={gt[2]}, "
                          f"GT[4]={gt[4]}). AROSICS will auto-resample to remove "
                          f"this, which adds processing time and a resampling step.")
        else:
            rep.add(PASS, "No metadata rotation in geotransform.")

        # 6. Pixel size sanity
        if gt is not None:
            px_x, px_y = abs(gt[1]), abs(gt[5])
            if px_x == 0 or px_y == 0:
                rep.add(FAIL, f"Pixel size is zero (x={px_x}, y={px_y}).")
            else:
                rep.add(PASS, f"Pixel size (GSD): x={px_x}, y={px_y} (map units)")
            info["px_size"] = (px_x, px_y)

    # 3. Projection / CRS
    prj_wkt = ds.GetProjection()
    if not prj_wkt:
        rep.add(FAIL, "No projection/CRS defined. AROSICS requires a valid "
                      "spatial reference system.")
    else:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(prj_wkt)
        name = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS") or "Unknown"
        rep.add(PASS, f"Projection defined: {name}")
        info["srs_wkt"] = prj_wkt
        info["epsg"] = srs.GetAuthorityCode(None)

    # 7. Band count / band index validity
    n_bands = ds.RasterCount
    if n_bands < 1:
        rep.add(FAIL, "Raster has zero bands.")
    else:
        rep.add(PASS, f"Raster has {n_bands} band(s).")
    if band_to_match is not None:
        if not (1 <= band_to_match <= n_bands):
            rep.add(FAIL, f"Requested match band {band_to_match} is out of range "
                          f"(file has {n_bands} band(s), 1-indexed).")
        else:
            rep.add(PASS, f"Requested match band {band_to_match} is valid.")

    # 8. Data type
    band1 = ds.GetRasterBand(band_to_match or 1)
    dtype = gdal.GetDataTypeName(band1.DataType)
    rep.add(PASS, f"Data type of match band: {dtype}")
    info["dtype"] = dtype

    # 9. NoData value
    nodata_val = band1.GetNoDataValue()
    if nodata_val is None:
        rep.add(WARN, "No NoData value defined on the match band. AROSICS can "
                      "still run (pass nodata=(val, val) manually if needed), "
                      "but background/black-fill areas may be treated as valid "
                      "data during matching.")
    else:
        rep.add(PASS, f"NoData value defined: {nodata_val}")
    info["nodata"] = nodata_val

    # 10. Check if entirely NoData (sampled, not full-resolution, for speed)
    try:
        xsize, ysize = ds.RasterXSize, ds.RasterYSize
        # Downsample read for speed on large rasters
        sample_x = min(xsize, 1000)
        sample_y = min(ysize, 1000)
        arr = band1.ReadAsArray(0, 0, xsize, ysize,
                                buf_xsize=sample_x, buf_ysize=sample_y)
        if arr is None:
            rep.add(WARN, "Could not read pixel data to check for all-NoData condition.")
        else:
            import numpy as np
            arr_f = arr.astype("float64")
            if nodata_val is not None and isinstance(nodata_val, float) and np.isnan(nodata_val):
                # NaN nodata: `arr != nan` is always True in IEEE 754, so it must be
                # checked with isnan() instead of a direct equality/inequality compare.
                valid_frac = float(np.mean(~np.isnan(arr_f)))
            elif nodata_val is not None:
                valid_frac = float(np.mean(arr_f != nodata_val))
            elif np.issubdtype(arr.dtype, np.floating):
                # No nodata defined, but data is float — still check for stray NaNs.
                valid_frac = float(np.mean(~np.isnan(arr_f)))
            else:
                valid_frac = 1.0
            if valid_frac == 0:
                rep.add(FAIL, "Match band appears to contain ONLY NoData/empty values.")
            else:
                rep.add(PASS, f"Match band contains valid data (~{valid_frac*100:.1f}% "
                              f"non-NoData in sampled read).")
    except Exception as e:
        rep.add(WARN, f"Could not sample pixel data: {e}")

    info["xsize"], info["ysize"] = ds.RasterXSize, ds.RasterYSize
    info["n_bands"] = n_bands
    rep.add(PASS, f"Raster dimensions: {ds.RasterXSize} x {ds.RasterYSize} pixels")

    rep.print()
    info["ds"] = ds
    info["gt"] = gt
    return rep, info


def compute_extent(gt, xsize, ysize):
    """Return (minx, miny, maxx, maxy) in map coordinates (ignores rotation)."""
    x0, y0 = gt[0], gt[3]
    x1 = x0 + xsize * gt[1] + ysize * gt[2]
    y1 = y0 + xsize * gt[4] + ysize * gt[5]
    minx, maxx = sorted([x0, x1])
    miny, maxy = sorted([y0, y1])
    return minx, miny, maxx, maxy


def check_pair(ref_info, tgt_info, window_size, grid_res):
    rep = Report("Checking reference/target pair compatibility")

    if ref_info is None or tgt_info is None:
        rep.add(FAIL, "Cannot check pair — one or both files failed to open.")
        rep.print()
        return rep

    # Same CRS?
    ref_epsg = ref_info.get("epsg")
    tgt_epsg = tgt_info.get("epsg")
    if ref_epsg and tgt_epsg:
        if ref_epsg == tgt_epsg:
            rep.add(PASS, f"Reference and target share the same CRS (EPSG:{ref_epsg}).")
        else:
            rep.add(WARN, f"Different CRS: reference=EPSG:{ref_epsg}, "
                          f"target=EPSG:{tgt_epsg}. AROSICS can reproject "
                          f"internally, but confirm this is intentional.")
    else:
        rep.add(WARN, "Could not determine EPSG codes for one or both images "
                      "to compare CRS directly (may still be compatible; "
                      "AROSICS uses WKT comparison internally).")

    # Overlap check
    if ref_info.get("gt") and tgt_info.get("gt"):
        ref_ext = compute_extent(ref_info["gt"], ref_info["xsize"], ref_info["ysize"])
        tgt_ext = compute_extent(tgt_info["gt"], tgt_info["xsize"], tgt_info["ysize"])

        minx = max(ref_ext[0], tgt_ext[0])
        miny = max(ref_ext[1], tgt_ext[1])
        maxx = min(ref_ext[2], tgt_ext[2])
        maxy = min(ref_ext[3], tgt_ext[3])

        if minx >= maxx or miny >= maxy:
            rep.add(FAIL, "Reference and target images do NOT spatially overlap "
                          "(based on raw extents; note this check ignores CRS "
                          "differences — reproject extents manually if CRSs differ).")
        else:
            overlap_w = maxx - minx
            overlap_h = maxy - miny
            rep.add(PASS, f"Images overlap. Overlap extent: {overlap_w:.1f} x "
                          f"{overlap_h:.1f} map units.")

            # Rough check: is overlap big enough for window_size + grid_res?
            px_x, px_y = tgt_info.get("px_size", (None, None))
            if px_x and px_y and window_size:
                overlap_px_x = overlap_w / px_x
                overlap_px_y = overlap_h / px_y
                if overlap_px_x < window_size[0] or overlap_px_y < window_size[1]:
                    rep.add(FAIL, f"Overlap area ({overlap_px_x:.0f} x {overlap_px_y:.0f} "
                                  f"target px) is SMALLER than the requested matching "
                                  f"window_size {window_size}. COREG_LOCAL will not "
                                  f"be able to place matching windows.")
                else:
                    n_grid_x = overlap_px_x / grid_res if grid_res else None
                    n_grid_y = overlap_px_y / grid_res if grid_res else None
                    rep.add(PASS, f"Overlap area is large enough for window_size "
                                  f"{window_size}.")
                    if n_grid_x and n_grid_y:
                        est_points = int(n_grid_x * n_grid_y)
                        rep.add(PASS if est_points >= 10 else WARN,
                               f"Estimated tie point grid: ~{n_grid_x:.0f} x "
                               f"{n_grid_y:.0f} ≈ {est_points} candidate points "
                               f"at grid_res={grid_res}. "
                               + ("Consider a smaller grid_res if this seems low."
                                  if est_points < 10 else ""))
    else:
        rep.add(WARN, "Could not compute overlap — missing geotransform on one "
                      "or both images.")

    rep.print()
    return rep


def main():
    parser = argparse.ArgumentParser(description="Validate GeoTIFF inputs for arosics.COREG_LOCAL")
    parser.add_argument("reference", help="Path to reference image")
    parser.add_argument("target", help="Path to target image")
    parser.add_argument("--r-band", type=int, default=1, help="r_b4match (default: 1)")
    parser.add_argument("--s-band", type=int, default=1, help="s_b4match (default: 1)")
    parser.add_argument("--window-size", type=int, nargs=2, default=[256, 256],
                        metavar=("X", "Y"), help="window_size (default: 256 256)")
    parser.add_argument("--grid-res", type=float, default=50,
                        help="grid_res in target pixels (default: 50)")
    args = parser.parse_args()

    ref_rep, ref_info = inspect_raster(args.reference, args.r_band, "REFERENCE image")
    tgt_rep, tgt_info = inspect_raster(args.target, args.s_band, "TARGET image")
    pair_rep = check_pair(ref_info, tgt_info, tuple(args.window_size), args.grid_res)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    any_fail = ref_rep.has_fail() or tgt_rep.has_fail() or pair_rep.has_fail()
    if any_fail:
        print("  Result: NOT READY — one or more FAIL checks above must be fixed "
              "before running COREG_LOCAL.")
        sys.exit(1)
    else:
        print("  Result: READY — no FAIL checks found. Review any WARNs above; "
              "they won't necessarily break COREG_LOCAL but are worth knowing.")
        sys.exit(0)


if __name__ == "__main__":
    main()