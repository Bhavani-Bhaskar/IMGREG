"""
main.py  --  AVHRR -> MODIS bow-tie geolocation correction pipeline (orchestrator)
==================================================================================

Runs the full data-driven correction end-to-end and writes every artifact into a
fresh per-granule results folder. The code is segregated by stage:

    pipeline/
      main.py                 <- this orchestrator
      common/preprocessing.py   raw granule bands -> common 0.01deg grid arrays
      stage1_ncc/               Stage 1: binary-coastline NCC tie-points + TPS warp
        bowtie_pipeline.py        (core)  + bowtie_report / visualize_overlay /
                                            assemble_report / zoom_swcoast (reports)
      stage2_phasecorr/         Stage 2: local phase-correlation residual refinement
        phasecorr_refine.py
      stage3_panoramic/         Stage 3: parametric panoramic model (hybrid) -- the
        panoramic_model.py                DEFAULT / recommended final product
      ground_truth/             optional curated tie points per granule (GT-anchoring)
      results/<granule>/        <- generated here (inputs/, output/, output/report/)

Pipeline per granule:
    preprocess -> Stage 1 (+reports) -> Stage 2 -> Stage 3
Final product:  results/<granule>/output/avhrr_bowtie_panoramic.tif

Run with the `geo` conda env:
    conda run -n geo python pipeline/main.py                    # default granule
    conda run -n geo python pipeline/main.py --granule <name>   # one granule
    conda run -n geo python pipeline/main.py --all              # every granule
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess

_BASE = os.path.dirname(os.path.abspath(__file__))          # pipeline/


def _find_root(d):
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "Data", "psdd_metop")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(_BASE)


_ROOT = _find_root(_BASE)
RAW = os.path.join(_ROOT, "Data", "psdd_metop", "metop")

COMMON = os.path.join(_BASE, "common")
S1 = os.path.join(_BASE, "stage1_ncc")
S2 = os.path.join(_BASE, "stage2_phasecorr")
S3 = os.path.join(_BASE, "stage3_panoramic")
GT = os.path.join(_BASE, "ground_truth")
RESULTS = os.path.join(_BASE, "results")

DEFAULT_GRANULE = "hrpt_M03_20250506_0420_33701"
PY = sys.executable   # inherits the conda env when launched via `conda run -n geo python`


def run(cmd, label):
    print(f"\n>>> {label}")
    print("    $", os.path.basename(cmd[1]), *cmd[2:])
    r = subprocess.run(cmd, cwd=_BASE)
    if r.returncode != 0:
        raise RuntimeError(f"{label} FAILED (exit {r.returncode})")


def _grep(path, startswith, last=False):
    """Return the first (or last) line starting with `startswith`, stripped."""
    if not os.path.exists(path):
        return ""
    hit = ""
    for ln in open(path):
        if ln.strip().startswith(startswith):
            hit = ln.strip()
            if not last:
                break
    return hit


def process(granule, results_dir):
    inp = os.path.join(results_dir, granule, "inputs")
    out = os.path.join(results_dir, granule, "output")
    rep = os.path.join(out, "report")
    for d in (inp, out, rep):
        os.makedirs(d, exist_ok=True)

    print("\n" + "#" * 70)
    print("# GRANULE:", granule)
    print("#" * 70)

    # ---- preprocessing: raw bands -> common-grid input arrays ----
    run([PY, os.path.join(COMMON, "preprocessing.py"),
         "--granule", granule, "--out", inp], "preprocess")

    # optional curated ground truth for this granule (Stage-3 GT-anchoring)
    for suff in ("ta", "off"):
        src = os.path.join(GT, f"{granule}_curated_{suff}.npy")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(inp, f"curated_{suff}.npy"))
    if os.path.exists(os.path.join(inp, "curated_ta.npy")):
        print("    (curated ground truth found -> Stage-3 will GT-anchor)")

    # ---- Stage 1: NCC coastline tie-points + TPS warp ----
    run([PY, os.path.join(S1, "bowtie_pipeline.py"),
         "--inputs", inp, "--output", out], "Stage 1  NCC tie-points + TPS")
    run([PY, os.path.join(S1, "bowtie_report.py"),
         "--inputs", inp, "--output", out], "Stage 1  report")
    run([PY, os.path.join(S1, "visualize_overlay.py"),
         "--original", os.path.join(inp, "a_arr.npy"),
         "--modis", os.path.join(inp, "m_arr.npy"),
         "--corrected", os.path.join(out, "avhrr_bowtie_corrected.tif"),
         "--out", rep], "Stage 1  overlay")
    run([PY, os.path.join(S1, "assemble_report.py"),
         "--inputs", inp, "--output", out], "Stage 1  assemble report")

    # ---- Stage 2: local phase-correlation refinement ----
    run([PY, os.path.join(S2, "phasecorr_refine.py"),
         "--inputs", inp, "--output", out], "Stage 2  phase-corr refinement")

    # ---- Stage 3: parametric panoramic model (hybrid) = final product ----
    run([PY, os.path.join(S3, "panoramic_model.py"),
         "--granule", granule, "--inputs", inp, "--output", out],
        "Stage 3  panoramic model (final)")

    s1 = _grep(os.path.join(rep, "ncc_validation.txt"), "overall mean NCC")
    s2 = _grep(os.path.join(rep, "ncc_validation_stage2.txt"), "DELTA")
    s3 = _grep(os.path.join(rep, "ncc_validation_panoramic.txt"), "overall", last=True)
    return {"granule": granule, "stage1": s1, "stage2": s2, "stage3_hybrid": s3,
            "final": os.path.join(out, "avhrr_bowtie_panoramic.tif")}


def main():
    p = argparse.ArgumentParser(description="3-stage AVHRR->MODIS bow-tie correction")
    p.add_argument("--granule", default=DEFAULT_GRANULE,
                   help="granule prefix in Data/psdd_metop/metop/ (default: %(default)s)")
    p.add_argument("--all", action="store_true",
                   help="process every *_geo_b2.tif granule found in the raw folder")
    p.add_argument("--results", default=RESULTS, help="output results root")
    args = p.parse_args()

    if args.all:
        granules = sorted(os.path.basename(f)[:-len("_geo_b2.tif")]
                          for f in glob.glob(os.path.join(RAW, "*_geo_b2.tif")))
    else:
        granules = [args.granule]
    print(f"Pipeline root: {_BASE}\nRaw data:      {RAW}\n"
          f"Results ->     {args.results}\nGranules:      {', '.join(granules)}")

    summ = []
    for g in granules:
        try:
            summ.append(process(g, args.results))
        except RuntimeError as e:
            summ.append({"granule": g, "error": str(e)})

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY  (final product = output/avhrr_bowtie_panoramic.tif)")
    print("=" * 70)
    for s in summ:
        print(f"\n{s['granule']}")
        if "error" in s:
            print(f"    !! {s['error']}")
            continue
        print(f"    Stage 1: {s['stage1']}")
        print(f"    Stage 2: {s['stage2']}")
        print(f"    Stage 3: {s['stage3_hybrid']}  (hybrid, cloud-masked gradient NCC vs MODIS)")
        print(f"    final -> {s['final']}")


if __name__ == "__main__":
    main()
