"""
run_all_granules.py
-------------------

Run the full data-driven pipeline on every AVHRR granule in Data/psdd_metop/metop
and produce a detailed self-referential report for each, in its own folder:

    bowtie_coreg/runs/<granule>/inputs/   (common-grid arrays from raw bands)
    bowtie_coreg/runs/<granule>/output/   (corrected tif, tie points, shift field)
    bowtie_coreg/runs/<granule>/output/report/   (arrows, overlays, gallery, NCC validation)

The pipeline is fully data-driven (no hardcoded regions), so each granule's
clouds / coasts / swath geometry are handled from its own data.

Run:  conda run -n geo python bowtie_coreg/run_all_granules.py
"""

import os
import sys
import glob
import subprocess

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
RAW = os.path.join(_ROOT, "Data", "psdd_metop", "metop")
RUNS = os.path.join(_BASE, "runs")
PY = sys.executable   # inherits conda env (run this script via `conda run -n geo python`)


def run(cmd, **kw):
    print("  $", " ".join(os.path.basename(x) if x.endswith(".py") else x for x in cmd[1:]))
    r = subprocess.run(cmd, cwd=_BASE, **kw)
    return r.returncode


def main():
    granules = sorted(os.path.basename(f)[:-len("_geo_b2.tif")]
                      for f in glob.glob(os.path.join(RAW, "*_geo_b2.tif")))
    print(f"Found {len(granules)} granules:", *granules, sep="\n  ")

    summary = []
    for g in granules:
        print("\n" + "=" * 70)
        print("GRANULE:", g)
        print("=" * 70)
        rundir = os.path.join(RUNS, g)
        inp = os.path.join(rundir, "inputs")
        out = os.path.join(rundir, "output")
        os.makedirs(inp, exist_ok=True)
        os.makedirs(out, exist_ok=True)

        rc = run([PY, "preprocessing.py", "--granule", g, "--out", inp])
        if rc != 0:
            summary.append((g, "PREPROCESS FAILED")); continue
        rc = run([PY, "bowtie_pipeline.py", "--inputs", inp, "--output", out])
        if rc != 0:
            summary.append((g, "PIPELINE FAILED")); continue
        run([PY, "bowtie_report.py", "--inputs", inp, "--output", out])
        run([PY, "visualize_overlay.py",
             "--original", os.path.join(inp, "a_arr.npy"),
             "--modis", os.path.join(inp, "m_arr.npy"),
             "--corrected", os.path.join(out, "avhrr_bowtie_corrected.tif"),
             "--out", os.path.join(out, "report")])
        run([PY, "assemble_report.py", "--inputs", inp, "--output", out])

        # pull the headline NCC line from the validation file
        vf = os.path.join(out, "report", "ncc_validation.txt")
        head = ""
        if os.path.exists(vf):
            for ln in open(vf):
                if ln.startswith("overall mean NCC"):
                    head = ln.strip(); break
        summary.append((g, head or "done"))

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for g, s in summary:
        print(f"{g}\n    {s}")
    print(f"\nPer-granule reports in: {RUNS}/<granule>/output/report/")


if __name__ == "__main__":
    main()
