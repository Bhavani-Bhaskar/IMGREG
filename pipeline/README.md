# AVHRR → MODIS bow-tie correction pipeline

Segregated, stage-by-stage version of the correction pipeline (a clean copy of the
code previously in `../bowtie_coreg/`, which is left untouched). One entry point,
`main.py`, runs everything and writes fresh per-granule results.

## Structure

```
pipeline/
  main.py                     orchestrator (preprocess → Stage 1 → Stage 2 → Stage 3)
  common/
    preprocessing.py          raw granule bands → common 0.01° grid arrays (inputs/)
  stage1_ncc/                 Stage 1 — binary-coastline NCC tie-points + TPS warp
    bowtie_pipeline.py          core: tie-point extraction + curation + TPS field
    bowtie_report.py            arrows / overlays / gallery / ncc_validation.txt
    visualize_overlay.py        red/cyan overlays vs MODIS
    assemble_report.py          one-page report_page.png
    zoom_swcoast.py             SW-coast zoom
  stage2_phasecorr/           Stage 2 — local phase-correlation residual refinement
    phasecorr_refine.py         on the Stage-1 output; sub-pixel + textured interior
  stage3_panoramic/           Stage 3 — parametric panoramic model (DEFAULT product)
    panoramic_model.py          scan-geometry hybrid; fills cloud/featureless regions
  ground_truth/               optional curated tie points per granule (GT-anchoring)
    <granule>_curated_ta.npy, <granule>_curated_off.npy
  results/<granule>/          generated output
    inputs/                     common-grid arrays + grid.npz (+ copied curated GT)
    output/                     corrected TIFs, shift fields, tie points
      avhrr_bowtie_corrected.tif          Stage 1
      avhrr_bowtie_corrected_stage2.tif   Stage 2
      avhrr_bowtie_panoramic.tif          Stage 3  ← FINAL / recommended product
      report/                             NCC validation + overlays
```

## Run (needs the `geo` conda env)

```bash
conda run -n geo python pipeline/main.py                    # default granule
conda run -n geo python pipeline/main.py --granule <name>   # one granule
conda run -n geo python pipeline/main.py --all              # every granule
```

Final product per granule: `results/<granule>/output/avhrr_bowtie_panoramic.tif`.

## Stages

1. **NCC tie-points + TPS** — matches sensor-invariant binary coastline masks locally
   (no global shift), curates the tie points, fits a trust-masked Thin-Plate-Spline.
2. **Phase-correlation refinement** — masked sub-pixel phase correlation on the
   Stage-1 output; residual capped, gated by the real-image NCC test.
3. **Parametric panoramic model (hybrid)** — recovers AVHRR scan geometry from the raw
   GCP nav, fits a smooth physical bow-tie model (GT-anchored when curated points
   exist), and blends: pure Stage-1 TPS where features exist, the physical model in the
   cloud/featureless regions the tie-point stages leave uncorrected, original beyond.
