# bowtie_coreg — Automatic AVHRR → MODIS geolocation correction

Corrects the **bow-tie / panoramic geolocation distortion** of a MetOp-C AVHRR
scene against a MODIS 1 km reference, fully automatically, by extracting valid
tie points and warping with a Thin-Plate Spline. The error is *not* a global
shift (mean dx ≈ 0) — it is a smooth across-track distortion where the swath
width is preserved at nadir and shrinks toward the top and bottom of the scene.

Run everything with the **`geo`** conda env (has scikit-image, GDAL, OpenCV,
scipy). Scripts resolve their paths relative to this folder, so they work from
any working directory.

## Run from scratch → results (5 commands)

```bash
cd /home/bhaskar/Documents/ImageReg
conda run -n geo python bowtie_coreg/preprocessing.py     # 0. raw bands -> inputs/  (run first, once)
conda run -n geo python bowtie_coreg/bowtie_pipeline.py   # 1. extract tie points + TPS warp -> corrected TIFF
conda run -n geo python bowtie_coreg/bowtie_report.py     # 2. per-tie-point validity report + arrows + gallery
conda run -n geo python bowtie_coreg/visualize_overlay.py # 3. AVHRR-on-MODIS overlays
conda run -n geo python bowtie_coreg/zoom_swcoast.py      # 4. Karnataka/Kerala coast zoom
conda run -n geo python bowtie_coreg/assemble_report.py   # 5. one-page report figure
```

Step 0 (`preprocessing.py`) reads the raw granule bands and writes the 7 input
arrays to `inputs/`; run it once. Steps 1–5 then produce the corrected AVHRR and
all report figures in `output/`. Expect ~105 matched tie points (+28 protection
anchors), all NCC-validated.

---

## Inputs

**Raw inputs** — the only files the pipeline truly needs (in `../Data/psdd_metop/metop/`):

| Raw file | Band | Role |
|---|---|---|
| `hrpt_M03_20250506_0420_33701_geo_b2.tif` | AVHRR visible ch2 | target imagery (what gets warped) |
| `hrpt_M03_20250506_0420_33701_geo_b3a.tif` | AVHRR SWIR ch3a | land/water masking (water dark in SWIR) |
| `hrpt_M03_20250506_0420_33701_geo_b4.tif` | AVHRR thermal ch4 | cloud detection (clouds cold) |
| `modis_1km.tif` | MODIS 1 km | geolocation reference |

`preprocessing.py` turns those into the **common-grid arrays** in `inputs/`
(one 0.01° grid, 3181 × 5086, origin 64.0889 E / 42.0742 N, snapped to the MODIS
pixel origin): `a_arr` (visible), `s_arr` (SWIR), `b4_arr` (thermal), `m_arr`
(MODIS), `s_land` / `m_land` (Otsu land masks), `cloud_mask` (thermal), plus
`grid.npz` (geotransform). These are what steps 1–5 read.

`inputs/curated_ta.npy` + `curated_off.npy` are 66 hand-picked tie points kept
**only for validation** (the pipeline runs fine without them). Regenerating the
arrays from raw reproduces the manual Stage 1–3 outputs to within rounding
(imagery correlation 1.00000, masks ≥ 99.98% identical), so results match.

## Outputs (`output/`)

| Output | What it is |
|---|---|
| `output/avhrr_bowtie_corrected.tif` | **the corrected AVHRR** (main deliverable), on the MODIS grid |
| `output/auto_ta.npy` / `auto_off.npy` / `auto_sc.npy` | tie-point locations, shifts, scores (`sc < 0` = Gangetic protection anchor) |
| `output/shift_field_dx.tif` / `dy.tif` | the fitted per-pixel displacement field |
| `output/report/report_page.png` | **one-page summary** (before/after + arrows + SW zoom + stats) |
| `output/report/tiepoint_vectors.png` | tie-point displacement arrows (bow-tie) |
| `output/report/overlay_*.png` | AVHRR-on-MODIS overlays (composite, blend, red/cyan) |
| `output/report/swcoast_*.png` | Karnataka/Kerala coast before/after zoom |
| `output/report/tiepoint_gallery.png` | per-point MODIS \| before \| after crops |
| `output/report/validity_summary.txt` | per-point NCC before→after table |

## Result (current)

105 matched tie points (all NCC-validated) + 28 Gangetic protection anchors.
Alignment (NCC vs MODIS, before → after): Himalaya 0.008 → 0.106, Central
Deccan −0.004 → 0.063, coasts 0.030 → 0.093, Gangetic plain preserved
(0.052 → 0.065). No global shift (mean dx ≈ 0) confirms a pure bow-tie error.

---

## Workflow / flow chart

```
 RAW BANDS           ┌──────────────────────────────────────────────┐
 Data/psdd_metop/    │  AVHRR geo_b2 (visible) · geo_b3a (SWIR)      │
   metop/            │  geo_b4 (thermal) · modis_1km.tif            │
                     └───────────────────────┬──────────────────────┘
                                             │  preprocessing.py
                                             │  (grid align 0.01° + cloud + land masks)
                                             ▼
                     ┌──────────────────────────────────────────────┐
 inputs/  (common    │  a_arr · s_arr · b4_arr · m_arr               │
 0.01° grid)         │  s_land · m_land · cloud_mask · grid.npz      │
                     └───────────────────────┬──────────────────────┘
                                             │
                                             ▼
  ╔══════════════════ bowtie_pipeline.py :: extract_tie_points ══════════════════╗
  ║                                                                              ║
  ║  1. COASTLINE MATCH        NCC tile-match binary land/water masks (no        ║
  ║     (dense_tile_match)      global shift). Multi-scale, distinctiveness gate. ║
  ║                                     │  coastline tie points                  ║
  ║  2. WEST INFILL            permissive pass west of nadir (Gujarat/Konkan);    ║
  ║     (west_infill_match)     relaxed cloud gate; kept honest by corroboration. ║
  ║                                     │                                        ║
  ║  3. SEED-AND-GROW         fit smooth bow-tie model → predict shift → search   ║
  ║     (grow_matches)          a small window around it (fills gaps).            ║
  ║                                     │                                        ║
  ║  4. STRAIGHT-COAST DX     Karnataka/Kerala straight coast: wide-x / narrow-y  ║
  ║     (straight_coast_match)  search → reliable across-shore dx, dy from model. ║
  ║                                     │                                        ║
  ║  5. INTERIOR / NORTH      cross-sensor gradient-STRUCTURE match (Himalayan    ║
  ║     (interior_structure_)   ridges/snow lines); wide search + NCC gate.       ║
  ║                                     │  all candidate tie points               ║
  ║                                     ▼                                        ║
  ║  6. CURATION CHAIN (reject bad/duplicate/inconsistent matches):              ║
  ║        dedup → neighbour+LOO → median-consistency → NCC-validity gate         ║
  ║                                     │                                        ║
  ║  7. GANGETIC PROTECTION   exclude flat plain from matching + add zero-shift   ║
  ║                            anchors so its already-good original is preserved. ║
  ║                                     │  final tie points (+ anchors)           ║
  ╚═════════════════════════════════════┼════════════════════════════════════════╝
                                        ▼
                    8. TPS WARP  (fit_tps_field → warp_with_field)
                                        │
                                        ▼
                     output/avhrr_bowtie_corrected.tif  ◄── corrected AVHRR
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  bowtie_report.py             visualize_overlay.py              assemble_report.py
  per-point NCC validity        AVHRR-on-MODIS overlays          one-page report figure
  + arrows + gallery            (composite/blend/red-cyan)       (+ zoom_swcoast.py)
                                        │
                                        ▼
                              VALIDATION (in bowtie_pipeline)
                     vs the 66 hand-picked ground-truth tie points
```

### Why each step exists (short)

- **No global pre-shift** — the error is a local bow-tie, not a translation
  (a raw global phase-correlation gives ~0). Confirmed against the manual
  ground truth (mean dx ≈ 0, sign flips across the nadir column).
- **Binary coastline matching** — sensor- and time-invariant, unlike raw
  cross-sensor intensity (which fails).
- **Curation chain** — automatic matching produces some self-consistent *wrong*
  clusters; each filter removes a different failure mode (duplicates,
  isolated/over-shooting points, matches that don't actually improve NCC).
- **Gangetic protection** — flat farmland has no matchable texture and was
  already aligned, so it is pinned rather than warped.

## Notes / limitations

- Interior *between* the coasts is corrected by the smooth field without needing
  tie points there.
- **Self-contained**: the pipeline depends only on the raw bands in
  `../Data/psdd_metop/metop/`. It no longer reads the `../manual_registration/`
  folder — `preprocessing.py` rebuilds the common-grid inputs from raw, and the
  66 ground-truth points were copied into `inputs/`.
- If you re-run `preprocessing.py`, everything downstream reproduces the same
  ~105 tie points and corrected output (verified: imagery arrays reproduce the
  manual Stage 1–3 to correlation 1.00000).
- Superseded earlier approaches (still at project root, not used here):
  `arosics_pipeline.py`, `phasecorr_pipeline.py`, `phasecorr_compare.py`.
