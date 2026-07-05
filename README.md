### Conda Env Setup

conda create -n geo python=3.12
conda activate geo
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install numpy scipy pandas matplotlib opencv gdal
conda install rasterio scikit-image pyproj tqdm (optional)





# MODIS–AVHRR Local-Window Registration Workflow — Revised (v2)

**Reference:** MODIS 1 km (reference) vs. AVHRR MetOp-C L1B (target, distorted, single GeoTIFF)
**Method basis:** Foroosh et al. 2002 (subpixel phase correlation) + Scheffler et al. 2017 / AROSICS (local co-registration, validated production pipeline)

**What changed from v1:** Stage 5 has been split into a **Core** (directly grounded in your two source papers — build and test this first) and a **Hardening Backlog** (real techniques, but not required by either paper — add only if metrics from the Core tell you they're needed). Stage 10/11 has been reconciled with your stated intent to correct the GeoTIFF directly rather than exporting a separate lat/lon offset layer. Everything else is unchanged from v1 except minor sequencing notes.

---

## Stage 1 — Input Data
1. MODIS 1 km → reference
2. AVHRR MetOp-C L1B → target (distorted, GeoTIFF)

## Stage 2 — Preprocessing
3. Select one common band (thermal-IR/NIR) — single band
4. Check projection/CRS consistency
5. Reproject only if CRS differ
6. Pixel-grid equalization (resample higher-res → lower-res; identical grid)
7. Extract common overlap
8. Cloud / no-data mask (future work)
9. Normalize (optional — phase correlation is intensity-robust, so skip unless you see illumination-driven artifacts)

## Stage 3 — Grid Generation
10. Window size (256×256 default)
11. Dense regular tie-point grid over overlap
12. Drop grid points on no-data/cloud/edge

## Stage 4 — Per-Window Matching (loop)
13. Select AVHRR window
14. Locate corresponding MODIS window via AVHRR lat/lon
15. Cloud % check → reject if over threshold

*(You've confirmed Stages 1–4 are done. Stage 5 below is restructured — build the Core column first, ship it, and only pull from the Backlog if your own reliability/MSSIM numbers say you need to.)*

## Stage 5 — Subpixel Phase-Correlation Engine (inside loop)

### 5A. Core — build and validate this first
These steps reconstruct the AROSICS local co-registration engine almost exactly, plus Foroosh's closed-form subpixel derivation it's built on. This is not a simplification — it's a real, published, operational pipeline. Don't add anything from 5B until this is working and you've looked at its output.

| # | Step | Grounding |
|---|------|-----------|
| 16 | **Cross-power spectrum** — F_a = FFT(avhrr), F_m = FFT(modis); Q = (F_a·F_m*) / \|F_a·F_m*\| | Foroosh Eq. 3 |
| 17 | **Integer peak** — c = IFFT(Q); locate peak (x_m, y_m) | Foroosh; AROSICS §2.3.1 |
| 18 | **Integer-shift validation** — re-shift target window by (x_m, y_m), recompute cross-power spectrum; a true match collapses to peak 0/0; iterate ≤5×; reject if it never converges | AROSICS §2.3.2, Fig. 3 — this is their published validation flowchart, not an ad hoc addition |
| 19 | **Subpixel bootstrap** — from the refreshed spectrum's peak + immediate neighbors: Δx = v₍₁,₀₎ / (v₍₁,₀₎ ± v₍₀,₀₎), same form for Δy; choose the root in [−1,1] | AROSICS Eqs. 1–2, derived from Foroosh's polyphase result |
| 20 | **Reliability (peak sharpness)** — R = 100 − 100·(µ_remain + 3σ_remain)/µ_peak over a 3×3 window at the peak vs. the rest of the spectrum; apply threshold (AROSICS default: reject R < 30%) | AROSICS §2.3.2, Eq. 6 — use their published 30% default as your starting threshold, not an arbitrary one |
| 21 | **MSSIM before vs. after** applying (dx, dy) | AROSICS validation method 4 — chosen specifically for sensitivity to small displacements; don't substitute a coarser similarity metric here |

### 5B. Hardening backlog — add only on evidence, not by default
None of these appear in either source paper's actual pipeline. They're legitimate techniques from the wider phase-correlation literature, but each adds tunable parameters and implementation cost. Implement one at a time, and only after inspecting the Core's R and MSSIM distributions shows a specific failure mode it would fix.

| # | Step | When to actually add it |
|---|------|--------------------------|
| 22 | **Apodize** — multiply both windows by a 2-D Hann window before the FFT | Add early if you visually see cross-shaped leakage artifacts in raw AVHRR windows — likely given your stated distortion problem, and it's a one-line, low-risk change. Neither paper uses it explicitly, but it's a standard fix (cf. Stone et al.'s Blackman windowing) for exactly the boundary-discontinuity problem non-periodic real scenes create. |
| 23 | **Spectral weighting** — Gaussian band-pass mask on Q | Only add if the Core's reliability scores are consistently low on windows that look visually well-matched — i.e., noise is the diagnosed problem, not something else. Requires hand-tuning cutoff frequencies; risk of stripping the high-frequency phase content subpixel accuracy depends on. |
| 24 | **Robust refinement** — upsampled-DFT (Guizar-Sicairos) around the peak, or phase-slope RANSAC on Q | Only add if 21a's 3-sample bootstrap is visibly unstable (noisy dx/dy on windows that pass the reliability/MSSIM gate). Real implementation cost — don't build this speculatively. |
| 25 | **Coarse-to-fine (pyramid)** | Only add if you're seeing large integer shifts that the 5-iteration re-shift loop (step 18) fails to converge on. It may be redundant with step 18 rather than complementary — check before building. |

**Practical build order for Stage 5:** 16 → 17 → 18 → 19 → 20 → 21, test on real MODIS/AVHRR window pairs, inspect R and MSSIM outputs, *then* decide whether 22–25 are needed and in what order.

## Stage 6 — Tie-Point Acceptance (loop)
26. Keep if reliability ≥ thr AND MSSIM increases AND |shift| ≤ max
27. Store tie point + (dx,dy) + reliability
28. Next window

## Stage 7 — Outlier Removal
29. RANSAC w/ preliminary affine
30. Residuals
31. Residual-threshold filter
32. Final inliers

## Stage 8 — Transformation Modeling
33. Affine (LS) on inliers
34. If residual high → 2nd-order polynomial (weighted LS)
35. (optional) LSM refinement

## Stage 9 — Tie-Point Statistics
36. Valid count
37. Density

## Stage 10 — Geometric Correction ⚠️ revised to match your stated intent
Your note says you're **not** exporting corrected lat/lon and instead want the corrected GeoTIFF directly. That's **Option B**, not Option A from v1 — flagging this explicitly since it changes what Stage 8's model is used for (warping pixels, not deriving an offset surface) and it does mean radiance values get resampled, which the original doc's recommendation was written to avoid.

**Option B (your stated approach): warp the AVHRR raster**
38B. Apply the Stage 8 affine/polynomial model to resample the AVHRR image onto the MODIS grid
39B. Choose resampling kernel (nearest for categorical/QA bands; bilinear or cubic for continuous radiance/reflectance — nearest will introduce blocky artifacts on a thermal-IR/NIR band)
40B. Output corrected GeoTIFF with updated geotransform/tags

**Trade-off worth being deliberate about before you commit:** Option B modifies the original AVHRR pixel values through resampling — any downstream radiometric analysis (temperature retrievals, band ratios, time-series comparisons) will be working with interpolated values, not sensor-native ones. Option A (correct lat/lon only, leave radiances untouched) avoids that but requires downstream tools that can consume a per-pixel lat/lon offset instead of a standard grid. If your downstream use is purely visual/qualitative or your tools require a regular grid anyway, B is the practical choice and what you've specified. If you need radiometrically pristine AVHRR values later, keep Option A's lat/lon export as a fallback output alongside B — it's cheap to generate from the same Stage 8 model.

## Stage 11 — Output
Corrected AVHRR GeoTIFF, resampled onto the MODIS grid (per Stage 10 Option B), with updated geolocation tags.

## Stage 12 — Accuracy Assessment
41. RMSE before
42. RMSE after on independent check points
43. CE90
44. mean/σ
45. tie-point count/density/mean reliability
46. processing time → Final Report