"""
Visualize the affine-transformed AVHRR overlapping the MODIS
reference, using a red/cyan composite (a standard registration
QC technique: target in red, reference in green+blue: perfect
alignment reads as neutral grey, misalignment shows as colored
fringing/ghosting).

Produces:
1. A whole-scene overview (downsampled) - before (original
   AVHRR vs MODIS) and after (affine-transformed AVHRR vs MODIS).
2. Full-resolution zoomed crops around the tie points with the
   largest shifts, where the correction effect is actually
   visible at pixel scale.

Output
------
gallery/affine_overlap.html
"""

import os
import base64
import numpy as np
import pandas as pd
import cv2
from osgeo import gdal

gdal.UseExceptions()

ROOT = os.path.join(os.path.dirname(__file__), "..")

AVHRR_FILE = os.path.join(ROOT, "2_outputs", "05_avhrr_float32.tif")
MODIS_FILE = os.path.join(ROOT, "2_outputs", "05_modis_float32.tif")
CUBIC_FILE = os.path.join(ROOT, "stage_v2", "avhrr_affine_cubic.tif")
BILINEAR_FILE = os.path.join(ROOT, "stage_v2", "avhrr_affine_bilinear.tif")
TIE_POINTS_CSV = os.path.join(ROOT, "stage_v2", "tie_points.csv")

OUTPUT_HTML = os.path.join(ROOT, "gallery", "affine_overlap.html")

OVERVIEW_MAX_DIM = 900
CROP_SIZE = 260
CROP_DISPLAY = 320


def normalize_to_uint8(window):

    valid = np.isfinite(window)

    if valid.sum() < 2:
        return np.zeros(window.shape, dtype=np.uint8)

    lo, hi = float(window[valid].min()), float(window[valid].max())

    if (hi - lo) < 1e-12:
        return np.zeros(window.shape, dtype=np.uint8)

    normalized = np.clip((window - lo) / (hi - lo), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0)

    return (normalized * 255).astype(np.uint8)


def red_cyan_composite(target_gray_u8, reference_gray_u8):
    """
    target -> red channel, reference -> green+blue (cyan).
    Aligned content reads neutral grey/white; misaligned content
    shows red/cyan fringing.
    """

    h, w = reference_gray_u8.shape

    composite = np.zeros((h, w, 3), dtype=np.uint8)

    composite[:, :, 2] = target_gray_u8      # R (cv2 is BGR)
    composite[:, :, 1] = reference_gray_u8   # G
    composite[:, :, 0] = reference_gray_u8   # B

    return composite


def to_data_uri_bgr(image_bgr, max_dim=None):

    if max_dim is not None:
        h, w = image_bgr.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            image_bgr = cv2.resize(
                image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])

    if not ok:
        raise RuntimeError("JPEG encode failed.")

    encoded = base64.b64encode(buf).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def crop(array, cx, cy, size):

    r0, r1 = int(cy - size / 2), int(cy + size / 2)
    c0, c1 = int(cx - size / 2), int(cx + size / 2)

    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, array.shape[0]), min(c1, array.shape[1])

    return array[r0:r1, c0:c1]


def main():

    avhrr_ds = gdal.Open(AVHRR_FILE)
    modis_ds = gdal.Open(MODIS_FILE)
    cubic_ds = gdal.Open(CUBIC_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    cubic = cubic_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    print("Building whole-scene overview...")

    modis_u8 = normalize_to_uint8(modis)
    avhrr_u8 = normalize_to_uint8(avhrr)
    cubic_u8 = normalize_to_uint8(cubic)

    before_overview = red_cyan_composite(avhrr_u8, modis_u8)
    after_overview = red_cyan_composite(cubic_u8, modis_u8)

    before_uri = to_data_uri_bgr(before_overview, max_dim=OVERVIEW_MAX_DIM)
    after_uri = to_data_uri_bgr(after_overview, max_dim=OVERVIEW_MAX_DIM)

    print("Building zoomed crops at tie points with the largest shifts...")

    df = pd.read_csv(TIE_POINTS_CSV)
    valid = df[df["OUTLIER"] == "False"].copy()
    valid["abs_shift"] = valid["ABS_SHIFT"].astype(float)
    top = valid.sort_values("abs_shift", ascending=False).head(6)

    crop_cards = []

    for _, row in top.iterrows():

        cx, cy = float(row.X_IM), float(row.Y_IM)

        avhrr_crop = crop(avhrr, cx, cy, CROP_SIZE)
        modis_crop = crop(modis, cx, cy, CROP_SIZE)
        cubic_crop = crop(cubic, cx, cy, CROP_SIZE)

        before_composite = red_cyan_composite(
            normalize_to_uint8(avhrr_crop), normalize_to_uint8(modis_crop)
        )
        after_composite = red_cyan_composite(
            normalize_to_uint8(cubic_crop), normalize_to_uint8(modis_crop)
        )

        before_uri_c = to_data_uri_bgr(before_composite, max_dim=CROP_DISPLAY)
        after_uri_c = to_data_uri_bgr(after_composite, max_dim=CROP_DISPLAY)

        crop_cards.append(f"""
        <article class="card">
          <header class="card-head">
            <span class="win-id">Point {int(row.POINT_ID):05d}</span>
            <span class="chip">|shift| = {row.abs_shift:.1f}px</span>
          </header>
          <div class="panels">
            <figure>
              <img src="{before_uri_c}" alt="before, point {int(row.POINT_ID)}" width="{CROP_DISPLAY}" height="{CROP_DISPLAY}" loading="lazy">
              <figcaption>Before (red=AVHRR, cyan=MODIS)</figcaption>
            </figure>
            <figure>
              <img src="{after_uri_c}" alt="after, point {int(row.POINT_ID)}" width="{CROP_DISPLAY}" height="{CROP_DISPLAY}" loading="lazy">
              <figcaption>After affine (red=AVHRR, cyan=MODIS)</figcaption>
            </figure>
          </div>
          <dl class="meta">
            <div><dt>dx&nbsp;/&nbsp;dy</dt><dd>{row.X_SHIFT_PX:+.2f}px&nbsp;/&nbsp;{row.Y_SHIFT_PX:+.2f}px</dd></div>
            <div><dt>reliability&nbsp;R</dt><dd>{row.RELIABILITY:.1f}</dd></div>
          </dl>
        </article>
        """)

    crop_html = "\n".join(crop_cards)

    html = TEMPLATE.format(
        before_overview=before_uri,
        after_overview=after_uri,
        crops=crop_html,
    )

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_HTML) / (1024 * 1024)

    print(f"\nSaved: {OUTPUT_HTML} ({size_mb:.2f} MB)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AVHRR / MODIS Affine Overlap (red/cyan)</title>
<style>
  :root {{
    --bg: #f4f6fb; --surface: #ffffff; --text: #131826; --text-muted: #5b667a;
    --border: #dde3ee; --accent: #2f6fed; --good: #1f9d6b; --good-bg: #e3f7ee;
    --shadow: 0 1px 2px rgba(20,30,60,0.06), 0 8px 24px rgba(20,30,60,0.05);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0b1220; --surface:#141c2e; --text:#e6ebf5; --text-muted:#8b96ad;
      --border:#232d43; --accent:#5b9dff; --good:#3ecf8e; --good-bg: rgba(62,207,142,0.14);
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35); }}
  }}
  :root[data-theme="dark"] {{ --bg:#0b1220; --surface:#141c2e; --text:#e6ebf5; --text-muted:#8b96ad;
    --border:#232d43; --accent:#5b9dff; --good:#3ecf8e; --good-bg: rgba(62,207,142,0.14); }}
  :root[data-theme="light"] {{ --bg:#f4f6fb; --surface:#ffffff; --text:#131826; --text-muted:#5b667a;
    --border:#dde3ee; --accent:#2f6fed; --good:#1f9d6b; --good-bg:#e3f7ee; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font-family: ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif; line-height:1.45; }}
  header.page {{ padding: 1.25rem 1.5rem 1rem; border-bottom: 1px solid var(--border); }}
  h1 {{ margin: 0 0 0.15rem; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; text-wrap: balance; }}
  h2 {{ font-size: 1.05rem; margin: 1.6rem 1.5rem 0.6rem; }}
  .subtitle {{ margin: 0; color: var(--text-muted); font-size: 0.9rem; max-width: 68ch; }}
  .overview-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 0 1.5rem; }}
  .overview-row figure {{ margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.6rem; box-shadow: var(--shadow); }}
  .overview-row img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  .overview-row figcaption {{ margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-muted); text-align: center; }}
  main {{ padding: 0.5rem 1.5rem 3rem; display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent);
    border-radius: 10px; box-shadow: var(--shadow); padding: 0.85rem; }}
  .card-head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem; }}
  .win-id {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace; font-size: 0.85rem;
    font-weight: 600; color: var(--text-muted); }}
  .chip {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; padding: 0.2rem 0.55rem; border-radius: 999px;
    background: var(--good-bg); color: var(--good); }}
  .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; }}
  .panels figure {{ margin: 0; }}
  .panels img {{ display: block; width: 100%; height: auto; border-radius: 6px; background: #000; }}
  .panels figcaption {{ margin-top: 0.3rem; font-size: 0.66rem; text-align: center; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.03em; }}
  dl.meta {{ margin: 0.7rem 0 0; display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; font-size: 0.78rem; }}
  dl.meta div {{ display: flex; gap: 0.35rem; }}
  dl.meta dt {{ color: var(--text-muted); }}
  dl.meta dd {{ margin: 0; font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
    font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>

<header class="page">
  <h1>AVHRR / MODIS Affine Overlap</h1>
  <p class="subtitle">Red = AVHRR, cyan = MODIS. Perfectly aligned content reads as neutral grey; misalignment shows as red/cyan fringing. The affine model was fit by least squares from the 18 valid v2 tie points and applied with cubic resampling.</p>
</header>

<h2>Whole scene</h2>
<div class="overview-row">
  <figure>
    <img src="{before_overview}" alt="before affine, whole scene">
    <figcaption>Before &mdash; original AVHRR vs. MODIS</figcaption>
  </figure>
  <figure>
    <img src="{after_overview}" alt="after affine, whole scene">
    <figcaption>After &mdash; affine-transformed AVHRR vs. MODIS</figcaption>
  </figure>
</div>

<h2>Zoomed crops at the largest tie-point shifts</h2>
<main>
{crops}
</main>

</body>
</html>
"""


if __name__ == "__main__":
    main()
