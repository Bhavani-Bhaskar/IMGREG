"""
Display the one entire locally-transformed AVHRR image, the
MODIS reference image, and their overlap (red/cyan composite:
AVHRR=red, MODIS=cyan; aligned content reads neutral grey,
misalignment shows as colored fringing).

Uses the smooth, spatially-varying (per-pixel) correction from
apply_local_transform.py - built from all 625 verified tie
points - not a single global affine.

Output
------
gallery/full_scene_overlap.html
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
TRANSFORMED_FILE = os.path.join(ROOT, "stage_v2", "avhrr_local_tiled.tif")
TIE_POINTS_CSV = os.path.join(ROOT, "stage_v2", "tie_points_verified.csv")

OUTPUT_HTML = os.path.join(ROOT, "gallery", "full_scene_overlap.html")

DISPLAY_MAX_DIM = 1500
CROP_SIZE = 220
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

    composite = np.zeros((*reference_gray_u8.shape, 3), dtype=np.uint8)

    composite[:, :, 2] = target_gray_u8      # R (cv2 is BGR)
    composite[:, :, 1] = reference_gray_u8   # G
    composite[:, :, 0] = reference_gray_u8   # B

    return composite


def to_data_uri_gray(image_u8, max_dim=DISPLAY_MAX_DIM):

    h, w = image_u8.shape[:2]
    scale = max_dim / max(h, w)

    if scale < 1.0:
        image_u8 = cv2.resize(image_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", image_u8, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if not ok:
        raise RuntimeError("JPEG encode failed.")

    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"


def to_data_uri_bgr(image_bgr, max_dim=DISPLAY_MAX_DIM):

    h, w = image_bgr.shape[:2]
    scale = max_dim / max(h, w)

    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if not ok:
        raise RuntimeError("JPEG encode failed.")

    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"


def crop(array, cx, cy, size):

    r0, r1 = int(cy - size / 2), int(cy + size / 2)
    c0, c1 = int(cx - size / 2), int(cx + size / 2)

    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, array.shape[0]), min(c1, array.shape[1])

    return array[r0:r1, c0:c1]


def main():

    avhrr_ds = gdal.Open(AVHRR_FILE)
    modis_ds = gdal.Open(MODIS_FILE)
    transformed_ds = gdal.Open(TRANSFORMED_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    transformed = transformed_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    print("Normalizing...")

    avhrr_u8 = normalize_to_uint8(avhrr)
    modis_u8 = normalize_to_uint8(modis)
    transformed_u8 = normalize_to_uint8(transformed)

    print("Building composites...")

    before_overlap = red_cyan_composite(avhrr_u8, modis_u8)
    after_overlap = red_cyan_composite(transformed_u8, modis_u8)

    print("Encoding whole-scene images...")

    avhrr_uri = to_data_uri_gray(avhrr_u8)
    modis_uri = to_data_uri_gray(modis_u8)
    transformed_uri = to_data_uri_gray(transformed_u8)
    before_uri = to_data_uri_bgr(before_overlap)
    after_uri = to_data_uri_bgr(after_overlap)

    print("Building zoomed crops at the largest tie-point shifts...")

    tie_points = pd.read_csv(TIE_POINTS_CSV)
    top = tie_points.sort_values("shift_px", ascending=False).head(6)

    crop_cards = []

    for _, row in top.iterrows():

        cx, cy = float(row.X_IM), float(row.Y_IM)

        avhrr_crop = crop(avhrr, cx, cy, CROP_SIZE)
        modis_crop = crop(modis, cx, cy, CROP_SIZE)
        transformed_crop = crop(transformed, cx, cy, CROP_SIZE)

        before_composite = red_cyan_composite(
            normalize_to_uint8(avhrr_crop), normalize_to_uint8(modis_crop)
        )
        after_composite = red_cyan_composite(
            normalize_to_uint8(transformed_crop), normalize_to_uint8(modis_crop)
        )

        before_uri_c = to_data_uri_bgr(before_composite, max_dim=CROP_DISPLAY)
        after_uri_c = to_data_uri_bgr(after_composite, max_dim=CROP_DISPLAY)

        crop_cards.append(f"""
        <figure class="crop-card">
          <div class="crop-panels">
            <img src="{before_uri_c}" alt="before, point {int(row.POINT_ID)}" width="{CROP_DISPLAY}" height="{CROP_DISPLAY}">
            <img src="{after_uri_c}" alt="after, point {int(row.POINT_ID)}" width="{CROP_DISPLAY}" height="{CROP_DISPLAY}">
          </div>
          <figcaption>Point {int(row.POINT_ID):05d} &mdash; |shift|={row.shift_px:.1f}px &mdash; left: before, right: after</figcaption>
        </figure>
        """)

    crop_html = "\n".join(crop_cards)

    html = TEMPLATE.format(
        avhrr=avhrr_uri,
        modis=modis_uri,
        transformed=transformed_uri,
        before=before_uri,
        after=after_uri,
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
<title>Full-Scene AVHRR / MODIS Overlap (v2, local transform)</title>
<style>
  :root {{
    --bg: #f4f6fb; --surface: #ffffff; --text: #131826; --text-muted: #5b667a;
    --border: #dde3ee; --accent: #2f6fed;
    --shadow: 0 1px 2px rgba(20,30,60,0.06), 0 8px 24px rgba(20,30,60,0.05);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0b1220; --surface:#141c2e; --text:#e6ebf5; --text-muted:#8b96ad;
      --border:#232d43; --accent:#5b9dff;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35); }}
  }}
  :root[data-theme="dark"] {{ --bg:#0b1220; --surface:#141c2e; --text:#e6ebf5; --text-muted:#8b96ad;
    --border:#232d43; --accent:#5b9dff; }}
  :root[data-theme="light"] {{ --bg:#f4f6fb; --surface:#ffffff; --text:#131826; --text-muted:#5b667a;
    --border:#dde3ee; --accent:#2f6fed; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font-family: ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif; line-height:1.45; }}
  header {{ padding: 1.25rem 1.5rem 1rem; border-bottom: 1px solid var(--border); }}
  h1 {{ margin: 0 0 0.15rem; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; text-wrap: balance; }}
  h2 {{ font-size: 1.05rem; margin: 1.8rem 1.5rem 0.7rem; }}
  .subtitle {{ margin: 0; color: var(--text-muted); font-size: 0.9rem; max-width: 72ch; }}
  .pair-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 0 1.5rem; }}
  .single-row {{ display: grid; grid-template-columns: 1fr; gap: 1rem; padding: 0 1.5rem 2rem; max-width: 1100px; margin: 0 auto; }}
  figure {{ margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.6rem; box-shadow: var(--shadow); }}
  img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ margin-top: 0.5rem; font-size: 0.82rem; color: var(--text-muted); text-align: center; }}
  .crop-grid {{ padding: 0 1.5rem 3rem; display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 1rem; }}
  .crop-card {{ margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.6rem; box-shadow: var(--shadow); }}
  .crop-panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; }}
  .crop-panels img {{ width: 100%; height: auto; border-radius: 6px; }}
</style>
</head>
<body>

<header>
  <h1>Full-Scene AVHRR / MODIS Overlap &mdash; v2 local transformation</h1>
  <p class="subtitle">AVHRR resampled through a mosaic of per-tile corrections: every pixel takes its nearest tie point's own (dx, dy) exactly (a Voronoi tiling by nearest tie point, 625 tie points total), rather than one fixed global transform or a smoothed/blended field that would dilute individual corrections. Red/cyan overlap: AVHRR=red, MODIS=cyan &mdash; aligned content reads neutral grey, misalignment shows as colored fringing.</p>
</header>

<h2>The two source images</h2>
<div class="pair-row">
  <figure>
    <img src="{transformed}" alt="Transformed AVHRR, whole scene">
    <figcaption>Transformed AVHRR (mosaic of tie-point tiles)</figcaption>
  </figure>
  <figure>
    <img src="{modis}" alt="MODIS reference, whole scene">
    <figcaption>MODIS reference</figcaption>
  </figure>
</div>

<h2>Overlap: before vs. after</h2>
<div class="pair-row">
  <figure>
    <img src="{before}" alt="Before transform overlap">
    <figcaption>Before &mdash; original AVHRR vs. MODIS</figcaption>
  </figure>
  <figure>
    <img src="{after}" alt="After transform overlap">
    <figcaption>After &mdash; locally-transformed AVHRR vs. MODIS</figcaption>
  </figure>
</div>

<h2>Zoomed crops at the largest shifts (whole-scene view is too zoomed out to show these)</h2>
<div class="crop-grid">
{crops}
</div>

</body>
</html>
"""


if __name__ == "__main__":
    main()
