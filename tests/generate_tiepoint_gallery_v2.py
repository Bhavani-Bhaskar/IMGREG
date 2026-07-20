"""
Generate a self-contained HTML gallery of the v2 (AROSICS
COREG_LOCAL) tie points, after independent NCC verification
(stage_v2/tie_points_verified.csv, produced by main_v2.py).

The "corrected" panel is cropped directly from AROSICS' own
corrected output raster (stage_v2/avhrr_corrected.tif), not a
locally reapplied shift.

To keep the page a reasonable size, every "real correction"
point (shift > 1px) is included, plus a random sample of the
"already aligned" anchor points (shift <= 1px) - the full set is
always in stage_v2/tie_points_verified.csv.

Output
------
gallery/tiepoint_gallery_v2.html
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
CORRECTED_FILE = os.path.join(ROOT, "stage_v2", "avhrr_corrected.tif")

TIE_POINTS_CSV = os.path.join(ROOT, "stage_v2", "tie_points_verified.csv")

OUTPUT_HTML = os.path.join(ROOT, "gallery", "tiepoint_gallery_v2.html")

THUMB_SIZE = 190
MAX_ANCHORS_SHOWN = 40


def normalize_to_uint8(window):

    valid = np.isfinite(window)

    if valid.sum() < 2:
        return np.zeros(window.shape, dtype=np.uint8)

    lo = float(window[valid].min())
    hi = float(window[valid].max())

    if (hi - lo) < 1e-12:
        return np.zeros(window.shape, dtype=np.uint8)

    normalized = np.clip((window - lo) / (hi - lo), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0)

    return (normalized * 255).astype(np.uint8)


def to_data_uri(image_uint8, size=THUMB_SIZE):

    resized = cv2.resize(image_uint8, (size, size), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 82])

    if not ok:
        raise RuntimeError("JPEG encode failed.")

    encoded = base64.b64encode(buf).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def crop_window(array, center_x, center_y, win_x, win_y):

    r0 = int(center_y - win_y / 2)
    r1 = int(center_y + win_y / 2)
    c0 = int(center_x - win_x / 2)
    c1 = int(center_x + win_x / 2)

    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, array.shape[0]), min(c1, array.shape[1])

    return array[r0:r1, c0:c1]


def build_card(row, avhrr, corrected, modis):

    point_id = int(row.POINT_ID)

    cx, cy = float(row.X_IM), float(row.Y_IM)
    wx, wy = float(row.X_WIN_SIZE), float(row.Y_WIN_SIZE)

    avhrr_crop = crop_window(avhrr, cx, cy, wx, wy)
    corrected_crop = crop_window(corrected, cx, cy, wx, wy)
    modis_crop = crop_window(modis, cx, cy, wx, wy)

    avhrr_uri = to_data_uri(normalize_to_uint8(avhrr_crop))
    corrected_uri = to_data_uri(normalize_to_uint8(corrected_crop))
    modis_uri = to_data_uri(normalize_to_uint8(modis_crop))

    is_real = row.shift_px > 1.0
    status_class = "is-real" if is_real else "is-anchor"
    status_label = "REAL CORRECTION" if is_real else "ANCHOR (already aligned)"

    delta = float(row.ncc_after) - float(row.ncc_before)
    delta_sign = "+" if delta >= 0 else ""
    delta_class = "is-up" if delta >= 0 else "is-down"

    return f"""
    <article class="card {status_class}" data-status="{'real' if is_real else 'anchor'}">
      <header class="card-head">
        <span class="win-id">Point {point_id:05d}</span>
        <span class="chip {status_class}">{status_label}</span>
      </header>
      <div class="panels">
        <figure>
          <img src="{avhrr_uri}" alt="AVHRR original, point {point_id}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" loading="lazy">
          <figcaption>AVHRR &middot; original</figcaption>
        </figure>
        <figure>
          <img src="{corrected_uri}" alt="AVHRR corrected, point {point_id}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" loading="lazy">
          <figcaption>AVHRR &middot; corrected</figcaption>
        </figure>
        <figure>
          <img src="{modis_uri}" alt="MODIS reference, point {point_id}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" loading="lazy">
          <figcaption>MODIS &middot; reference</figcaption>
        </figure>
      </div>
      <dl class="meta">
        <div><dt>dx&nbsp;/&nbsp;dy</dt><dd>{row.X_SHIFT_PX:+.2f}px&nbsp;/&nbsp;{row.Y_SHIFT_PX:+.2f}px</dd></div>
        <div><dt>|shift|</dt><dd>{row.shift_px:.2f}px</dd></div>
        <div><dt>NCC</dt><dd>{row.ncc_before:.3f} &rarr; {row.ncc_after:.3f}
          <span class="{delta_class}">({delta_sign}{delta:.3f})</span></dd></div>
      </dl>
    </article>
    """


def main():

    df = pd.read_csv(TIE_POINTS_CSV)

    real = df[df["shift_px"] > 1.0].sort_values("shift_px", ascending=False)
    anchors = df[df["shift_px"] <= 1.0]

    if len(anchors) > MAX_ANCHORS_SHOWN:
        anchors = anchors.sample(MAX_ANCHORS_SHOWN, random_state=0)

    shown = pd.concat([real, anchors]).sort_values("POINT_ID").reset_index(drop=True)

    avhrr_ds = gdal.Open(AVHRR_FILE)
    corrected_ds = gdal.Open(CORRECTED_FILE)
    modis_ds = gdal.Open(MODIS_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    corrected = corrected_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    total_verified = len(df)
    real_count = int((df["shift_px"] > 1.0).sum())
    anchor_count = total_verified - real_count
    mean_improvement = float((real["ncc_after"] - real["ncc_before"]).mean())

    print(f"Building v2 gallery: {len(shown)} of {total_verified} verified tie points shown "
          f"(all {real_count} real corrections + {len(anchors)} of {anchor_count} anchors)...")

    cards = [build_card(row, avhrr, corrected, modis) for _, row in shown.iterrows()]

    cards_html = "\n".join(cards)

    html = TEMPLATE.format(
        total=total_verified,
        shown=len(shown),
        real_count=real_count,
        anchor_count=anchor_count,
        mean_improvement=f"{mean_improvement:+.4f}",
        cards=cards_html,
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
<title>Stage 5 Tie Point Gallery (v2, AROSICS + NCC verified)</title>
<style>
  :root {{
    --bg: #f4f6fb; --surface: #ffffff; --text: #131826; --text-muted: #5b667a;
    --border: #dde3ee; --accent: #2f6fed; --good: #1f9d6b; --good-bg: #e3f7ee;
    --warn: #b5790a; --warn-bg: #fbf0dc;
    --shadow: 0 1px 2px rgba(20, 30, 60, 0.06), 0 8px 24px rgba(20, 30, 60, 0.05);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0b1220; --surface: #141c2e; --text: #e6ebf5; --text-muted: #8b96ad;
      --border: #232d43; --accent: #5b9dff; --good: #3ecf8e; --good-bg: rgba(62,207,142,0.14);
      --warn: #e0a458; --warn-bg: rgba(224,164,88,0.14);
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0b1220; --surface: #141c2e; --text: #e6ebf5; --text-muted: #8b96ad;
    --border: #232d43; --accent: #5b9dff; --good: #3ecf8e; --good-bg: rgba(62,207,142,0.14);
    --warn: #e0a458; --warn-bg: rgba(224,164,88,0.14);
  }}
  :root[data-theme="light"] {{
    --bg: #f4f6fb; --surface: #ffffff; --text: #131826; --text-muted: #5b667a;
    --border: #dde3ee; --accent: #2f6fed; --good: #1f9d6b; --good-bg: #e3f7ee;
    --warn: #b5790a; --warn-bg: #fbf0dc;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.45; }}
  header.page {{ position: sticky; top: 0; z-index: 10; background: var(--bg);
    border-bottom: 1px solid var(--border); padding: 1.25rem 1.5rem 1rem; }}
  h1 {{ margin: 0 0 0.15rem; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em; text-wrap: balance; }}
  .subtitle {{ margin: 0 0 0.9rem; color: var(--text-muted); font-size: 0.9rem; max-width: 68ch; }}
  .stat-row {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 0.9rem; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.45rem 0.75rem; font-size: 0.82rem; }}
  .stat b {{ display: block; font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
    font-variant-numeric: tabular-nums; font-size: 1.05rem; color: var(--text); }}
  .stat span {{ color: var(--text-muted); text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.04em; }}
  .filters {{ display: flex; gap: 0.4rem; }}
  .filters button {{ font: inherit; font-size: 0.82rem; font-weight: 600; padding: 0.4rem 0.85rem;
    border-radius: 999px; border: 1px solid var(--border); background: var(--surface); color: var(--text-muted); cursor: pointer; }}
  .filters button.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
  main {{ padding: 1.25rem 1.5rem 3rem; display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 1rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--border);
    border-radius: 10px; box-shadow: var(--shadow); padding: 0.85rem; }}
  .card.is-real {{ border-left-color: var(--accent); }}
  .card.is-anchor {{ border-left-color: var(--good); }}
  .card[hidden] {{ display: none; }}
  .card-head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem; gap: 0.5rem; }}
  .win-id {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace; font-size: 0.85rem;
    font-weight: 600; color: var(--text-muted); }}
  .chip {{ font-size: 0.62rem; font-weight: 700; letter-spacing: 0.03em; padding: 0.2rem 0.5rem; border-radius: 999px; white-space: nowrap; }}
  .chip.is-real {{ background: rgba(47,111,237,0.14); color: var(--accent); }}
  .chip.is-anchor {{ background: var(--good-bg); color: var(--good); }}
  .panels {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem; }}
  .panels figure {{ margin: 0; }}
  .panels img {{ display: block; width: 100%; height: auto; aspect-ratio: 1/1; object-fit: cover; border-radius: 6px; background: #000; }}
  .panels figcaption {{ margin-top: 0.3rem; font-size: 0.66rem; text-align: center; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.03em; }}
  dl.meta {{ margin: 0.7rem 0 0; display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; font-size: 0.78rem; }}
  dl.meta div {{ display: flex; gap: 0.35rem; }}
  dl.meta dt {{ color: var(--text-muted); }}
  dl.meta dd {{ margin: 0; font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
    font-variant-numeric: tabular-nums; }}
  .is-up {{ color: var(--good); }}
  .is-down {{ color: var(--warn); }}
  @media (max-width: 480px) {{ main {{ grid-template-columns: 1fr; padding: 1rem; }} }}
</style>
</head>
<body>

<header class="page">
  <h1>Stage 5 Tie Point Gallery &mdash; v2 (AROSICS + independent NCC verification)</h1>
  <p class="subtitle">Tie points that survived AROSICS' own SSIM + RANSAC filtering, AND independently re-verified via normalized cross-correlation computed directly from the source rasters. "Anchor" points (shift &le; 1px) are already-aligned constraints, kept without requiring NCC improvement. Showing all real corrections plus a sample of anchors - full set in stage_v2/tie_points_verified.csv.</p>
  <div class="stat-row">
    <div class="stat"><b>{total}</b><span>Total verified</span></div>
    <div class="stat"><b>{real_count}</b><span>Real corrections (&gt;1px)</span></div>
    <div class="stat"><b>{anchor_count}</b><span>Anchor points</span></div>
    <div class="stat"><b>{mean_improvement}</b><span>Mean NCC &Delta; (real)</span></div>
    <div class="stat"><b>{shown}</b><span>Shown below</span></div>
  </div>
  <div class="filters" role="group" aria-label="Filter tie points">
    <button type="button" class="active" data-filter="all">All</button>
    <button type="button" data-filter="real">Real corrections</button>
    <button type="button" data-filter="anchor">Anchors</button>
  </div>
</header>

<main>
{cards}
</main>

<script>
  const buttons = document.querySelectorAll(".filters button");
  const cards = document.querySelectorAll(".card");
  buttons.forEach(btn => {{
    btn.addEventListener("click", () => {{
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const filter = btn.dataset.filter;
      cards.forEach(card => {{
        card.hidden = !(filter === "all" || card.dataset.status === filter);
      }});
    }});
  }});
</script>

</body>
</html>
"""


if __name__ == "__main__":
    main()
