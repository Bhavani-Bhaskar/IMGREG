"""
Generate a self-contained HTML gallery of every Stage 5 tie point.

For each window that reached Step 21 (mssim_results.csv - the
full Stage 5 output, subpixel + reliability + MSSIM all merged),
renders three panels: the original AVHRR crop, the AVHRR crop
after applying its final (dx, dy) correction, and the MODIS
reference crop. Thumbnails are embedded as base64 JPEG data URIs
so the page has no external dependencies and can be opened
directly in a browser.

Output
------
gallery/tiepoint_gallery.html
"""

import os
import base64
import numpy as np
import pandas as pd
import cv2
from osgeo import gdal
from scipy.ndimage import shift as nd_shift

gdal.UseExceptions()

ROOT = os.path.join(os.path.dirname(__file__), "..")

AVHRR_FILE = os.path.join(ROOT, "2_outputs", "05_avhrr_float32.tif")
MODIS_FILE = os.path.join(ROOT, "2_outputs", "05_modis_float32.tif")

MSSIM_CSV = os.path.join(
    ROOT, "stage5_phase_correlation", "stage5_mssim", "mssim_results.csv"
)

RELIABILITY_CSV = os.path.join(
    ROOT, "stage5_phase_correlation", "stage5_reliability", "reliability_results.csv"
)

OUTPUT_HTML = os.path.join(ROOT, "gallery", "tiepoint_gallery.html")

THUMB_SIZE = 190


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

    resized = cv2.resize(
        image_uint8, (size, size), interpolation=cv2.INTER_AREA
    )

    ok, buf = cv2.imencode(
        ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 82]
    )

    if not ok:
        raise RuntimeError("JPEG encode failed.")

    encoded = base64.b64encode(buf).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def build_card(row, avhrr, modis):

    window_id = int(row.window_id)

    r0, r1 = int(row.row_start), int(row.row_end)
    c0, c1 = int(row.col_start), int(row.col_end)

    dx, dy = float(row.final_dx), float(row.final_dy)

    avhrr_crop = avhrr[r0:r1, c0:c1]
    modis_crop = modis[r0:r1, c0:c1]

    shifted_crop = nd_shift(
        avhrr_crop, shift=(dy, dx), order=1, mode="constant", cval=np.nan
    )

    avhrr_uri = to_data_uri(normalize_to_uint8(avhrr_crop))
    shifted_uri = to_data_uri(normalize_to_uint8(shifted_crop))
    modis_uri = to_data_uri(normalize_to_uint8(modis_crop))

    accepted = bool(row.reliability_accepted)
    improved = bool(row.mssim_increased)

    status_class = "is-accepted" if accepted else "is-rejected"
    status_label = "ACCEPTED" if accepted else "REJECTED"

    delta_sign = "+" if row.mssim_delta >= 0 else ""
    delta_class = "is-up" if improved else "is-down"

    return f"""
    <article class="card {status_class}" data-status="{'accepted' if accepted else 'rejected'}">
      <header class="card-head">
        <span class="win-id">Window {window_id:05d}</span>
        <span class="chip {status_class}">{status_label}</span>
      </header>
      <div class="panels">
        <figure>
          <img src="{avhrr_uri}" alt="AVHRR original, window {window_id}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" loading="lazy">
          <figcaption>AVHRR &middot; original</figcaption>
        </figure>
        <figure>
          <img src="{shifted_uri}" alt="AVHRR shifted, window {window_id}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" loading="lazy">
          <figcaption>AVHRR &middot; shifted</figcaption>
        </figure>
        <figure>
          <img src="{modis_uri}" alt="MODIS reference, window {window_id}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" loading="lazy">
          <figcaption>MODIS &middot; reference</figcaption>
        </figure>
      </div>
      <dl class="meta">
        <div><dt>dx&nbsp;/&nbsp;dy</dt><dd>{dx:+.2f}px&nbsp;/&nbsp;{dy:+.2f}px</dd></div>
        <div><dt>reliability&nbsp;R</dt><dd>{row.reliability:.1f}</dd></div>
        <div><dt>MSSIM</dt><dd>{row.mssim_before:.3f} &rarr; {row.mssim_after:.3f}
          <span class="{delta_class}">({delta_sign}{row.mssim_delta:.3f})</span></dd></div>
      </dl>
    </article>
    """


def main():

    mssim_df = pd.read_csv(MSSIM_CSV)

    reliability_df = pd.read_csv(RELIABILITY_CSV)[
        ["window_id", "reliability", "reliability_accepted"]
    ]

    df = mssim_df.merge(
        reliability_df, on="window_id", how="left"
    ).sort_values("window_id").reset_index(drop=True)

    avhrr_ds = gdal.Open(AVHRR_FILE)
    modis_ds = gdal.Open(MODIS_FILE)

    avhrr = avhrr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    modis = modis_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    total = len(df)
    accepted_count = int(df["reliability_accepted"].sum())
    improved_count = int(df["mssim_increased"].sum())
    mean_r = float(df["reliability"].mean())
    mean_delta = float(df["mssim_delta"].mean())

    print(f"Building gallery for {total} tie points...")

    cards = []

    for index, row in df.iterrows():

        cards.append(build_card(row, avhrr, modis))

        if (index + 1) % 10 == 0 or (index + 1) == total:
            print(f"  {index + 1}/{total}")

    cards_html = "\n".join(cards)

    html = TEMPLATE.format(
        total=total,
        accepted_count=accepted_count,
        rejected_count=total - accepted_count,
        improved_count=improved_count,
        mean_r=f"{mean_r:.1f}",
        mean_delta=f"{mean_delta:+.4f}",
        cards=cards_html,
    )

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_HTML) / (1024 * 1024)

    print(f"\nSaved: {OUTPUT_HTML} ({size_mb:.1f} MB)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 5 Tie Point Gallery</title>
<style>
  :root {{
    --bg: #f4f6fb;
    --surface: #ffffff;
    --text: #131826;
    --text-muted: #5b667a;
    --border: #dde3ee;
    --accent: #2f6fed;
    --good: #1f9d6b;
    --good-bg: #e3f7ee;
    --warn: #b5790a;
    --warn-bg: #fbf0dc;
    --shadow: 0 1px 2px rgba(20, 30, 60, 0.06), 0 8px 24px rgba(20, 30, 60, 0.05);
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0b1220;
      --surface: #141c2e;
      --text: #e6ebf5;
      --text-muted: #8b96ad;
      --border: #232d43;
      --accent: #5b9dff;
      --good: #3ecf8e;
      --good-bg: rgba(62, 207, 142, 0.14);
      --warn: #e0a458;
      --warn-bg: rgba(224, 164, 88, 0.14);
      --shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 8px 24px rgba(0, 0, 0, 0.35);
    }}
  }}

  :root[data-theme="dark"] {{
    --bg: #0b1220;
    --surface: #141c2e;
    --text: #e6ebf5;
    --text-muted: #8b96ad;
    --border: #232d43;
    --accent: #5b9dff;
    --good: #3ecf8e;
    --good-bg: rgba(62, 207, 142, 0.14);
    --warn: #e0a458;
    --warn-bg: rgba(224, 164, 88, 0.14);
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 8px 24px rgba(0, 0, 0, 0.35);
  }}

  :root[data-theme="light"] {{
    --bg: #f4f6fb;
    --surface: #ffffff;
    --text: #131826;
    --text-muted: #5b667a;
    --border: #dde3ee;
    --accent: #2f6fed;
    --good: #1f9d6b;
    --good-bg: #e3f7ee;
    --warn: #b5790a;
    --warn-bg: #fbf0dc;
    --shadow: 0 1px 2px rgba(20, 30, 60, 0.06), 0 8px 24px rgba(20, 30, 60, 0.05);
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.45;
  }}

  .mono {{
    font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
    font-variant-numeric: tabular-nums;
  }}

  header.page {{
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 1.25rem 1.5rem 1rem;
  }}

  h1 {{
    margin: 0 0 0.15rem;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    text-wrap: balance;
  }}

  .subtitle {{
    margin: 0 0 0.9rem;
    color: var(--text-muted);
    font-size: 0.9rem;
    max-width: 62ch;
  }}

  .stat-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.9rem;
  }}

  .stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
    font-size: 0.82rem;
  }}

  .stat b {{
    display: block;
    font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.05rem;
    color: var(--text);
  }}

  .stat span {{ color: var(--text-muted); text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.04em; }}

  .filters {{
    display: flex;
    gap: 0.4rem;
  }}

  .filters button {{
    font: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-muted);
    cursor: pointer;
  }}

  .filters button.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }}

  .filters button:focus-visible {{
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }}

  main {{
    padding: 1.25rem 1.5rem 3rem;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 1rem;
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow);
    padding: 0.85rem;
  }}

  .card.is-accepted {{ border-left-color: var(--good); }}
  .card.is-rejected {{ border-left-color: var(--warn); }}

  .card[hidden] {{ display: none; }}

  .card-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
  }}

  .win-id {{
    font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
  }}

  .chip {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
  }}

  .chip.is-accepted {{ background: var(--good-bg); color: var(--good); }}
  .chip.is-rejected {{ background: var(--warn-bg); color: var(--warn); }}

  .panels {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.4rem;
  }}

  .panels figure {{
    margin: 0;
  }}

  .panels img {{
    display: block;
    width: 100%;
    height: auto;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: 6px;
    background: #000;
  }}

  .panels figcaption {{
    margin-top: 0.3rem;
    font-size: 0.66rem;
    text-align: center;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}

  dl.meta {{
    margin: 0.7rem 0 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 1rem;
    font-size: 0.78rem;
  }}

  dl.meta div {{ display: flex; gap: 0.35rem; }}

  dl.meta dt {{ color: var(--text-muted); }}

  dl.meta dd {{
    margin: 0;
    font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
    font-variant-numeric: tabular-nums;
  }}

  .is-up {{ color: var(--good); }}
  .is-down {{ color: var(--warn); }}

  @media (max-width: 480px) {{
    main {{ grid-template-columns: 1fr; padding: 1rem; }}
  }}
</style>
</head>
<body>

<header class="page">
  <h1>Stage 5 Tie Point Gallery</h1>
  <p class="subtitle">Every window that reached Step 21 (subpixel refinement, reliability, MSSIM). Each card shows the AVHRR crop before correction, after applying its estimated (dx, dy), and the MODIS reference it was matched against.</p>

  <div class="stat-row">
    <div class="stat"><b class="mono">{total}</b><span>Tie points</span></div>
    <div class="stat"><b class="mono">{accepted_count}</b><span>Reliability accepted</span></div>
    <div class="stat"><b class="mono">{rejected_count}</b><span>Rejected</span></div>
    <div class="stat"><b class="mono">{improved_count}</b><span>MSSIM increased</span></div>
    <div class="stat"><b class="mono">{mean_r}</b><span>Mean R</span></div>
    <div class="stat"><b class="mono">{mean_delta}</b><span>Mean MSSIM &Delta;</span></div>
  </div>

  <div class="filters" role="group" aria-label="Filter tie points">
    <button type="button" class="active" data-filter="all">All</button>
    <button type="button" data-filter="accepted">Accepted</button>
    <button type="button" data-filter="rejected">Rejected</button>
  </div>
</header>

<main id="gallery">
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
        const show = filter === "all" || card.dataset.status === filter;
        card.hidden = !show;
      }});
    }});
  }});
</script>

</body>
</html>
"""


if __name__ == "__main__":
    main()
