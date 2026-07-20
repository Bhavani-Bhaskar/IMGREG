# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:01:18 2026

@author: laksh
"""

"""
STAGE 4: Dense tile matching (tie-point candidate generation)
================================================================
Method: slide a tile from the AVHRR-derived land mask across a search window
in the MODIS-derived land mask, score with normalized cross-correlation
(cv2.matchTemplate, TM_CCOEFF_NORMED). This IS NCC -- score range 0 to 1.

Several filters, each added because the previous version produced bad
matches:

- tile must be >=90-95% valid data (skip image edges / nodata)
- tile must not be near-uniform land or near-uniform water (frac_bounds) --
  a tile entirely inside one class has no boundary information to match on
- offset must NOT sit at the search-window boundary (abs(dx) or abs(dy) >=
  search-8): a match at the boundary means the true best match is probably
  OUTSIDE the search window and what we found is a false local optimum
- (cloud-gated version) tile must be <15% cloud per the Stage-2 thermal mask,
  since a cloud-contaminated AVHRR tile is matching against ground that may
  look completely different in MODIS (different acquisition time)

A global, single phase-correlation check (cv2.phaseCorrelate) was also run,
once, purely as a sanity check -- NOT as the main matching method. Result:
shift (-0.01, -0.007) px, response ~0.40. Near-zero shift + low confidence
told us there's no single uniform whole-image translation, which is why we
went to local tile matching instead of relying on a single global shift.
"""
import numpy as np
import cv2


def global_phase_correlation_check(img_a, img_b, common_mask):
    """One-shot sanity check: is there a single dominant global shift?"""
    a = img_a.astype(np.float32).copy()
    b = img_b.astype(np.float32).copy()
    a[~common_mask] = 0
    b[~common_mask] = 0
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    shift, response = cv2.phaseCorrelate(a * win, b * win)
    return shift, response


def dense_tile_match(templ_img, templ_valid, ref_img, ref_valid,
                      extra_gate=None,  # e.g. "clear" boolean mask, same shape as templ_img
                      tile=200, step=100, search=170,
                      score_thresh=0.6, frac_bounds=(0.03, 0.97),
                      valid_frac_min=0.9, gate_frac_min=0.85):
    H, W = templ_img.shape
    tie_a, tie_b, scores = [], [], []
    for r in range(search + tile, H - search - tile, step):
        for c in range(search + tile, W - search - tile, step):
            tmask = templ_valid[r:r + tile, c:c + tile]
            if tmask.mean() < valid_frac_min:
                continue
            if extra_gate is not None:
                gate = extra_gate[r:r + tile, c:c + tile]
                if gate.mean() < gate_frac_min:
                    continue
            t = templ_img[r:r + tile, c:c + tile]
            frac = (t > 127).mean()
            if frac < frac_bounds[0] or frac > frac_bounds[1]:
                continue
            sr0, sr1 = r - search, r + tile + search
            sc0, sc1 = c - search, c + tile + search
            rmask = ref_valid[sr0:sr1, sc0:sc1]
            if rmask.mean() < valid_frac_min:
                continue
            region = ref_img[sr0:sr1, sc0:sc1]
            res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, maxloc = cv2.minMaxLoc(res)
            if maxval < score_thresh:
                continue
            dy = maxloc[1] - search
            dx = maxloc[0] - search
            if abs(dx) >= search - 8 or abs(dy) >= search - 8:
                continue
            tie_a.append((c + tile / 2, r + tile / 2))
            tie_b.append((c + tile / 2 + dx, r + tile / 2 + dy))
            scores.append(maxval)
    return np.array(tie_a), np.array(tie_b), np.array(scores)


if __name__ == '__main__':
    s_land = (np.load('s_land.npy').astype(np.uint8) * 255)
    m_land = (np.load('m_land.npy').astype(np.uint8) * 255)
    s_land = cv2.GaussianBlur(s_land, (5, 5), 1.0)
    m_land = cv2.GaussianBlur(m_land, (5, 5), 1.0)

    s_arr = np.load('s_arr.npy')
    m_arr = np.load('m_arr.npy')
    s_valid = s_arr > 0
    m_valid = ~np.isnan(m_arr) & (m_arr > 0)

    cloud = np.load('cloud_mask.npy')
    clear = ~cloud

    # one-time global sanity check (not the main method)
    shift, response = global_phase_correlation_check(s_arr, m_arr, s_valid & m_valid)
    print('global phase-correlation shift:', shift, 'response:', response)

    ta, tm, scores = dense_tile_match(
        s_land, s_valid, m_land, m_valid,
        extra_gate=clear, tile=200, step=100, search=170, score_thresh=0.6,
    )
    print(f'{len(ta)} candidate tie points')
    np.save('tie_a.npy', ta)
    np.save('tie_m.npy', tm)
    np.save('tie_scores.npy', scores)