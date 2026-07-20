# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:00:02 2026

@author: laksh
"""

"""
STAGE 2: Cloud masking
=======================
First attempt (brightness threshold on the visible/SWIR band) failed:
snow on the Himalaya is also bright, so it got misclassified as cloud and
removed exactly the most useful high-contrast terrain feature.

Fix: use the thermal IR channel (ch4, ~11um). Clouds are cold -> distinct
digital-count population well separated from the warm land/ocean surface.
Checked the global histogram of valid thermal pixels: a clear main peak
around DN~474-492 (surface), then a secondary, broader elevated population
from DN~600-900 (cloud tops). A fixed threshold around 600 cleanly separates
real clouds from land -- including correctly leaving cold high-altitude
terrain (Tibetan plateau, snow) classified as clear, since it's not as
extreme as actual cloud-top counts.

A row-banded Otsu threshold (which worked nicely for land/water masking
later in Stage 3) was tried here FIRST and discarded -- thermal brightness
isn't reliably bimodal within each row band the way land/water reflectance
is, so per-band Otsu produced obviously wrong masks (entire clear bands
marked as ~100% cloud). Lesson: check the global histogram before assuming
Otsu will do something sensible.
"""
import numpy as np


def thermal_cloud_mask(b4_arr, threshold=600):
    valid = b4_arr > 0
    cloud = (b4_arr > threshold) & valid
    return cloud, valid


def inspect_histogram(b4_arr, bins=50):
    valid = b4_arr > 0
    vals = b4_arr[valid]
    hist, edges = np.histogram(vals, bins=bins)
    for h, e in zip(hist, edges):
        print(int(e), h)
    return hist, edges


if __name__ == '__main__':
    b4_arr = np.load('b4_arr.npy')
    inspect_histogram(b4_arr)          # used to pick threshold=600 by eye
    cloud, valid = thermal_cloud_mask(b4_arr, threshold=600)
    print('cloud fraction of valid pixels:', cloud[valid].mean())
    np.save('cloud_mask.npy', cloud)