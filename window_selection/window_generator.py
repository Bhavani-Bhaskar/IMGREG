
"""
window_generator.py

Purpose:
--------
Generate overlapping sliding windows for the AVHRR image.

This module only creates window geometry.
It does NOT:
    - perform cloud screening
    - find MODIS windows
    - perform registration

Author: Bhaskar
"""

from dataclasses import dataclass


@dataclass
class Window:
    """
    Represents one AVHRR sliding window.
    """
    id: int

    row_start: int
    row_end: int

    col_start: int
    col_end: int

    center_row: int
    center_col: int


def generate_windows(image_shape,
                     window_size=256,
                     stride=128):
    """
    Generate overlapping sliding windows.

    Parameters
    ----------
    image_shape : tuple
        (rows, cols)

    window_size : int

    stride : int

    Returns
    -------
    list[Window]
    """
    print("Window Size :", window_size)
    print("Stride :", stride)
    rows, cols = image_shape

    windows = []

    window_id = 0

    for r in range(0,
                   rows - window_size + 1,
                   stride):

        for c in range(0,
                       cols - window_size + 1,
                       stride):

            center_row = r + window_size // 2
            center_col = c + window_size // 2

            windows.append(

                Window(

                    id=window_id,

                    row_start=r,
                    row_end=r + window_size,

                    col_start=c,
                    col_end=c + window_size,

                    center_row=center_row,
                    center_col=center_col

                )

            )

            window_id += 1

    return windows
