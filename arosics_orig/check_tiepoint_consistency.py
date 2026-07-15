import pandas as pd
import numpy as np

#df = pd.read_csv("/home/claude/tiepoint_diagnostics.csv") if False else None
# NOTE: point this at the fresh table from your latest run instead -
# re-save it right after calculate_spatial_shifts() in run_coreg_local.py, e.g.:
#   CRL.CoRegPoints_table.to_csv("latest_tiepoints.csv", index=False)
# then run this script on that file.

df = pd.read_csv("/home/bhaskar/Documents/ImageReg/arosics_orig/latest_tiepoints.csv")

valid = df[(df['ABS_SHIFT'] != -9999)]
if 'L1_OUTLIER' in valid.columns:
    valid = valid[(valid['L1_OUTLIER'] == False) & (valid['L2_OUTLIER'] == False)]
    if 'L3_OUTLIER' in valid.columns:
        valid = valid[valid['L3_OUTLIER'] == False]

print(f"Valid tie points analyzed: {len(valid)}")
print("\nX_SHIFT_M stats:")
print(valid['X_SHIFT_M'].describe())
print("\nY_SHIFT_M stats:")
print(valid['Y_SHIFT_M'].describe())
print("\nCoefficient of variation (std/mean) — LOW means consistent/trustworthy, HIGH means scattered/noisy:")
print("X:", valid['X_SHIFT_M'].std() / abs(valid['X_SHIFT_M'].mean()))
print("Y:", valid['Y_SHIFT_M'].std() / abs(valid['Y_SHIFT_M'].mean()))

# spatial spread check: are points clustered in one small area, or spread across the overlap?
print("\nSpatial extent of valid points:")
print("X_MAP range:", valid['X_MAP'].min(), "to", valid['X_MAP'].max())
print("Y_MAP range:", valid['Y_MAP'].min(), "to", valid['Y_MAP'].max())