from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

ds = gdal.Open("/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_reflectance_ch2.tif")
arr = ds.GetRasterBand(1).ReadAsArray()

plt.figure(figsize=(8, 12))
plt.imshow(np.isnan(arr), cmap="gray")
plt.title("NaN mask (white = NaN)")
plt.savefig("nan_mask_check.png", dpi=100)
print(f"Valid: {np.mean(~np.isnan(arr))*100:.1f}%")