from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

filename = "/home/bhaskar/Documents/ImageReg/my_project/05_avhrr_reflectance_ch2__shifted_to__modis_1km.bsq"

ds = gdal.Open(filename)

if ds is None:
    raise RuntimeError("Cannot open image")

print("Bands:", ds.RasterCount)

img = ds.GetRasterBand(1).ReadAsArray()

print(img.shape)
print(img.dtype)

plt.figure(figsize=(8,8))
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.show()