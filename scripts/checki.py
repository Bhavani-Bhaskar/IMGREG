from osgeo import gdal
import numpy as np
reference = "/home/bhaskar/Documents/ImageReg/2_outputs/05_modis_float32.tif"

# ds = gdal.Open(reference)
# #print(ds.RasterXSize, ds.RasterYSize)
target = "/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_float32.tif"
# ds = gdal.Open(target)
# #print(ds.RasterXSize, ds.RasterYSize)


# Quick sanity check - histogram compare

# ref = gdal.Open(reference).ReadAsArray()
# tgt = gdal.Open(target).ReadAsArray()
# print(f"Ref: mean={np.nanmean(ref):.3f}, std={np.nanstd(ref):.3f}")
# print(f"Tgt: mean={np.nanmean(tgt):.3f}, std={np.nanstd(tgt):.3f}")



# gdal.UseExceptions()

# for path in [reference, target]:
#     ds = gdal.Open(path)
#     print(path)
#     print(" Bands:", ds.RasterCount)
#     for i in range(1, ds.RasterCount + 1):
#         band = ds.GetRasterBand(i)
#         print(f"  Band {i}: min={band.GetMinimum()}, max={band.GetMaximum()}, "
#               f"desc={band.GetDescription()}, NoData={band.GetNoDataValue()}")
#     md = ds.GetMetadata()
#     print(" Metadata:", md)



# for f in [reference, target]:
#     ds = gdal.Open(f)

#     # print("\n", f)
#     # print("Size:", ds.RasterXSize, ds.RasterYSize)
#     # print("GeoTransform:", ds.GetGeoTransform())
#     # print("Projection:", ds.GetProjection())
#     # print("NoData:", ds.GetRasterBand(1).GetNoDataValue())

#     img = ds.ReadAsArray()

#     print("\n", f)
#     print("Min :", np.min(img))
#     print("Max :", np.max(img))
#     print("Unique zeros :", np.sum(img == 0))
#     print("NaNs :", np.isnan(img).sum())

#     print(np.nanmin(img))
#     print(np.nanmax(img))

# mask = gdal.Open("/home/bhaskar/Documents/ImageReg/2_outputs/07_avhrr_mask.tif").ReadAsArray()

# print(np.unique(mask))



# from osgeo import gdal
# import numpy as np

# gdal.UseExceptions()

# filename = "/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_float32.tif"

# print("Opening file...")

# ds = gdal.Open(filename)

# if ds is None:
#     print("Failed to open image.")
#     exit()

# print("Opened successfully.")

# band = ds.GetRasterBand(1)

# img = band.ReadAsArray()

# print("Array shape:", img.shape)
# print("dtype:", img.dtype)

# print("Zeros:", np.sum(img == 0))
# print("NaNs:", np.isnan(img).sum())

# ds = gdal.Open(filename, gdal.GA_Update)

# band = ds.GetRasterBand(1)

# img = band.ReadAsArray().astype(np.float32)

# print("Before:", np.sum(img == 0))

# img[img == 0] = np.nan

# band.WriteArray(img)
# band.FlushCache()

# ds = None

# print("Done.")




# from osgeo import gdal
# import numpy as np

# modis = gdal.Open("/home/bhaskar/Documents/ImageReg/2_outputs/05_modis_float32.tif").ReadAsArray()
# avhrr = gdal.Open("/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_float32.tif").ReadAsArray()

# valid_modis = ~np.isnan(modis)
# valid_avhrr = ~np.isnan(avhrr)

# print("MODIS valid :", valid_modis.sum())
# print("AVHRR valid :", valid_avhrr.sum())

# common = valid_modis & valid_avhrr

# print("Common valid :", common.sum())

# rows, cols = np.where(valid_modis)

# print("MODIS")
# print(rows.min(), rows.max())
# print(cols.min(), cols.max())

# rows, cols = np.where(valid_avhrr)

# print("AVHRR")
# print(rows.min(), rows.max())
# print(cols.min(), cols.max())




# from osgeo import gdal
# gdal.UseExceptions()

# path = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/hrpt_M03_20250506_0420_33701_geo_b2.tif"  # <-- the file BEFORE your conversion
# ds = gdal.Open(path)

# print("=== Dataset-level metadata ===")
# for k, v in ds.GetMetadata().items():
#     print(f"  {k}: {v}")

# print("\n=== All metadata domains ===")
# for domain in ds.GetMetadataDomainList() or []:
#     print(f"-- Domain: {domain} --")
#     for k, v in ds.GetMetadata(domain).items():
#         print(f"  {k}: {v}")

# print("\n=== Band info ===")
# for i in range(1, ds.RasterCount + 1):
#     b = ds.GetRasterBand(i)
#     print(f"Band {i}: dtype={gdal.GetDataTypeName(b.DataType)}, "
#           f"scale={b.GetScale()}, offset={b.GetOffset()}, "
#           f"desc={b.GetDescription()}, unittype={b.GetUnitType()}")

# print("\n=== Raw file tags (via gdalinfo -json equivalent) ===")
# print(ds.GetDriver().ShortName)
# print("Corner coords:", ds.GetGeoTransform())



ds = gdal.Open("2_outputs/05_avhrr_reflectance_ch2.tif")
arr = ds.GetRasterBand(1).ReadAsArray()
print(f"Mean: {np.nanmean(arr):.4f}, Std: {np.nanstd(arr):.4f}")