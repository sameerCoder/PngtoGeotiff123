# Testing 15:45
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat

datamat = loadmat("IOEC_ECM_noDA_20190703_masked")
lat = datamat['Yp'].flatten()
lon = datamat['Xp'].flatten()


def plot():
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    # import mplleaflet
    import datetime
    import pandas as pd
    from dateutil.parser import parse
    # import matplotlib as mpl
    # from pylab import  *
    import numpy as np
    import os
    # import rasterio
    # fig=plt.figure()
    # fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(projection=projection))

    # os.environ['PROJ_LIB'] = r'C:\Users\walps\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'

    from datetime import timedelta

    import matplotlib.tri as mtri

    strt = datetime.datetime(2019, 7, 4, 0, 0)
    end = datetime.datetime(2017, 1, 21, 0, 0)

    def perdelta(strt, end, delta):
        curr = strt
        while curr < end:
            yield curr
            curr += delta

    # Read element file
    # data = pd.read_table('fort.ele',delim_whitespace=True,names=('A','B','C','D'))
    tri_new = pd.read_csv('fort.ele', delim_whitespace=True, names=('A', 'B', 'C', 'D'), usecols=[1, 2, 3], skiprows=1,
                          dtype={'D': np.int})
    # data1=data[['B','C','D']]
    # tri=data1[1:]
    dateList = []
    for result in perdelta(strt, strt + timedelta(days=2), timedelta(hours=3)):
        dat = result
        # print(result)
        dt = parse(str(dat))
        yr = dt.year
        mn = dt.month
        d = dt.day
        hr = dt.hour
        mi = dt.minute
        # print(y,mn,d,hr,mi)
        if hr < 10:
            # d='0'+str(d)
            hr = '0' + str(hr)
        else:
            d = str(d)
            hr = str(hr)
        if int(d) < 10:
            d = '0' + str(d)
        else:
            d = str(d)
        varname = 'Hsig_' + str(yr) + '0' + str(mn) + str(d) + '_' + hr + '0000'
        print(varname)

        z = datamat[varname]
        z1 = z.flatten()

        tri_sub = tri_new.apply(lambda x: x - 1)
        triang = mtri.Triangulation(lon, lat, triangles=tri_sub)
        print("type(triang):", type(triang))

        print("type(z):", type(z))

        ax = plt.tripcolor(triang, z1, vmin=0, vmax=2, cmap="Greys")

        break

    from PIL import Image
    import io

    # Save as PNG to a memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')

    # Open image in PIL and access via Numpy and save as you wish
    PILImage = Image.open(buffer)
    na = np.array(PILImage)

    # Now you have Red channel in na[:,:,0], Green channel in na[:,:,1] and Blue channel in na[:,:,2]
    print("ndim na:", np.ndim(na))
    # na = na.flatten()
    na = na[:, :, 0]
    # print(f'Mean: {na.mean()}, Min: {na.min()}, Max: {na.max()}')
    print(f'Mean: {na}')

    return na


na = plot()
import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt

na = np.asarray(na)
# array = na.mean(axis=-1)
array = na
# My image array

# For each pixel I know it's latitude and longitude.
# As you'll see below you only really need the coordinates of
# one corner, and the resolution of the file.

xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]
nrows, ncols = np.shape(array)
# ncols = np.shape(array)
# nrows = len(array)
# ncols = len(array)
xres = (xmax - xmin) / float(ncols)
yres = (ymax - ymin) / float(nrows)
geotransform = (xmin, xres, 0, ymax, 0, -yres)
# That's (top left x, w-e pixel resolution, rotation (0 if North is up),
#         top left y, rotation (0 if North is up), n-s pixel resolution)
# I don't know why rotation is in twice???
# geotransform = (76.93737, 0.03488650145354094, 0, 22.46191, 0, -0.07284429338243273)
output_raster = gdal.GetDriverByName('GTiff').Create('myrasterfinal66.tif', ncols, nrows, 1,
                                                     gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
srs = osr.SpatialReference()  # Establish its coordinate encoding
srs.ImportFromEPSG(4326)  # This one specifies WGS84 lat long.
# Anyone know how to specify the
# IAU2000:49900 Mars encoding?
output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system
# to the file
output_raster.GetRasterBand(1).WriteArray(array)  # Writes my array to the raster

output_raster.FlushCache()