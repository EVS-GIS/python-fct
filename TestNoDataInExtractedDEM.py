import rasterio as rio
import numpy as np
import os
from tqdm import tqdm

with open('ZONEHYDR/ZoneHydroRhoneAlpes.list') as fp:
    zones = [info.strip().split(' ') for info in fp]

for bassin, zone in tqdm(zones):
    # demfile = os.path.join('ZONEHYDR', bassin, zone, 'RGEALTI5M.tif')
    demfile = os.path.join('ZONEHYDR', bassin, zone, 'DEM5M.tif')
    with rio.open(demfile) as ds:
        dem = ds.read(1)
        height, width = dem.shape
        nodata_cnt = np.sum(dem == ds.nodata)
        print(zone, '%.2f' % (100.0 * nodata_cnt / (height*width)))
