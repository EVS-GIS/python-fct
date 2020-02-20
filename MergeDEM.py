# coding: utf-8

"""
Remplace les pixels no-data du RGE Alti 5 m
par les valeurs interpolées de la BD Alti 25 m.

Date: 2020-02-05
"""

import rasterio as rio
import numpy as np
import os
from tqdm import tqdm

def merge(bassin, zone):
    """
    Opération élémentaire pour une zone hydrographique :
    1. chargement du RGE
    2. chargement de le BD Alti
    3. remplacement des pixes no-data du RGE
    4. écriture du résultat dans un nouveau raster
    """

    rge5mfile = os.path.join('ZONEHYDR', bassin, zone, 'RGEALTI5M.tif')
    bdaltifile = os.path.join('ZONEHYDR', bassin, zone, 'BDALTI_UPSCALED5M.tif')
    target = os.path.join('ZONEHYDR', bassin, zone, 'DEM5M.tif')
    
    with rio.open(rge5mfile) as ds:
    
        rge5m = ds.read(1)
    
        with rio.open(bdaltifile) as ds2:
            bdalti = ds2.read(1)
            mask = (rge5m == ds.nodata) & (bdalti != ds2.nodata)
            rge5m[mask] = bdalti[mask]
    
        with rio.open(target, 'w', **ds.profile) as dst:
            dst.write(rge5m, 1)

with open('ZONEHYDR/ZoneHydroAlti25m.list') as fp:
    zones = [info.strip().split(' ') for info in fp]

for bassin, zone in tqdm(zones):
    merge(bassin, zone)
