# coding: utf-8

"""
Reclassification des métadonnées du RGE Alti 5 m 2017
décrivant les sources des données utilisées
pour assembler le produit final
"""

import rasterio as rio
from rasterio.features import sieve
import numpy as np
import os

srcds = rio.open('RGEALTI_FXX_SRC_50M.tif')
src = srcds.read(1)

maskds = rio.open('RGEALTI_FXX_MASK_50M.tif')
mask = maskds.read(1)

t = [
    (1, 19),    # Raccord entre sources hétérogènes
    (20, 25),   # Photogrammétrie (obsolète)
    (26, 49),   # LiDAR mixte Topo/Bathy
    (50, 99),   # LiDAR Topo
    (100, 159), # Corrélation automatique d'images
    (160, 164), # Radar et données interpolées
    (165, 169), # MNT grille issu de LiDAR
    (170, 189)  # LiDAR Topo IGN en forêt
]

rcls = np.zeros_like(src)

for i, (a, b) in enumerate(t):
    sel = (src >= a) & (src <= b)
    rcls[sel] = i+1

rcls[mask == 0] = 10 # BD Alti 25 m
# rcls[src == 255] = 255

rcls = sieve(rcls, 100)

profile = srcds.profile.copy()
profile.update(nodata=0)

with rio.open('RGEALTI_FXX_SRCLS_50M.tif', 'w', **profile) as dst:
    dst.write(rcls, 1)
