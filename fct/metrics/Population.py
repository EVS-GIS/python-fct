#!/usr/bin/env python
# coding: utf-8

"""
Désagrège les données carroyées de l'INSEE
à la résolution du raster d'occupation du sol,
en utilisant la surface urbanisée.

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
from multiprocessing import Pool

import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

from .. import transform as fct
from .. import terrain_analysis as ta
from ..cli import starcall
from ..config import config

# workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def grid_extent(geometry, transform):

    # mini = minj = maxi = maxj = None

    # for k in range(len(geometry)):

    #     i, j = ta.index(geometry[k, 0], geometry[k, 1], transform)

    #     if k == 0:
    #         mini = maxi = i
    #         minj = maxj = j
    #         continue

    #     if i < mini:
    #         mini = i
    #     if i > maxi:
    #         maxi = i
    #     if j < minj:
    #         minj = j
    #     if j > maxj:
    #         maxj = j

    # return mini, minj, maxi, maxj

    ij = fct.worldtopixel(geometry, transform)
    return np.min(ij[:, 0]), np.min(ij[:, 1]), np.max(ij[:, 0]), np.max(ij[:, 1])

def TileDisaggregatePopulationData(tile):
    """
    Désagrège les données carroyées de l'INSEE
    à la résolution du raster d'occupation du sol,
    en utilisant la surface urbanisée.
    """

    # height = int(parameter('input.height'))
    # width = int(parameter('input.width'))

    # pop_shapefile = os.path.join(workdir, 'GLOBAL', 'POPULATION', 'POP_INSEE_200M_LA93.shp')
    # landcover_raster = os.path.join(workdir, 'GLOBAL', 'LANDCOVER', 'CESBIO_%02d_%02d.tif' % (row, col))
    # # landcover_raster = os.path.join(workdir, 'CESBIO_2018.vrt')
    # output = os.path.join(workdir, 'GLOBAL', 'POPULATION', 'POP_INSEE_%02d_%02d.tif' % (row, col))

    pop_shapefile = config.datasource('population').filename
    landcover_raster = config.tileset('landcover').tilename('landcover', row=tile.row, col=tile.col)
    output = config.tileset('landcover').tilename('population', row=tile.row, col=tile.col)

    # if os.path.exists(output) and not overwrite:
    #     click.secho('Output already exists: %s' % output, fg='yellow')
    #     return

    with rio.open(landcover_raster) as ds:

        # i, j = ds.index(x0, y0)
        # window = Window(j, i, width, height)
        # landcover = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        landcover = ds.read(1)
        height, width = landcover.shape

        urban_mask = np.uint8((landcover == 6) | (landcover == 7)) + 1
        nodata_mask = (landcover == ds.nodata)
        urban_mask[nodata_mask] = 0
        del landcover

        out = np.zeros((height, width), dtype=np.int32)
        # transform = ds.transform * ds.transform.translation(j, i)
        transform = ds.transform

        # def isdata(i, j):
        #     """
        #     Check bounds for pixel (i, j)
        #     """
        #     return i >= 0 and i < ds.height and j >= 0 and j < ds.width

        def accept(geom):

            mini, minj, maxi, maxj = grid_extent(geom, transform)
            center_i = 0.5 * (mini + maxi)
            center_j = 0.5 * (minj + maxj)

            return all([
                center_i >= 0,
                center_i < height,
                center_j >= 0,
                center_j < width
            ])

        with fiona.open(pop_shapefile) as fs:

            # value_total = 0
            feature_mask = np.zeros((height, width), dtype=np.uint8)

            for feature in fs.filter(bbox=tile.bounds):

                geometry = np.array(feature['geometry']['coordinates'][0][0], dtype='float32')
                value = int(feature['properties']['Ind'])

                if accept(geometry[:, :2]):

                    # value_total += value

                    ta.disaggregate(
                        geometry[:, :2],
                        urban_mask,
                        value,
                        transform,
                        feature_mask,
                        out)

            # click.secho('Total value %d' % value_total)
            # click.secho('Total count %d' % np.sum(out))
            # click.secho('Max value %d' % np.max(out))

            out[nodata_mask] = -1

            profile = ds.profile.copy()
            profile.update(
                driver='GTiff',
                transform=transform,
                height=height,
                width=width,
                dtype='int32',
                nodata=-1)

            with rio.open(output, 'w', **profile) as dst:
                dst.write(out, 1)

def DisaggregatePopulationData(processes=1, **kwargs):

    # tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')

    # with fiona.open(tile_shapefile) as fs:

    #     arguments = list()

    #     for feature in fs:

    #         minx, miny, maxx, maxy = fs.bounds

    #         properties = feature['properties']
    #         x0 = properties['left']
    #         y0 = properties['top']
    #         x1 = properties['right']
    #         y1 = properties['bottom']
    #         bounds = (x0, y1, x1, y0)

    #         row = int((maxy - y0) // 10000)
    #         col = int((x0 - minx) // 10000)

    tileindex = config.tileset('landcover').tileindex
    arguments = [(TileDisaggregatePopulationData, tile, kwargs) for tile in tileindex.values()]

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
