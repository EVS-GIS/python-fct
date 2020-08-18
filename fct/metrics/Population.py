#!/usr/bin/env python
# coding: utf-8

"""
Disaggregate population data to the resolution of landcover data,
using landcover urban classes.
TODO move outside metrics => data preparation

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
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
from shapely.geometry import asShape, box

from .. import transform as fct
from .. import terrain_analysis as ta
from ..cli import starcall
from ..config import config

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

def DisaggregatePopulationTile(
        tile,
        variable='Ind',
        tileset='default',
        datasource='population',
        landcoverset='landcover',
        destination='population'):
    """
    Désagrège les données carroyées de l'INSEE
    à la résolution du raster d'occupation du sol,
    en utilisant la surface urbanisée.
    """

    pop_shapefile = config.datasource(datasource).filename
    landcover_raster = config.tileset(tileset).tilename(landcoverset, row=tile.row, col=tile.col)
    output = config.tileset(tileset).tilename(destination, row=tile.row, col=tile.col)

    with rio.open(landcover_raster) as ds:

        landcover = ds.read(1)
        height, width = landcover.shape

        urban_mask = np.uint8((landcover == 6) | (landcover == 7)) + 1
        nodata_mask = (landcover == ds.nodata)
        urban_mask[nodata_mask] = 0
        del landcover

        out = np.zeros((height, width), dtype=np.float32)
        transform = ds.transform

        # def accept(geom):
        #     """
        #     Assign feature to the tile containing feature's centroid
        #     """

        #     mini, minj, maxi, maxj = grid_extent(geom, transform)
        #     center_i = 0.5 * (mini + maxi)
        #     center_j = 0.5 * (minj + maxj)

        #     return all([
        #         center_i >= 0,
        #         center_i < height,
        #         center_j >= 0,
        #         center_j < width
        #     ])

        with fiona.open(pop_shapefile) as fs:

            feature_mask = np.zeros((height, width), dtype=np.uint8)

            for feature in fs.filter(bbox=tile.bounds):

                geometry = np.array(feature['geometry']['coordinates'][0][0], dtype='float32')

                feature_geom = asShape(feature['geometry'])
                tile_geom = box(*tile.bounds)
                coverage = tile_geom.intersection(feature_geom).area / feature_geom.area

                value = coverage * feature['properties'][variable]

                if value == 0:
                    continue

                if variable == 'Ind':
                    increment = 1.0
                else:
                    increment = value / (coverage * feature['properties']['Ind'])


                ta.disaggregate(
                    geometry[:, :2],
                    urban_mask,
                    value,
                    increment,
                    transform,
                    feature_mask,
                    out)

            out[nodata_mask] = -1

            profile = ds.profile.copy()
            profile.update(
                driver='GTiff',
                transform=transform,
                height=height,
                width=width,
                dtype='float32',
                nodata=-1)

            with rio.open(output, 'w', **profile) as dst:
                dst.write(out, 1)

def DisaggregatePopulation(processes=1, tileset='default', **kwargs):
    """
    Disaggregate population data to the resolution of landcover data,
    using landcover urban classes.

    Parameters
    ----------

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Keyword arguments
    -----------------

    variable: str

        Variable to disaggregate
        'Ind' or 'Ind_snv' in INSEE Filosofi 2015

    tileset: str

        logical tileset
        defaults to `landcover`

    datasource: str

        logical name of
        population dataset to process,
        defaults to `population`

    landcoverset: str

        logical name of
        landcover dataset to process,
        defaults to `landcover`

    destination: str

        logical name of destination dataset,
        defaults to `population`

    Other keywords are passed to dataset filename templates.
    """

    kwargs.update(tileset=tileset)
    tileindex = config.tileset(tileset)

    def arguments():

        for tile in tileindex.tiles():
            yield (
                DisaggregatePopulationTile,
                tile,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileindex)) as iterator:
            for _ in iterator:
                pass

def test():

    config.default()

    DisaggregatePopulation(
        processes=5,
        variable='Ind',
        destination='population',
        landcoverset='landcover-bdt')

    DisaggregatePopulation(
        processes=5,
        variable='Ind_snv',
        destination='population-income',
        landcoverset='landcover-bdt')
