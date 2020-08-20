#!/usr/bin/env python
# coding: utf-8

"""
1ère étape du calcul du plan de drainage global :

- remplit les zones NODATA du MNT de résolution 5 m (RGE Alti 5 m)
  avec les valeurs interpolées de la BD Alti 25 m

- identifie et numérote les bassins versants
  et les zones continues de même altitude,
  avec remplissage des creux

- construit le graphe de connection
  entre bassins versants contigus

Séquence :

1. Extract and Path DEM
2. Fill Sinks and Label Flats
3. Resolve Global Spillover Graph
4. Apply Spillover Elevations and Map Flats

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
from collections import defaultdict
from heapq import heappush, heappop
import numpy as np

import click
import rasterio as rio
from rasterio.features import rasterize
import fiona
import fiona.crs

from ..config import config
from .. import terrain_analysis as ta
from .. import speedup
from ..tileio import ReadRasterTile
from .Burn import BurnTile

def workdir():
    """
    Return default working directory
    """
    return config.workdir

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

def silent(msg):
    pass

def TileExtendedBoundingBox(row, col, padding=20):

    template = config.filename('patched', row=row, col=col)
    output = os.path.join(workdir(), 'TILEBOXES.shp')

    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('TILE', 'int')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    tile_index = tileindex()

    with rio.open(template) as ds:

        gid = tile_index[(row, col)].gid
        height, width = ds.shape
        xmin, ymax = ds.xy(-padding, -padding)
        xmax, ymin = ds.xy(height+padding, width+padding)

        box = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
        geom = {'type': 'Polygon', 'coordinates': [box]}
        props = {'TILE': gid}

        if os.path.exists(output):
            mode = 'a'
        else:
            mode = 'w'

        with fiona.open(output, mode, **options) as dst:
            feature = {'geometry': geom, 'properties': props}
            dst.write(feature)

def ExtractAndPatchTile(
        row, col,
        overwrite=True,
        verbose=False,
        smooth=0,
        exterior='exterior-domain',
        exterior_data=9000.0):
    """
    Remplit les zones NODATA du MNT de résolution 5 m (RGE Alti 5 m)
    avec les valeurs interpolées de la BD Alti 25 m
    """

    from scipy.ndimage import uniform_filter as ndfilter

    # tile_index = tileindex()
    # noout = config.tileset().noout

    # if (row, col) not in tile_index:
    #     return

    output = config.tileset().tilename('dem', row=row, col=col)

    if verbose:

        def info(msg):
            click.secho(msg, fg='cyan')

        def step(msg):
            click.secho(msg, fg='yellow')

    else:

        info = step = silent

    if os.path.exists(output) and not overwrite:
        info('Output already exists: %s' % output)
        return

    # info('Processing tile (%02d, %02d)' % (row, col))
    # step('Read and patch elevations')

    # with rio.open(DEM) as ds:

        # row_offset, col_offset = ds.index(tile.x0, tile.y0)

    elevations, profile = ReadRasterTile(row, col, 'dem1', 'dem2', padding=smooth)
    transform = profile['transform']
    nodata = profile['nodata']

    if smooth > 0:
        out = ndfilter(elevations, smooth)
        mask = ndfilter(np.uint8(elevations != nodata), smooth)
        out[mask < 1] = elevations[mask < 1]
        out[mask == 0] = nodata
        out = out[smooth:-smooth, smooth:-smooth]
        del mask
    else:
        out = elevations

    if exterior and exterior != 'off':

        with fiona.open(config.datasource(exterior).filename) as fs:
            mask = rasterize(
                [f['geometry'] for f in fs],
                out_shape=out.shape,
                transform=transform,
                fill=0,
                default_value=1,
                dtype='uint8')

        out[(out == nodata) & (mask == 1)] = exterior_data

    # profile = ds.profile.copy()
    # profile.update(
    #     compress='deflate',
    #     transform=transform,
    #     height=tile_height,
    #     width=tile_width,
    #     nodata=nodata
    # )

    with rio.open(output, 'w', **profile) as dst:
        dst.write(out, 1)

def MeanFilter(
        row, col,
        overwrite=True,
        window=5):
    """
    Smooth elevations by applying a mean filter
    on a square window of size `size`
    """

    from scipy.ndimage import uniform_filter as ndfilter

    # tile_index = tileindex()

    # if (row, col) not in tile_index:
    #     return

    output = config.tileset().tilename('smoothed', row=row, col=col)

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    with rio.open(config.tileset().tilename('dem', row=row, col=col)) as ds:

        data = ds.read(1)
        out = ndfilter(data, window)
        out[data == ds.nodata] = ds.nodata

        profile = ds.profile.copy()

        with rio.open(output, 'w', **profile) as dst:
            dst.write(data, 1)
