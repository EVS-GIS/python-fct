# coding: utf-8

"""
Sequence :

1. FlowDirection (*)
2. Outlets (*)
3. AggregateOutlets

(*) Possibly Parallel Steps

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
import numpy as np

import click
import rasterio as rio
from rasterio.features import rasterize
import fiona
import fiona.crs

from ..config import (
    config,
    DatasetParameter,
    DatasourceParameter
)
from .. import terrain_analysis as ta
from ..tileio import PadRaster

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

class Parameters():
    """
    Flow direction parameters
    """

    exterior = DatasourceParameter('exterior domain')

    elevations = DatasetParameter('filled-resolved elevation raster (DEM)', type='input')
    flow = DatasetParameter('flow direction raster', type='output')
    
    def __init__(self):
        """
        Default paramater values
        """

        self.exterior = 'exterior-domain'
        self.dem = 'dem-drainage-resolved'
        self.flow = 'flow'

def WallFlats(padded, nodata):

    ci = [ -1, -1,  0,  1,  1,  1,  0, -1 ]
    cj = [  0,  1,  1,  1,  0, -1, -1, -1 ]

    height, width = padded.shape
    zwall = np.max(padded)
    fixed = 0

    flow = ta.flowdir(padded, nodata)

    # top and bottom
    for ik in [0, height-1]:
        for jk in range(width):
            direction = flow[ik, jk]
            if direction != -1 and direction != 0:
                n = int(np.log2(direction))
                i = ik + ci[n]
                j = jk + cj[n]
                if all([i >= 0, i < height, j >= 0, j < width]):
                    if flow[i, j] == 0 and padded[ik, jk] > padded[i, j]:
                        padded[ik, jk] = zwall
                        fixed += 1

    # left and right
    for jk in [0, width-1]:
        for ik in range(height):
            direction = flow[ik, jk]
            if direction != -1 and direction != 0:
                n = int(np.log2(direction))
                i = ik + ci[n]
                j = jk + cj[n]
                if all([i >= 0, i < height, j >= 0, j < width]):
                    if flow[i, j] == 0 and padded[ik, jk] > padded[i, j]:
                        padded[ik, jk] = zwall
                        fixed += 1

    return fixed


def FlowDirectionTile(
        row, col,
        params,
        overwrite=True,
        **kwargs):
    """
    Resolve flats drainage direction and
    calculate D8 flow direction from adjusted elevations.
    """

    # elevation_raster = config.tileset().filename('filled', row=row, col=col)
    output = params.flow.tilename(row=row, col=col, **kwargs)
    # config.tileset().tilename('flow', row=row, col=col)

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    # with rio.open(elevation_raster) as ds:

    # padded = PadRaster(row, col, 'filled')
    padded, profile = PadRaster(row, col, params.elevations.name)
    transform = profile['transform']
    nodata = profile['nodata']

    WallFlats(padded, nodata)

    # ***********************************************************************
    # Wall flowing border flats

    # flow = ta.flowdir(padded, ds.nodata)
    # labels, outlets = speedup.flat_labels(flow, padded, ds.nodata)
    # notflowing = {k+1 for k, (i, j) in enumerate(outlets) if i == -1 and j == -1}

    # height, width = labels.shape
    # boxes = speedup.flat_boxes(labels)

    # borders = set()
    # for w, (mini, minj, maxi, maxj, count) in boxes.items():
    #     if mini == 0 or minj == 0 or maxi == (height-1) or maxj == (width-1):
    #         if w not in notflowing:
    #             borders.add(w)

    # @np.vectorize
    # def bordermask(x):
    #     return x in borders

    # mask = bordermask(labels)
    # mask[1:-1, 1:-1] = False
    # padded[mask] = np.max(padded)

    # ***********************************************************************

    # Option 1
    # extended = rd.rdarray(padded, no_data=ds.nodata)
    # rd.FillDepressions(extended, True, True, 'D8')
    # flow = ta.flowdir(padded, ds.nodata)

    # Option 2
    flow = ta.flowdir(padded, nodata)
    flat_mask, flat_labels = ta.resolve_flat(padded, flow)
    ta.flat_mask_flowdir(flat_mask, flow, flat_labels)

    # extended = rd.rdarray(flat_mask, no_data=0)
    # rd.FillDepressions(extended, True, True, 'D8')

    # extended = rd.rdarray(padded, no_data=ds.nodata)
    # # rd.BreachDepressions(extended, True, 'D8')
    # # rd.ResolveFlats(extended, True)
    # rd.FillDepressions(extended, True, True, 'D8')
    # flow = ta.flowdir(padded, ds.nodata)

    exterior = params.exterior.filename()

    if exterior and os.path.exists(exterior):

        with fiona.open(exterior) as fs:
            mask = rasterize(
                [f['geometry'] for f in fs],
                out_shape=flow.shape,
                transform=transform,
                fill=0,
                default_value=1,
                dtype='uint8')

        flow[mask == 1] = -1

    # noout = float(parameter('input.noout'))
    # with rio.open(config.tileset().filename('tiled', row=row, col=col)) as ds2:
    #     flow[ds2.read(1) == noout] = -1

    # profile = ds.profile.copy()
    flow = flow[1:-1, 1:-1]
    height, width = flow.shape
    transform = transform * transform.translation(1, 1)
    profile.update(
        height=height, width=width,
        dtype=np.int16,
        nodata=-1,
        transform=transform)

    with rio.open(output, 'w', **profile) as dst:
        dst.write(flow, 1)
