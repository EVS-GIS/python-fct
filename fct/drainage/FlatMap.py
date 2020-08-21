# coding: utf-8

"""
Calculate Depression Depth Map

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

import rasterio as rio
from rasterio.features import sieve

from .. import terrain_analysis as ta
from .. import speedup
from ..config import config

def DepressionDepthMap(row, col, **kwargs):
    """
    Calculate raster map
    of how much flat cells have been raised
    after DEM depression filling
    """

    # from scipy.ndimage.morphology import binary_closing

    # reference_raster = config.tileset().filename('tiled', row=row, col=col)
    filled_raster = config.tileset().tilename('dem-drainage-resolved', row=row, col=col)
    # filled_raster = config.tileset().tilename('dem-filled-resolved', row=row, col=col)

    output = config.tileset().tilename('depression-depth', row=row, col=col)
    overwrite = kwargs.get('overwrite', False)

    if os.path.exists(output) and not overwrite:
        # click.secho('Output already exists: %s' % output, fg='yellow')
        return

    reference_raster = config.tileset().tilename('dem', row=row, col=col)
    with rio.open(reference_raster) as ds:
        reference = ds.read(1)

    with rio.open(filled_raster) as ds:

        # reference = ds.read(1)
        # reference, transform, nodata = ReadRasterTile(row, col, 'dem1', 'dem2')
        filled = ds.read(1)
        
        flow = ta.flowdir(filled, ds.nodata)
        labels, _ = speedup.flat_labels(flow, filled, ds.nodata)
        depth = filled - reference

        # filled = filled - reference
        # filled[reference == ds.nodata] = ds.nodata
        
        # mask = np.uint8(flow == 0)
        # structure = np.array([
        #     [0, 0, 1, 0, 0],
        #     [0, 1, 1, 1, 0],
        #     [1, 1, 1, 1, 1],
        #     [0, 1, 1, 1, 0],
        #     [0, 0, 1, 0, 0]], dtype=np.uint8)
        # # structure = np.array([
        # #     [0, 1, 0],
        # #     [1, 1, 1],
        # #     [0, 1, 0]], dtype=np.uint8)
        # mask =  np.uint8(binary_closing(mask, structure=structure, iterations=2))
        # filled[mask == 0] = ds.nodata

        depth[(reference == ds.nodata) | (labels == 0)] = ds.nodata

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(depth, 1)

def FlatMap(row, col, min_drainage, **kwargs):
    """
    Flat areas continuous to drainage network

    Values :
        1: Flat (low topography, continuous to drainage network)
        2: Not Flat
        3: Slope/Crest Flat
        255: No-data
    """

    from scipy.ndimage.morphology import binary_closing

    dem_raster = config.tileset().tilename('filled', row=row, col=col)
    flow_raster = config.tileset().tilename('flow', row=row, col=col)
    acc_raster = config.tileset().tilename('acc', row=row, col=col)
    output = config.tileset().tilename('flatmap', row=row, col=col)

    with rio.open(dem_raster) as ds:

        flow = ta.flowdir(ds.read(1), ds.nodata)
        flats = np.uint8(flow == 0)
        del flow
        
        # Sieve/Morphological Closing

        # structure = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=np.uint8)
        structure = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]], dtype=np.uint8)
        flats =  np.uint8(binary_closing(flats, structure=structure, iterations=2))
        flats = sieve(flats, 800)

        # Continuity with stream network derived from acc

        method = 1

        if method == 1:

            # Method 1 Watershed max

            # 1 = flat, 2 = not flat
            flats = 2 - np.float32(flats)

            with rio.open(acc_raster) as ds2:

                mask = (ds2.read(1) >= min_drainage)
                out = np.zeros_like(mask, dtype=np.float32)
                out[mask] = 1
                # flats[mask] = 1

            with rio.open(flow_raster) as ds2:
                flow = ds2.read(1)

            ta.watershed_max(flow, out, flats, fill_value=0, feedback=None)

            out = np.uint8(out)
            out[mask] = flats[mask]

            out[out == 2] = 3
            out[(flats == 1) & (out == 3)] = 2
            out[flow == -1] = 255

        elif method == 2:

            # Method 2 Shortest Max

            # 1 = stream, 2 = flat, 3 = not flat
            flats = 3 - np.float32(flats)

            with rio.open(acc_raster) as ds2:

                mask = (ds2.read(1) >= min_drainage)
                out = np.zeros_like(mask, dtype=np.float32)
                # out[mask] = 1
                flats[mask] = 1

            ta.shortest_max(flats, 0, 1, out=out, feedback=ta.ConsoleFeedback())

            out = np.uint8(out) - 1
            out[mask] = flats[mask] - 1

            out[out == 2] = 3
            out[(flats == 2) & (out == 3)] = 2

        # End Method Options

        speedup.spread_connected(out, 1)

        profile = ds.profile.copy()
        profile.update(compress='deflate', dtype=np.uint8, nodata=255)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)
