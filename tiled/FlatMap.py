# coding: utf-8

import os
from collections import defaultdict, Counter
import numpy as np

import click
import rasterio as rio
from rasterio.features import sieve
import terrain_analysis as ta
import speedup

from config import tileindex, filename

def FlatDepth(row, col, **kwargs):
    """
    Calculate raster map
    of how much flat cells have been raised
    after DEM depression filling
    """

    reference_raster = filename('dem', row=row, col=col)
    filled_raster = filename('prefilled', row=row, col=col)
    output = filename('flats', row=row, col=col)
    overwrite = kwargs.get('overwrite', False)

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    with rio.open(reference_raster) as ds:

        reference = ds.read(1)

        with rio.open(filled_raster) as ds2:
            filled = ds2.read(1)

        filled = filled - reference
        filled[reference == ds.nodata] = ds.nodata

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(filled, 1)

def FlatMap(row, col, min_drainage, **kwargs):
    """
    Flat areas continuous to drainage network

    Values :
        1: Flat (low topography, continuous to drainage network)
        2: Not Flat
        3: Slope/Crest Flat
        255: No-data
    """

    dem_raster = filename('prefilled', row=row, col=col)
    flow_raster = filename('flow', row=row, col=col)
    acc_raster = filename('acc', row=row, col=col)
    output = filename('flatmap', row=row, col=col)

    with rio.open(dem_raster) as ds:

        flow = ta.flowdir(ds.read(1), ds.nodata)
        flats = np.uint8(flow == 0)
        del flow
        
        # sieve/closing

        sieve(flats, 400)

        # 1 = flat, 2 = not flat
        flats = 2 - np.float32(flats)
        
        # continuity with stream network derived from acc

        with rio.open(acc_raster) as ds2:

            mask = (ds2.read(1) >= min_drainage)
            out = np.zeros_like(mask, dtype=np.float32)
            out[mask] = 1
            flats[mask] = 1
            del mask

        with rio.open(flow_raster) as ds2:
            flow = ds2.read(1)

        ta.watershed_max(flow, out, flats, fill_value=0, feedback=None)
        out = np.uint8(out)
        out[(flats == 1.0) & (out == 2)] = 3
        out[flow == -1] = 255

        profile = ds.profile.copy()
        profile.update(compress='deflate', dtype=np.uint8, nodata=255)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)
