# coding: utf-8

"""
Accumulate values from metric raster
according to D8 flow raster.

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
import itertools
from operator import itemgetter

from multiprocessing import Pool
import numpy as np
import xarray as xr

import click
import rasterio as rio
import fiona

from .. import transform as fct
from .. import speedup
from ..tileio import PadRaster
from ..cli import starcall
from ..config import config
from ..drainage.ValleyBottom import border
from .BufferDistance import BufferMeasure

def SampleWatershedsTile(tileset, axis, row, col, samples):
    
    # distance_raster = config.tileset(tileset).tilename(
    #     'ax_buffer_distance',
    #     row=row,
    #     col=col,
    #     axis=axis)

    output = config.tileset(tileset).tilename(
        'ax_subgrid_watershed',
        row=row,
        col=col,
        axis=axis)

    flow, profile = PadRaster(
        row,
        col,
        dataset='flow',
        tileset='landcover',
        padding=1)

    height, width = flow.shape
    nodata = profile['nodata']
    transform = profile['transform']

    if os.path.exists(output):
        with rio.open(output) as ds:
            control = ds.read(1)
    else:
        control = np.zeros_like(flow, dtype='uint32')

    watersheds = np.copy(control)

    def intile(i, j):

        return all([
            i >= 0,
            i < height,
            j >= 0,
            j < width
        ])

    for gid, x, y in samples:

        i, j = fct.index(x, y, transform)
        if intile(i, j):
            watersheds[i, j] = gid

    speedup.watershed(flow, watersheds)

    distance, dist_profile = PadRaster(
        row,
        col,
        axis=axis,
        dataset='ax_buffer_distance',
        tileset='landcover',
        padding=1)

    watersheds[(flow == nodata) | (distance == dist_profile['nodata'])] = 0

    del distance
    del flow

    profile.update(
        dtype='uint32',
        nodata=0,
        compress='deflate'
    )

    output += '.tmp'

    with rio.open(output, 'w', **profile) as dst:
        dst.write(watersheds, 1)

    spillovers = list()

    def attribute(pixel):
        """
        Returns value, dest. row, dest. col for pixel (i, j)
        """

        i, j = pixel
        x, y = fct.xy(i, j, transform)

        if i == 0:
            prow = row - 1
        elif i == height-1:
            prow = row + 1
        else:
            prow = row

        if j == 0:
            pcol = col - 1
        elif j == width-1:
            pcol = col + 1
        else:
            pcol = col

        return watersheds[pixel], x, y, prow, pcol

    for pixel in border(height, width):
        if control[pixel] <= 0 < watersheds[pixel]:
            spillovers.append(attribute(pixel))

    # print(output)
    return spillovers, output

def SampleWatershedsIteration(axis, spillovers, processes=1, **kwargs):

    attributes = itemgetter(0, 1, 2)
    tile = itemgetter(3, 4)
    spillovers = sorted(spillovers, key=tile)
    tiles = itertools.groupby(spillovers, key=tile)

    g_spillover = list()
    tmpfiles = list()

    def arguments():

        for (row, col), samples in tiles:
            samples = [attributes(sample) for sample in samples]
            yield (SampleWatershedsTile, 'landcover', axis, row, col, samples, kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        for t_spillover, output in pooled:
            g_spillover.extend(t_spillover)
            tmpfiles.append(output)

    for tmpfile in tmpfiles:
        os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    return g_spillover

def CropRasterTile(axis, row, col, tileset='default', dataset='ax_subgrid_watershed', padding=1):

    rasterfile = config.tileset(tileset).tilename(
        dataset,
        row=row,
        col=col,
        axis=axis)

    with rio.open(rasterfile) as ds:

        profile = ds.profile.copy()
        transform = ds.transform * ds.transform.translation(padding, padding)
        height = ds.height - 2*padding
        width = ds.width - 2*padding
        data = ds.read(1)

    profile.update(
        transform=transform,
        height=height,
        width=width,
        compress='deflate')

    with rio.open(rasterfile, 'w', **profile) as dst:
        dst.write(data[padding:-padding, padding:-padding], 1)

def CropRasterTiles(axis, tiles, processes=1, **kwargs):

    def arguments():

        for row, col in tiles:
            yield (CropRasterTile, axis, row, col, kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        
        with click.progressbar(pooled, length=len(tiles)) as iterator:
            for _ in iterator:
                pass

def DefineSampleWatersheds(axis, processes=1):

    sample_shapefile = config.filename('ax_subgrid_profile', axis=axis)

    with fiona.open(sample_shapefile) as fs:

        def as_sample(feature):

            gid = feature['properties']['GID']
            row = feature['properties']['ROW']
            col = feature['properties']['COL']
            x, y = feature['geometry']['coordinates']

            return gid, x, y, row, col

        spillovers = [as_sample(feature) for feature in fs]

    tile = itemgetter(3, 4)
    iteration = 1
    g_tiles = set()

    while spillovers:

        tiles = set([tile(s) for s in spillovers])
        g_tiles.update(tiles)
        iteration += 1
        spillovers = SampleWatershedsIteration(axis, spillovers, processes)

    CropRasterTiles(
        axis,
        g_tiles,
        processes=processes,
        padding=1,
        tileset='landcover',
        dataset='ax_subgrid_watershed')

def AccumulateMetricTile(
        axis, row, col,
        tileset='default',
        dataset='population',
        band=1,
        buffer_width=1000.0,
        width_step=200.0,
        buffer_distance='ax_buffer_distance',
        buffer_profile='ax_subgrid_watershed',
        **kwargs):

    data_raster = config.tileset(tileset).tilename(
        dataset,
        row=row,
        col=col,
        **kwargs)

    distance_raster = config.tileset(tileset).tilename(
        buffer_distance,
        row=row,
        col=col,
        axis=axis,
        **kwargs)

    sample_raster = config.tileset(tileset).tilename(
        buffer_profile,
        row=row,
        col=col,
        axis=axis,
        **kwargs)

    if not all([
            os.path.exists(data_raster),
            os.path.exists(distance_raster),
            os.path.exists(sample_raster)
        ]):

        return dict()

    n = int(buffer_width // width_step)
    breaks = np.linspace(0.0, buffer_width, n+1)

    with rio.open(distance_raster) as ds:
        distance = np.uint32(np.digitize(ds.read(1), breaks))

    with rio.open(sample_raster) as ds:
        samples = ds.read(1)

    with rio.open(data_raster) as ds:

        data = ds.read(band)
        return speedup.cumulate_by_id2(data, samples, distance, ds.nodata)

def MetricArray(cumulated, m, n):

    out = np.zeros((m, n), dtype='float32')

    for i in range(m):
        for j in range(n):

            if (i+1, j+1) in cumulated:
                out[i, j] = cumulated[i+1, j+1]

    return out

def AccumulateMetric(axis, processes=1, tileset='default', **kwargs):
    """
    Accumulate raster data within buffer area,
    for each spatial unit and width bins.

    Parameters
    ----------

    axis: int

        Axis identifier

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Returns
    -------

    cumulated: array(m, n)

        Cumulated values for each pair (profile unit, width bin)

        m: number of profile units,
           ids in range {1..m} at index id-1

        n: number of width bins,
           bins in range {1..n}, at index bin-1

        Width bins are given by the formula :

            n = int(buffer_width // width_step)
            breaks = np.linspace(0.0, buffer_width, n+1)
            # i in {1..n}
            bin[i] = (breaks[i-1], breaks[i])

        Profile unit with ID=0 (nodata),
        bin 0 (nodata) and
        bin n+1 (outside buffer) are excluded.

    Keyword arguments
    -----------------

    tileset: str

        logical tileset
        defaults to `default`

    dataset: str

        logical name of
        raster dataset to process,
        defaults to 'population'

    band: int

        band to process, if raster dataset is multibanded,
        defaults to 1 (one-band dataset)

    buffer_width: float

        maximum buffer width,
        defaults to 1000.0

    width_step: float

        width step for metric output,
        defaults to 200.0

    buffer_distance: str

        logical name of
        distance raster to use,
        defaults to `ax_buffer_distance`

    buffer_profile: str

        logical name of
        longitudinal unit identifier raster

    Other keywords are passed to dataset filename templates.
    """

    kwargs.update(tileset=tileset)
    tileindex = config.tileset(tileset)

    def arguments():

        for tile in tileindex.tiles():
            yield (
                AccumulateMetricTile,
                axis,
                tile.row,
                tile.col,
                kwargs
            )

    cumulated = dict()
    m = 0
    n = 0

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileindex)) as iterator:
            for t_cumulated in iterator:

                if t_cumulated:

                    cumulated.update({
                        k: cumulated.get(k, 0) + t_cumulated[k]
                        for k in t_cumulated
                    })

                    t_m = max(m for m, n in t_cumulated)
                    t_n = max(n for m, n in t_cumulated)
                    m = max(m, t_m)
                    n = max(n, t_n)

    # Exclude last width bin, outside of buffer
    return MetricArray(cumulated, m, n-1)

def WatershedMetrics(axis, processes=1, buffer_profile='ax_buffer_profile'):

    buffer_width = 1000.0
    width_step = 200.0
    n = int(buffer_width // width_step)
    breaks = np.linspace(0.0, buffer_width, n+1)
    print(n, len(breaks))

    measures = BufferMeasure(axis, processes=processes)

    scale = 1e-3 # -> thousands
    pop = scale * AccumulateMetric(
        axis,
        processes=processes,
        tileset='landcover',
        dataset='population',
        buffer_distance='ax_buffer_distance',
        buffer_profile=buffer_profile)

    scale = 1.0
    income = scale * AccumulateMetric(
        axis,
        processes=processes,
        tileset='landcover',
        dataset='population-income',
        buffer_distance='ax_buffer_distance',
        buffer_profile=buffer_profile)

    m, n = pop.shape
    lck = np.zeros((m, n, 9), dtype='float32')
    scale = 25e-6 # 25 m² -> km²

    for k in range(0, 9):

        lck[:, :, k] = scale * AccumulateMetric(
            axis,
            processes=processes,
            tileset='landcover',
            dataset='landcover-separate',
            band=k+1,
            buffer_distance='ax_buffer_distance',
            buffer_profile=buffer_profile)

    return xr.Dataset(
        {
            'pop': (('measure', 'width'), pop),
            'income': (('measure', 'width'), income),
            'lck': (('measure', 'width', 'landcover'), lck)
        },
        coords={
            'axis': axis,
            'measure': measures,
            'width': breaks[1:],
            'landcover': [
                'Water Channel',
                'Gravel Bars',
                'Natural Open',
                'Forest',
                'Grassland',
                'Crops',
                'Diffuse Urban',
                'Dense Urban',
                'Infrastructures'
            ]
        })

def WriteWatershedMetrics(axis, data, destination='metrics_watershed', subset='buf1k'):

    # output = os.path.join(config.workdir, 'AXES', 'AX%04d' % axis, 'METRICS', 'WATERSHED_METRICS_BUF1K.nc')
    output = config.filename(destination, axis=axis, subset=subset.upper())

    data.to_netcdf(
        output, 'w',
        encoding={
            'pop': dict(zlib=True, complevel=9, least_significant_digit=2),
            'income': dict(zlib=True, complevel=9, least_significant_digit=0),
            'lck': dict(zlib=True, complevel=9, least_significant_digit=3),
            'width': dict(least_significant_digit=0),
            'measure': dict(least_significant_digit=0)
        })

# def workflow(axis):

#     config.default()

#     DefineSubGrid()
#     DefineAxisSubGridSamples()
#     RetileDatasource('flow', 'landcover', processes=6)
#     CalculateBufferDistance()
#     DefineSampleWatersheds(axis)
#     AccumulatePopulationMetrics()
#     AccumulaleLandCoverMetrics()
#     WriteWatershedMetrics()
