#!/usr/bin/env python
# coding: utf-8

"""
SubGrid Procedures

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
import itertools
from collections import defaultdict
from operator import itemgetter
from multiprocessing import Pool
import math

import numpy as np

import click
from affine import Affine
import rasterio as rio
from rasterio.windows import Window
import fiona
from shapely.geometry import asShape

from .. import transform as fct
from .. import terrain_analysis as ta
from .. import speedup
from ..cli import starcall
from ..config import config
from ..tileio import as_window

def DefineSubGridMask():
    """
    Using np.ma conventions:
    True means masked/nodata
    """

    # subgrid_shapefile = os.path.join(workdir, 'SUBGRID', 'SubGrid200m.shp')
    # output = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')

    tileset = config.tileset('subgrid')
    subgrid = tileset.tileindex
    # height = tileset.height
    # width = tileset.width
    output = config.filename('subgrid_mask')
    resolution = 200.0

    # with fiona.open(subgrid_shapefile) as fs:
    # for tile in subgrid.values():

    x0, y1, x1, y0 = tileset.bounds
    x0 = math.floor(x0 / resolution) * resolution
    x1 = math.ceil(x1 / resolution) * resolution
    y0 = math.ceil(y0 / resolution) * resolution
    y1 = math.floor(y1 / resolution) * resolution
    height = int((y0 - y1) // resolution)
    width = int((x1- x0) // resolution)
    transform = Affine.from_gdal(x0, resolution, 0.0, y0, 0.0, -resolution)

    mask = np.full((height, width), True)

    def intile(i, j):
        return all([
            i >= 0,
            i < height,
            j >= 0,
            j < width
        ])

    with click.progressbar(subgrid.values(), length=len(subgrid)) as iterator:
        for tile in iterator:

            # properties = feature['properties']
            # cx = 0.5 * (properties['right'] + properties['left'])
            # cy = 0.5 * (properties['top'] + properties['bottom'])
            left, bottom, right, top = tile.bounds
            cx = 0.5 * (left + right)
            cy = 0.5 * (top + bottom)
            i, j = fct.index(cx, cy, transform)
            
            assert(intile(i, j))
            
            mask[i, j] = False

        profile = dict(
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='uint8',
            compress='deflate',
            transform=transform,
            crs=rio.crs.CRS.from_epsg(config.srid)
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(np.uint8(mask), 1)

        click.echo('Dimensions:   %d x %d' % (height, width))
        click.echo('GeoTransform: %s' % list(transform.to_gdal()))
        click.echo('CRS:          %s' % 'EPSG:2154')
        click.echo('Data:         %d' % np.sum(~mask))
        click.echo('Nodata:       %d' % np.sum(mask))

def AggregateCell(i, j, datafile, window, fun, nodata):

    with rio.open(datafile) as datas:

        data = datas.read(1, window=window, boundless=True, fill_value=datas.nodata)
        return i, j, fun(data, datas, nodata)

def AggregateMetric(datafile, output, fun, dtype, nodata, processes, **kwargs):

    mask_file = config.filename('subgrid_mask')

    with rio.open(mask_file) as ds:

        mask = ds.read(1)
        height, width = mask.shape

        out = np.zeros_like(mask, dtype=dtype)

        with rio.open(datafile) as datas:

            resolution = int(ds.transform.a // datas.transform.a)
            half_resolution = int(resolution // 2)
            arguments = list()

            for i in range(height):
                for j in range(width):

                    if mask[i, j] == 0:

                        x, y = ds.xy(i, j)
                        di, dj = datas.index(x, y)

                        window = Window(
                            dj - half_resolution,
                            di - half_resolution,
                            resolution,
                            resolution)

                        arguments.append((AggregateCell, i, j, datafile, window, fun, nodata, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)

            with click.progressbar(pooled, length=len(arguments)) as iterator:
                for i, j, value in iterator:
                    out[i, j] = value

        out[mask == 1] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype=dtype,
            nodata=nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def aggregate_sum(data, ds, nodata):

    masked = np.ma.masked_array(data, data == ds.nodata)
    value = np.ma.sum(masked)

    if value is np.ma.masked:
        return nodata

    return value

def AggregatePopulation(processes=1):

    # datafile = os.path.join(workdir, 'GLOBAL', 'POP_2015.vrt')
    # output = os.path.join(workdir, 'SUBGRID', 'POP_2015.tif')

    datafile = config.filename('population')
    output = config.filename('subgrid_population')

    AggregateMetric(
        datafile,
        output,
        aggregate_sum,
        'uint32',
        np.iinfo('uint32').max,
        processes)

def AggregateLandCoverCell(i, j, datafile, window, n):

    with rio.open(datafile) as datas:

        data = datas.read(1, window=window, boundless=True, fill_value=datas.nodata)
        return i, j, speedup.count_by_uint8(data, datas.nodata, n)

def AggregateLandCover(processes=1, dataset='landcover', **kwargs):

    # mask_file = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')
    # datafile = os.path.join(workdir, 'GLOBAL', 'LANDCOVER_2018.vrt')
    # output = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018.tif')

    mask_file = config.filename('subgrid_mask')
    datafile = config.filename(dataset)
    output = config.filename('subgrid_landcover')

    dtype = 'float32'
    nodata = -99999.0
    classes = 9

    with rio.open(mask_file) as ds:

        mask = ds.read(1)
        height, width = mask.shape

        out = np.zeros((classes, height, width), dtype=dtype)

        with rio.open(datafile) as datas:

            resolution = int(ds.transform.a // datas.transform.a)
            pixel_area = 1 / resolution**2
            half_resolution = int(resolution // 2)
            arguments = list()

            for i in range(height):
                for j in range(width):

                    if mask[i, j] == 0:

                        x, y = ds.xy(i, j)
                        di, dj = datas.index(x, y)

                        window = Window(
                            dj - half_resolution,
                            di - half_resolution,
                            resolution,
                            resolution)

                        arguments.append((AggregateLandCoverCell, i, j, datafile, window, classes, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)

            with click.progressbar(pooled, length=len(arguments)) as iterator:
                for i, j, counts in iterator:
                    out[:, i, j] = counts[:-1] * pixel_area

        out[:, mask == 1] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype=dtype,
            nodata=nodata,
            compress='deflate',
            count=classes
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out)

def DominantLandCover():

    # landcover_file = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018.tif')
    # mask_file = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')
    # output = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_DOMINANT.tif')

    mask_file = config.filename('subgrid_mask')
    landcover_file = config.filename('subgrid_landcover')
    output = config.filename('subgrid_dominant_landcover')

    with rio.open(landcover_file) as ds:

        landcover = ds.read()
        # landcover[landcover == ds.nodata] = 0

        dominant = np.argmax(landcover, axis=0)

        with rio.open(mask_file) as mskds:

            mask = mskds.read(1)
            dominant[mask == 1] = 255

            profile = ds.profile.copy()
            profile.update(
                count=1,
                dtype='uint8',
                nodata=255,
                compress='deflate')

            with rio.open(output, 'w', **profile) as dst:
                dst.write(np.uint8(dominant), 1)



def CreateBufferMask(axis, distance):

    subgrid_raster = os.path.join(
        workdir,
        'SUBGRID',
        'SUBGRID_MASK.tif')

    outlet_shapefile = os.path.join(
        workdir,
        'SUBGRID',
        'SUBGRID_OUTLETS.shp')

    buffer_raster = os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'BUFFER%.0f.vrt' % distance)

    output = os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'SUBGRID_MASK_BUFFER%.0f.tif' % distance)

    with rio.open(subgrid_raster) as ds:

        mask = np.zeros((ds.height, ds.width), dtype='uint8')
        profile = ds.profile.copy()

        with rio.open(buffer_raster) as src:
            with fiona.open(outlet_shapefile) as fs:
                with click.progressbar(fs.filter(bbox=src.bounds)) as iterator:
                    for feature in iterator:

                        i = feature['properties']['i']
                        j = feature['properties']['j']
                        x, y = feature['geometry']['coordinates']
                        buf_value = next(src.sample([(x, y)], 1))
                        
                        if buf_value > 1:
                            mask[i, j] = 1

        with rio.open(output, 'w', **profile) as dst:
            dst.write(mask, 1)

    return mask

def AccumulatePopulation(graph, axis=None, mask=None, dataset=''):
    """
    DOCME
    """
    
    population_raster = os.path.join(workdir, 'SUBGRID', 'POP_2015.tif')
    
    if axis is not None:
        output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, dataset, 'POP_2015_ACC.tif')
    else:
        output = os.path.join(workdir, 'SUBGRID', 'POP_2015_ACC.tif')
    
    nodata = -99999.0

    with rio.open(population_raster) as ds:

        population = ds.read(1)
        nodata_mask = (population == ds.nodata)
        population = np.float32(population)
        
        if mask is None:
            population[nodata_mask] = 0.0
        else:
            population[nodata_mask | mask] = 0.0
        
        out = np.zeros_like(population)
        
        speedup.raster_acc(population, graph, out=out, coeff=0.001)
        out[nodata_mask] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype = 'float32',
            nodata = nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def AccumulateLandcover(graph, axis=None, mask=None, dataset=''):
    """
    DOCME
    """
    
    landcover_raster = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018.tif')

    if axis is not None:
        output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, dataset, 'LANDCOVER_2018_ACC.tif')
    else:
        output = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018_ACC.tif')

    nodata = -99999.0

    with rio.open(landcover_raster) as ds:

        landcover = ds.read()
        nodata_mask = (landcover == ds.nodata)

        click.echo('Apply mask')
        print(landcover.shape, nodata_mask.shape, mask.shape)
        
        if mask is None:
            landcover[nodata_mask] = 0.0
        else:
            # for k in range(ds.count):
            #     landcover[nodata_mask[k, :, :] | mask] = 0.0
            landcover[nodata_mask | mask[np.newaxis, :, :]] = 0.0

        click.echo('Accumulate')

        out = np.zeros_like(landcover)
        
        speedup.multiband_raster_acc(landcover, graph, out=out, coeff=0.04)
        out[nodata_mask] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype = 'float32',
            nodata = nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out)

def workflow():

    click.secho('Create SubGrid Data', fg='cyan')

    click.echo('Define SubGrid')
    DefineSubGridMask()

    click.echo('Aggregate Population')
    AggregatePopulation(7)
    click.echo('Aggregate Land Cover')
    AggregateLandCover(7)
    click.echo('Calculate Dominant Land Cover')
    DominantLandCover()

    from .SubGridGraph import workflow as mkpixgraph

    pixgraph = mkpixgraph()

    click.secho('Accumulate SubGrid', fg='cyan')
    click.echo('Accumulate Population')
    AccumulatePopulation(pixgraph)
    click.echo('Accumulate Land Cover')
    AccumulateLandcover(pixgraph)


# if __name__ == '__main__':
#     workflow()
