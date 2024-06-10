"""
Fix Flow Direction for No-Flow pixels

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
import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from .. import transform as fct
from .. import speedup
from .. import terrain_analysis as ta

from ..config import (
    config,
    DatasetParameter,
    LiteralParameter
)

from multiprocessing import Pool
from ..cli import starcall_nokwargs

class Parameters():
    """
    Resolve no-flow pixels parameters
    """

    flow = DatasetParameter(
        'flow direction raster',
        type='input')

    acc = DatasetParameter(
        'accumulation raster (drainage area)',
        type='input')

    drainage_network = DatasetParameter(
        'drainage network shapefile',
        type='input')

    drainage_raster = DatasetParameter(
        'drainage network raster',
        type='output')

    noflow = DatasetParameter(
        'no-flow pixels (point) shapefile',
        type='output')

    fixed = DatasetParameter(
        'no-flow destination pixels (point) shapefile',
        type='output')

    min_drainage = LiteralParameter(
        'minimum drainage area for stream extraction expressed in square kilometers')

    def __init__(self):
        """
        Default paramater values
        """

        self.flow = 'flow'
        self.acc = 'acc'
        self.drainage_network = 'streams-from-sources' # 'dem-drainage-network'
        self.drainage_raster = 'drainage-raster-from-sources'
        self.noflow = 'noflow'
        self.fixed = 'noflow-targets-from-sources'
        self.min_drainage = 5.0

def DrainageRasterTile(row, col, params, tileset='default'):
    """
    Rasterize back drainage network
    """

    acc_raster = params.acc.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename(
    #     'acc',
    #     row=row,
    #     col=col)

    stream_features = params.drainage_network.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename(
    #     'streams-from-sources',
    #     row=row,
    #     col=col)

    output = params.drainage_raster.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename(
    #     'drainage-raster-from-sources',
    #     row=row,
    #     col=col)

    min_drainage = params.min_drainage

    if not os.path.isfile(stream_features):
        return 
    
    with rio.open(acc_raster) as ds:

        acc = ds.read(1)

        if min_drainage > 0:
            streams = np.int16(acc > min_drainage)
        else:
            streams = np.zeros_like(acc, dtype='int16')

        streams[acc == ds.nodata] = -1
        height, width = streams.shape

        profile = ds.profile.copy()

    with fiona.open(stream_features) as fs:
        for feature in fs:

            coordinates = np.array(feature['geometry']['coordinates'], dtype='float32')
            pixels = fct.worldtopixel(coordinates, ds.transform)

            for i, j in pixels:
                if 0 <= i < height and 0 <= j < width:
                    streams[i, j] = 1

    profile.update(
        dtype='int16',
        nodata=-1)

    with rio.open(output, 'w', **profile) as dst:
        dst.write(streams, 1)

def NoFlowPixelsTile(row, col, params, tileset='default'):

    flow_raster = params.flow.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('flow', row=row, col=col)
    acc_raster = params.acc.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('acc', row=row, col=col)
    output = params.noflow.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('noflow', row=row, col=col)

    min_drainage = params.min_drainage

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('GID', 'int'),
            ('ROW', 'int:4'),
            ('COL', 'int:4')
        ]
    }
    crs = fiona.crs.from_epsg(config.srid)
    options = dict(driver=driver, crs=crs, schema=schema)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)

        with rio.open(acc_raster) as ds2:
            streams = np.int16(ds2.read(1) > min_drainage)

        with fiona.open(output, 'w', **options) as dst:

            pixels = speedup.noflow(streams, flow)

            if pixels:

                coordinates = ta.pixeltoworld(
                    np.int32(pixels),
                    ds.transform,
                    gdal=False)

                for current, point in enumerate(coordinates):
                    dst.write({
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': point},
                        'properties': {'GID': current, 'ROW': row, 'COL': col}
                    })

def AggregateNoFlowPixels(params, tileset='default'):
    """
    Aggregate No Flow Shapefiles
    """

    # tile_index = tileindex()
    tile_index = config.tileset(tileset)
    output = params.noflow.filename(tileset=tileset)
    # config.tileset().filename('noflow')

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('GID', 'int'),
            ('ROW', 'int'),
            ('COL', 'int')
        ]
    }
    crs = fiona.crs.from_epsg(config.srid)
    options = dict(driver=driver, crs=crs, schema=schema)

    gid = itertools.count(1)

    with fiona.open(output, 'w', **options) as dst:
        with click.progressbar(tile_index.tiles(), length=len(tile_index)) as iterator:
            for tile in iterator:
                row = tile.row
                col = tile.col
                
                with fiona.open(config.tileset(tileset).tilename('noflow', row=row, col=col)) as fs:
                    for feature in fs:
                        feature['properties']['GID'] = next(gid)
                        dst.write(feature)

    count = next(gid) - 1
    click.secho('Found %d not-flowing stream nodes' % count, fg='cyan')

def FixNoFlowPoint(x0, y0, tileset1, tileset2, params, fix=False):
    """
    DOCME
    """

    ci = [ -1, -1,  0,  1,  1,  1,  0, -1 ]
    cj = [  0,  1,  1,  1,  0, -1, -1, -1 ]
    upward = [ 16,  32,  64,  128,  1,  2,  4,  8 ]

    flow1_data = acc1_data = stream1_data = np.array(0)
    height = width = 0
    ds = None
    profile = dict()

    # flow_raster1 = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'
    # acc_raster1 = '/var/local/fct/RMC/ACC_RGE5M_TILES.vrt'
    flow_raster1 = params.flow.filename(tileset=tileset1)
    # config.tileset(tileset1).filename('flow')
    acc_raster1 = params.acc.filename(tileset=tileset1)
    # config.tileset(tileset1).filename('acc')
    stream_raster1 = params.drainage_raster.filename(tileset=tileset1)
    # config.tileset(tileset1).filename('drainage-raster-from-sources')
    
    flow1ds = rio.open(flow_raster1)
    acc1ds = rio.open(acc_raster1)
    stream1ds = rio.open(stream_raster1)

    # flow_raster2 = '/var/local/fct/RMC/FLOW_RGE5M_TILES2.vrt'
    # acc_raster2 = '/var/local/fct/RMC/ACC_RGE5M_TILES2.vrt'
    flow_raster2 = params.flow.filename(tileset=tileset2)
    # config.tileset(tileset2).filename('flow')
    # acc_raster2 = config.tileset(tileset2).filename('acc')
    stream_raster2 = params.drainage_raster.filename(tileset=tileset2)
    # config.tileset(tileset2).filename('drainage-raster-from-sources')
    flow2ds = rio.open(flow_raster2)
    # acc2ds = rio.open(acc_raster2)
    stream2ds = rio.open(stream_raster2)

    min_drainage = params.min_drainage

    def read_tile(row, col):
        """
        DOCME
        """

        nonlocal ds
        nonlocal profile
        nonlocal flow1_data
        nonlocal acc1_data
        nonlocal stream1_data
        nonlocal height
        nonlocal width

        # click.echo('Reading tile (%d, %d)' % (row, col))

        flow_raster1 = params.flow.tilename(tileset=tileset1, row=row, col=col)
        # config.tileset(tileset1).tilename('flow', row=row, col=col)
        acc_raster1 = params.acc.tilename(tileset=tileset1, row=row, col=col)
        # config.tileset(tileset1).tilename('acc', row=row, col=col)
        stream_raster1 = params.drainage_raster.tilename(tileset=tileset1, row=row, col=col)
        # config.tileset(tileset1).tilename(
        #     'drainage-raster-from-sources',
        #     row=row,
        #     col=col)

        with rio.open(acc_raster1) as ds:
            acc1_data = ds.read(1)

        with rio.open(stream_raster1) as ds:
            stream1_data = ds.read(1)

        with rio.open(flow_raster1) as ds:
            profile = ds.profile.copy()
            flow1_data = ds.read(1)
            height, width = flow1_data.shape

    def intile(i, j):
        return 0 <= i < height and 0 <= j < width

    def flow1(i, j):

        if intile(i, j):
            return flow1_data[i, j]

        x, y = ds.xy(i, j)
        return int(next(flow1ds.sample([(x, y)], 1)))

    def acc1(i, j):

        if intile(i, j):
            return acc1_data[i, j]

        x, y = ds.xy(i, j)
        return float(next(acc1ds.sample([(x, y)], 1)))

    # def streams1(i, j):
    #     return acc1(i, j) > min_drainage

    def streams1(i, j):

        if intile(i, j):
            return stream1_data[i, j] == 1

        x, y = ds.xy(i, j)
        return int(next(stream1ds.sample([(x, y)], 1))) == 1

    def flow2(i, j):
        x, y = ds.xy(i, j)
        return int(next(flow2ds.sample([(x, y)], 1)))

    # def streams2(i, j):
    #     x, y = ds.xy(i, j)
    #     return bool(next(acc2ds.sample([(x, y)], 1)) > min_drainage)

    def streams2(i, j):
        x, y = ds.xy(i, j)
        return int(next(stream2ds.sample([(x, y)], 1))) == 1

    if fix:

        def write_tile(row, col):

            flow_raster1 = params.flow.tilename(row=row, col=col, tileset=tileset1)
            # config.tileset().tilename('flow', row=row, col=col)

            with rio.open(flow_raster1, 'w', **profile) as dst:
                dst.write(flow1_data, 1)

    else:

        def write_tile(row, col): #pylint:disable=unused-argument
            """ no-op """

    # row, col = xy2tile(x0, y0)
    row, col = config.tileset(tileset1).index(x0, y0)
    read_tile(row, col)
    i, j = ds.index(x0, y0)

    # step 1. walk upstream on drainage 1
    #         until we reach a common point
    #         between drainage 1 and drainage 2

    while not streams2(i, j):

        # TODO handle confluence, select max upstream acc

        max_acck = min_drainage
        max_ijk = None

        for k in range(8):

            ik, jk = i + ci[k], j + cj[k]

            if flow1(ik, jk) == upward[k]:
                acck = acc1(ik, jk)
                if acck > max_acck:
                    max_acck = acck
                    max_ijk = (ik, jk)

        if max_ijk is not None:

            i, j = max_ijk

            if not intile(i, j):
                x, y = ds.xy(i, j)
                # row, col = xy2tile(x, y)
                row, col = config.tileset(tileset1).index(x, y)
                read_tile(row, col)
                i, j = ds.index(x, y)

        else:

            raise ValueError('No match for (%f, %f)' % (x0, y0))

    # step2. walk downstream on drainage 2
    #        until we leave drainage 1

    while streams1(i, j):

        direction = flow2(i, j)
        if direction in (-1, 0):
            break

        flow1_data[i, j] = direction
        k = int(np.log2(direction))
        i, j = i + ci[k], j + cj[k]

        if not intile(i, j):

            write_tile(row, col)
            x, y = ds.xy(i, j)
            # row, col = xy2tile(x, y)
            row, col = config.tileset(tileset1).index(x, y)

            try:
                read_tile(row, col)
            except rio.RasterioIOError:
                raise ValueError('No match for (%f, %f)' % (x0, y0))

            i, j = ds.index(x, y)

    # step3. walk downstream on drainage 2
    #        until we get back on drainage 1 ;
    #        update drainage 1 to reflect drainage 2
    #        as we walk downstream

    while not streams1(i, j):

        direction = flow2(i, j)
        if direction in (-1, 0):
            break

        flow1_data[i, j] = direction
        k = int(np.log2(direction))
        i, j = i + ci[k], j + cj[k]

        if not intile(i, j):

            write_tile(row, col)
            x, y = ds.xy(i, j)
            # row, col = xy2tile(x, y)
            row, col = config.tileset(tileset1).index(x, y)

            try:
                read_tile(row, col)
            except rio.RasterioIOError:
                raise ValueError('No match for (%f, %f)' % (x0, y0))
            
            i, j = ds.index(x, y)

    write_tile(row, col)

    flow1ds.close()
    acc1ds.close()
    stream1ds.close()
    flow2ds.close()
    # acc2ds.close()
    stream2ds.close()

    return ds.xy(i, j)

def FixNoFlow(params, tileset1='10k', tileset2='10kbis', fix=False):
    """
    TODO finalize
    DOCME
    """

    # noflow = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/NOFLOW_RGE5M_ALL.shp'
    # targets = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/NOFLOW_TARGETS.shp'

    # noflow = config.tileset(tileset1).filename('noflow')
    # targets = config.tileset(tileset1).filename('noflow-targets')

    noflow = params.noflow.filename(tileset=tileset1)
    # config.tileset(tileset1).filename('noflow-from-sources')
    targets = params.fixed.filename(tileset=tileset1)
    # config.tileset(tileset1).filename('noflow-targets-from-sources')

    with fiona.open(noflow) as fs:

        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

        with fiona.open(targets, 'w', **options) as dst:
            with click.progressbar(fs) as progress:

                for f in progress:

                    x, y = f['geometry']['coordinates']

                    try:

                        tox, toy = FixNoFlowPoint(x, y, tileset1, tileset2, params, fix=fix)
                        f['geometry']['coordinates'] = [tox, toy]
                        dst.write(f)

                    except ValueError as error:

                        print(f['properties']['GID'], error)
                        continue


def DrainageRaster(params, tileset='default', processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                DrainageRasterTile,
                row,
                col,
                params,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
            
def NoFlowPixels(params, tileset='default', processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                NoFlowPixelsTile,
                row,
                col,
                params,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass