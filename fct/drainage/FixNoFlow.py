# coding: utf-8

"""
Fix Flow Direction for No-Flow Cells

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np

import rasterio as rio
import fiona
import click

from .. import transform as fct
from ..config import config

def DrainageRaster(row, col, min_drainage=5.0):
    """
    Rasterize back drainage network
    """

    acc_raster = config.tileset().tilename(
        'acc',
        row=row,
        col=col)

    stream_features = config.tileset().tilename(
        'streams-from-sources',
        row=row,
        col=col)

    output = config.tileset().tilename(
        'drainage-raster-from-sources',
        row=row,
        col=col)

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

def FixNoFlow(x0, y0, tileset1, tileset2, min_drainage=5.0, fix=False):
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
    flow_raster1 = config.tileset(tileset1).filename('flow')
    acc_raster1 = config.tileset(tileset1).filename('acc')
    stream_raster1 = config.tileset(tileset1).filename('drainage-raster-from-sources')
    flow1ds = rio.open(flow_raster1)
    acc1ds = rio.open(acc_raster1)
    stream1ds = rio.open(stream_raster1)

    # flow_raster2 = '/var/local/fct/RMC/FLOW_RGE5M_TILES2.vrt'
    # acc_raster2 = '/var/local/fct/RMC/ACC_RGE5M_TILES2.vrt'
    flow_raster2 = config.tileset(tileset2).filename('flow')
    # acc_raster2 = config.tileset(tileset2).filename('acc')
    stream_raster2 = config.tileset(tileset2).filename('drainage-raster-from-sources')
    flow2ds = rio.open(flow_raster2)
    # acc2ds = rio.open(acc_raster2)
    stream2ds = rio.open(stream_raster2)

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

        flow_raster1 = config.tileset(tileset1).tilename('flow', row=row, col=col)
        acc_raster1 = config.tileset(tileset1).tilename('acc', row=row, col=col)
        stream_raster1 = config.tileset(tileset1).tilename(
            'drainage-raster-from-sources',
            row=row,
            col=col)

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

            flow_raster1 = config.tileset().tilename('flow', row=row, col=col)

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
            read_tile(row, col)
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
            read_tile(row, col)
            i, j = ds.index(x, y)

    write_tile(row, col)

    flow1ds.close()
    acc1ds.close()
    stream1ds.close()
    flow2ds.close()
    # acc2ds.close()
    stream2ds.close()

    return ds.xy(i, j)

def test(tileset1='10k', tileset2='10kbis', fix=False):
    """
    TODO finalize
    DOCME
    """

    # noflow = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/NOFLOW_RGE5M_ALL.shp'
    # targets = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/NOFLOW_TARGETS.shp'

    # noflow = config.tileset(tileset1).filename('noflow')
    # targets = config.tileset(tileset1).filename('noflow-targets')

    config.default()
    noflow = config.tileset(tileset1).filename('noflow-from-sources')
    targets = config.tileset(tileset1).filename('noflow-targets-from-sources')

    with fiona.open(noflow) as fs:

        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

        with fiona.open(targets, 'w', **options) as dst:
            with click.progressbar(fs) as progress:

                for f in progress:

                    x, y = f['geometry']['coordinates']

                    try:

                        tox, toy = FixNoFlow(x, y, tileset1, tileset2, fix=fix)
                        f['geometry']['coordinates'] = [tox, toy]
                        dst.write(f)

                    except ValueError as error:

                        print(f['properties']['GID'], error)
                        continue
