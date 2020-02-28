#!/usr/bin/env python
# coding: utf-8

"""
Reclassification des pentes en 4 classes
et continuité de la classe de pente par rapport au réseau hydrographique

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
import click
import numpy as np

from console import (
    success,
    info,
    important,
    warning
)

import rasterio as rio
import fiona
import fiona.crs

import terrain_analysis as ta
from rasterize import rasterize_linestring

ci = [-1, -1,  0,  1,  1,  1,  0, -1]
cj = [ 0,  1,  1,  1,  0, -1, -1, -1]

def RasterizeStream(raster, transform, nodata, shapefile, fill):
    """
    Rastérisation du réseau hydrographique cartographié.
    """

    height, width = raster.shape
    # streams = np.zeros((height, width), dtype=np.uint32)
    priorities = np.zeros((height, width), dtype=np.uint32)
    out = np.full((height, width), fill, dtype=np.float32)

    def isdata(px, py):
        """
        True if (py, px) is a valid pixel coordinate
        """

        return px >= 0 and py >= 0 and px < width and py < height

    def set_data(row, col, gid, value, priority):
        """
        Set Pixel Value to Line Primary Field
        """

        current_priority = priorities[row, col]

        if current_priority == 0 or priority < current_priority:
            # Override with the smallest ID
            # streams[row, col] = gid
            priorities[row, col] = priority
            out[row, col] = value

    with fiona.open(shapefile) as fs:

        for feature in fs:

            gid = feature['properties']['GID']
            priority = feature['properties']['HACK']
            # cdzone = feature['properties']['CDZONEHYDR']
            # cdzoneidx = cdzones[cdzone]
            geom = feature['geometry']['coordinates']
            linestring = np.fliplr(ta.worldtopixel(np.float32(geom), transform, gdal=False))

            burn_on = False

            for a, b in zip(linestring[:-1], linestring[1:]):
                for col, row in rasterize_linestring(a, b):
                    if isdata(col, row) and raster[row, col] != nodata:
                        for x in range(8):
                            ix = row + ci[x]
                            jx = col + cj[x]
                            if not isdata(jx, ix) or raster[ix, jx] == nodata:
                                # on arrive au bord
                                # on laisse burn_on à sa valeur précédente
                                # si on était à l'extérieur, on n'écrit pas le pixel
                                # si on était à l'intérieur, on écrit le pixel
                                # qui est un potentiel exutoire
                                break
                        else:
                            # on n'est pas au bord
                            burn_on = True
                        if burn_on:
                            set_data(row, col, gid, 1, priority)
                    else:
                        burn_on = False

    return out

def SlopeContinuity(bassin, zone, workdir, overwrite):
    """
    DCOME
    """

    from rasterio.features import sieve

    basename = 'SLOPE_CLS_CONTINUOUS.tif'
    flow_raster = os.path.join(workdir, bassin, zone, 'FLOW.tif')
    slope_raster = os.path.join(workdir, bassin, zone, 'SLOPE_CLS.tif')
    stream_network = os.path.join(workdir, bassin, zone, 'StreamNetwork.shp')
    output = os.path.join(workdir, bassin, zone, basename)

    if os.path.exists(output) and not overwrite:
        important('Output already exists : %s' % output)
        return

    with rio.open(slope_raster) as ds:

        slopes = sieve(ds.read(1), 800)
        streams = RasterizeStream(slopes, ds.transform, ds.nodata, stream_network, 0)

        # data = np.float32(slopes) + 1
        # data[slopes == ds.nodata] = ds.nodata
        # data[streams == 1] = 0

        # feedback = ta.ConsoleFeedback()

        # distance = np.float32(ta.shortest_distance(data, ds.nodata, 0, feedback=feedback))
        # out = np.zeros_like(slopes, dtype=np.float32)
        # data[distance < 100] = 1
        # data[streams == 1] = 0

        # feedback = ta.ConsoleFeedback()
        # ta.shortest_max(data, ds.nodata, 0, out=out, feedback=feedback)

        # out = np.uint8(out) - 1
        # out[distance < 100] = slopes[distance < 100]
        # out[slopes == ds.nodata] = ds.nodata

        out = np.zeros_like(slopes, dtype=np.float32)
        out[streams == 1] = 1

        with rio.open(flow_raster) as ds2:
            flow = ds2.read(1)

        feedback = ta.ConsoleFeedback()
        ta.watershed_max(flow, out, np.float32(slopes), fill_value=0, feedback=feedback)

        out = np.uint8(out)
        # out[streams == 1] = slopes[streams == 1]
        out[slopes == ds.nodata] = ds.nodata

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

    success('Saved result to %s' % output)

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def zone(basin, zone, workdir, overwrite):

    SlopeContinuity(basin, zone, workdir, overwrite)

@cli.command()
@click.argument('zonelist')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def batch(zonelist, workdir, overwrite):

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    def display_item(item):
        if item:
            return item[1]
        return '...'

    with click.progressbar(zones, item_show_func=display_item) as progress:
        for basin, zone in progress:

            click.echo('\r')
            SlopeContinuity(basin, zone, workdir, overwrite)

if __name__ == '__main__':
    cli()
