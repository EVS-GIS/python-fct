#!/usr/bin/env python
# coding: utf-8

"""
Désagrège les données carroyées de l'INSEE
à la résolution du raster d'occupation du sol,
en utilisant la surface urbanisée.

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
import rasterio as rio
from rasterio.features import rasterize
import fiona

def grid_extent(geometry, ds):

    mini = minj = maxi = maxj = -1

    for k in range(len(geometry)):

        i, j = ds.index(geometry[k, 0], geometry[k, 1])

        if i < mini or mini == -1:
            mini = i
        if i > maxi:
            maxi = i
        if j < minj or minj == -1:
            minj = j
        if j > maxj:
            maxj = j

    return mini, minj, maxi, maxj

@click.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--root', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def RasterizePopInsee(basin, zone, root, overwrite):
    """
    Désagrège les données carroyées de l'INSEE
    à la résolution du raster d'occupation du sol,
    en utilisant la surface urbanisée.
    """

    landcover_raster = os.path.join(root, basin, zone, 'LANDCOVER5M.tif')
    zone_shapefile = os.path.join(root, basin, zone, 'ZONEHYDRO_BDC.shp')
    pop_shapefile = os.path.join(root, basin, zone, 'POP_INSEE_200m.shp')
    output = os.path.join(root, basin, zone, 'POP_INSEE_5M.tif')

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    with rio.open(landcover_raster) as ds:

        zone_mask = np.zeros((ds.height, ds.width), dtype=np.int32)

        def shapes():
            """
            Enumerate polygons for rasterization
            """
            with fiona.open(zone_shapefile) as fs:
                for feature in fs:
                    geom = feature['geometry']
                    yield geom, 1

        rasterize(shapes(), out=zone_mask, transform=ds.transform, fill=0)

        landcover = ds.read(1)
        mask = (zone_mask == 1) & (landcover == 60)
        del landcover

        out = np.zeros((ds.height, ds.width), dtype=np.int32)

        def isdata(i, j):
            """
            Check bounds for pixel (i, j)
            """
            return i >= 0 and i < ds.height and j >= 0 and j < ds.width

        with fiona.open(pop_shapefile) as fs:

            value_total = 0
            feature_mask = np.zeros((ds.height, ds.width), dtype=np.int32)
            urban = 0
            non_urban = 0

            with click.progressbar(fs) as progress:
                for feature in progress:

                    geom = feature['geometry']
                    value = int(feature['properties']['IND'])

                    # Cythonize --->

                    rasterize([(geom, 1)], out=feature_mask, transform=ds.transform, fill=0)

                    mini, minj, maxi, maxj = grid_extent(np.float32(geom['coordinates'][0]), ds)
                    current_mask = mask & (feature_mask == 1)

                    if np.sum(current_mask) == 0:
                    
                        current_mask = zone_mask & (feature_mask == 1)
                        non_urban += 1
                        if np.sum(current_mask) == 0:
                            click.secho('Invalid cell %s' % feature['properties']['GEOHASH'], fg='red')
                            value_total += value
                            continue
                    
                    else:
                    
                        urban += 1

                    count = 0

                    while count < value:

                        for ri, rj in zip(
                                np.random.randint(low=mini, high=maxi+1, size=100),
                                np.random.randint(low=minj, high=maxj+1, size=100)):

                            if isdata(ri, rj) and current_mask[ri, rj]:
                                out[ri, rj] = out[ri, rj] + 1
                                count += 1

                            if count >= value:
                                break

                    value_total += value
                    feature_mask[feature_mask == 1] = 0

                    # End Cythonize

            click.secho('Total value %d' % value_total)
            click.secho('Total count %d' % np.sum(out))
            click.secho('Max value %d' % np.max(out))
            click.secho('Non urban/Urban %d/%d' % (non_urban, urban))

            out[zone_mask == 0] = -1

            profile = ds.profile.copy()
            profile.update(dtype=np.int32, nodata=-1, compress='deflate')

            with rio.open(output, 'w', **profile) as dst:
                dst.write(out, 1)

if __name__ == '__main__':
    RasterizePopInsee()
