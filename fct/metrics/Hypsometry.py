# coding: utf-8

"""
Hypsometry (elevations distribution)

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
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape
import xarray as xr

from .. import speedup
from ..config import config
from ..tileio import DownsampleRasterTile
from ..cli import starcall
from ..metadata import set_metadata

def TileMinMax(row, col):
    """
    Returns (minz, maxz) for tile (row, col)
    """

    elevation_raster = config.tileset().tilename('dem', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        elevations = ds.read(1)
        elevations = elevations[elevations != ds.nodata]

        if elevations.size == 0:
            return row, col, ds.nodata, ds.nodata

        return row, col, np.min(elevations), np.max(elevations)

def MinMax(processes=1, **kwargs):
    """
    >>> from operator import itemgetter
    >>> mmx = MinMax(6)
    >>> for (row, col), (zmin, zmax) in sorted([t for t in mmx.items()], key=itemgetter(1)): 
    >>>    print(row, col, zmin, zmax)
    """

    tileset = config.tileset()
    minmax = dict()

    def arguments():

        for tile in tileset.tiles():

            yield (
                TileMinMax,
                tile.row,
                tile.col,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for row, col, zmin, zmax in iterator:
                minmax[row, col] = (zmin, zmax)

    return minmax

def TileHypsometry(axis, row, col, zbins):
    """
    DOCME
    """

    elevation_raster = config.tileset().tilename('dem', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        elevations = ds.read(1)
        elevation_nodata = ds.nodata
        # elevations = elevations[elevations != ds.nodata]

    if axis is not None:

        watershed_raster = config.tileset().tilename('ax_watershed_raster', axis=axis, row=row, col=col)

        if os.path.exists(watershed_raster):

            with rio.open(watershed_raster) as ds:

                watershed = ds.read(1)
                elevations[watershed != axis] = elevation_nodata

    binned = np.digitize(elevations, zbins)
    binned[elevations == elevation_nodata] = 0
    # represented = set(np.unique(binned))

    # def area(k):

    #     if k in represented:
    #         return np.count_nonzero(binned == k)
    #     return 0

    # areas = {k: area(k) for k in range(1, zbins.size)}
    # areas[0] = 25.0*np.count_nonzero(elevations == nodata)

    return speedup.count_by_value(binned)

def Hypsometry(axis, processes=1, **kwargs):
    """
    Calculate elevation distributions (hypsometer)

    @api    fct-metrics:hypsometry

    @input  dem: dem
    @input  watershed_mask: ax_watershed_raster

    @param  minz: 0.0
    @param  maxz: 4800.0
    @param  dz: 10.0
    @param  pixel_area: 25.0e-6

    @output metrics_hypsometry: metrics_hypsometry_global
    @output metrics_hypsometry: metrics_hypsometry
    """

    tileset = config.tileset()
    areas = defaultdict(lambda: 0)

    minz = 0.0
    maxz = 4800.0
    dz = 10.0
    zbins = np.arange(minz, maxz + dz, dz)

    def arguments():

        for tile in tileset.tiles():

            yield (
                TileHypsometry,
                axis,
                tile.row,
                tile.col,
                zbins,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for t_areas in iterator:
                areas.update({k: areas[k] + 25.0e-6*t_areas[k] for k in t_areas})

    if axis is None:

        dataset = xr.Dataset({
            'area':  ('z', np.array([areas[k] for k in range(zbins.size)], dtype='float32')),
            'dz': dz
        }, coords={
            'z': np.float32(zbins)
        })

        set_metadata(dataset, 'metrics_hypsometry')

        output = config.filename('metrics_hypsometry_global')
        dataset.to_netcdf(output, 'w')

    else:

        dataset = xr.Dataset({
            'area':  ('z', np.array([areas[k] for k in range(zbins.size)], dtype='float32')),
            'dz': dz
        }, coords={
            'axis': axis,
            'z': np.float32(zbins)
        })

        set_metadata(dataset, 'metrics_hypsometry')

        output = config.filename('metrics_hypsometry', axis=axis)
        dataset.to_netcdf(output, 'w')


    return dataset

def TileElevationContour(row, col, breaks, resample_factor=1):
    """
    DOCME
    """

    elevations, profile = DownsampleRasterTile(row, col, 'dem50', None, resample_factor)
    # nodata = profile['nodata']
    transform = profile['transform']

    binned = np.uint8(np.digitize(elevations, breaks))
    binned = features.sieve(binned, 400)
    polygons = features.shapes(
        binned,
        binned != 0,
        connectivity=4,
        transform=transform)

    return [(polygon, value, row, col) for polygon, value in polygons if value > 0]

def ElevationContour(breaks, processes=1, **kwargs):
    """
    DOCME
    """

    tileset = config.tileset()

    def arguments():

        for tile in tileset.tiles():

            yield (
                TileElevationContour,
                tile.row,
                tile.col,
                breaks,
                kwargs
            )

    output = '/media/crousson/Backup/PRODUCTION/HYPSOMETRY/RMC_CONTOURS.shp'

    driver = 'ESRI Shapefile'
    crs = fiona.crs.from_epsg(2154)
    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('ROW', 'int:4'),
            ('COL', 'int:4'),
            ('VALUE', 'int'),
            ('Z', 'float:10.0')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    with fiona.open(output, 'w', **options) as dst:
        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments())

            with click.progressbar(pooled, length=len(tileset)) as iterator:
                for result in iterator:

                    for polygon, value, row, col in result:
                        z = breaks[int(value)-1]
                        geom = asShape(polygon).buffer(0.0)
                        properties = {'ROW': row, 'COL': col, 'VALUE': value, 'Z': z}
                        dst.write({
                            'geometry': geom.__geo_interface__,
                            'properties': properties
                        })
