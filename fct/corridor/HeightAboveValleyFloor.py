# coding: utf-8

"""
Height above valley floor

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import namedtuple
from multiprocessing import Pool

import numpy as np
import xarray as xr

import rasterio as rio
import fiona
import fiona.crs

from .. import transform as fct
from ..config import config
from ..metrics import nearest_value_and_distance
from ..cli import starcall

DatasetParameter = namedtuple('DatasetParameter', [
    'tileset',
    'elevation',
    'mask',
    'valley_floor',
    'height'
])

def HeightAboveValleyFloorTile(
        axis,
        row,
        col,
        datasets,
        # buffer_width=30.0,
        # resolution=5.0,
        **kwargs):
    """
    see DistanceAndHeightAboveNearestDrainage
    """

    tileset = config.tileset(datasets.tileset)

    elevation_raster = tileset.tilename(datasets.elevation, row=row, col=col, **kwargs)
    valley_floor_file = config.filename(datasets.valley_floor, axis=axis, **kwargs)
    # drainage_shapefile = config.filename(datasets.drainage, axis=axis, **kwargs)
    # valley_bottom_rasterfile = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)
    mask_rasterfile = tileset.tilename(datasets.mask, axis=axis, row=row, col=col, **kwargs)

    output_height = tileset.tilename(datasets.height, axis=axis, row=row, col=col, **kwargs)
    # output_distance = tileset.tilename(datasets.distance, axis=axis, row=row, col=col, **kwargs)

    with rio.open(mask_rasterfile) as ds:

        mask = ds.read(1)
        # speedup.raster_buffer(mask, ds.nodata, buffer_width / resolution)
        height, width = mask.shape

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        def accept_pixel(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        # data = np.load(valley_floor_file, allow_pickle=True)
        # valley_floor = data['valley_profile']

        valley_profile = xr.open_dataset(valley_floor_file)
        valley_pixels = fct.worldtopixel(
            np.column_stack([
                valley_profile['x'],
                valley_profile['y']]),
            ds.transform)
        valley_mask = np.zeros(valley_pixels.shape[0], dtype=np.bool)

        for k, (i, j) in enumerate(valley_pixels):
            valley_mask[k] = accept_pixel(i, j)

        with rio.open(elevation_raster) as ds2:

            elevations = ds2.read(1)

        if np.sum(valley_mask) == 0:
            return

        reference, _ = nearest_value_and_distance(
            np.column_stack([
                valley_pixels[valley_mask],
                valley_profile['z'][valley_mask]
            ]),
            mask,
            ds.nodata)

        havf = elevations - reference
        havf[mask == ds.nodata] = ds.nodata

        with rio.open(output_height, 'w', **profile) as dst:
            dst.write(havf, 1)

def HeightAboveValleyFloor(
        axis,
        processes=1,
        ax_tiles='ax_tiles',
        tileset='landcover',
        elevation='tiled',
        valley_floor='ax_refaxis_valley_profile',
        mask='ax_valley_bottom',
        height='ax_valley_height',
        **kwargs):
    """
    Calculate distance and height above nearest drainage

    Parameters
    ----------

    axis: int

        Axis identifier

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Keyword Parameters
    ------------------

    buffer_width: float

        Width (real world units) of the buffer
        used to expand domain mask,
        defaults to 30.0 m

    resolution: float

        Raster resolution (real world units),
        used to scale distance,
        defaults to 5.0 m

    ax_tiles: str, logical name

        Axis list of intersecting tiles

    tileset: str, logical name

        Tileset to process

    elevation: str, logical name

        Absolute elevation raster (DEM)

    valley_floor: str, logical name

        Valley floor elevation profile,
        defaults to `ax_refaxis_valley_z`

    mask: str, logical name

        Mask raster,
        which defines the domain area to process
        from data/nodata values.
        ax_flow_height | ax_valley_bottom | ax_nearest_height

    height: str, logical name

        Output raster for calculated height
        above nearest drainage

    Other keywords are passed to dataset filename templates.
    """

    datasets = DatasetParameter(
        tileset=tileset,
        elevation=elevation,
        valley_floor=valley_floor,
        mask=mask,
        height=height
    )

    tilefile = config.filename(ax_tiles, axis=axis, **kwargs)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:
                _, row, col = tuple(int(x) for x in line.split(','))
                yield (
                    HeightAboveValleyFloorTile,
                    axis,
                    row,
                    col,
                    datasets,
                    kwargs
                )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def test(axis):

    #pylint: disable=import-outside-toplevel

    from ..metrics.LongProfileMetrics import (
        MetricSlopes,
        AdjustRefElevationGaps,
        ExportValleyProfile)

    config.default()

    _, _, seg2 = MetricSlopes(axis)
    seg3 = AdjustRefElevationGaps(seg2)

    valley_floor_file = config.filename('ax_refaxis_valley_profile', axis=axis)
    valley_profile = np.float32(np.concatenate(seg3, axis=0))
    ExportValleyProfile(axis, valley_profile, valley_floor_file)

    HeightAboveValleyFloor(axis, processes=5)
