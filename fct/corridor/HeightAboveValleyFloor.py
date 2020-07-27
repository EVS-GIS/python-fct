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

from math import ceil
import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from .. import transform as fct
from ..config import config
from ..metrics import nearest_value_and_distance
from ..cli import starcall

DatasetParameter = namedtuple('DatasetParameter', [
    'elevation',
    'mask',
    'valley_floor',
    'height'
])

def AdjustRefElevationGaps(segments):
    """
    Adjust gaps in the output of metrics.Metrics.MetricSlopes()
    """

    def measure(segment):
        """sort by m coordinate of first point"""
        return segment[0][3]

    sorted_segments = sorted(
        [
            np.copy(segment) for segment in segments
            if segment.size > 0 and segment[0, 3] != segment[-1, 3]
        ],
        key=measure)

    missing = list()

    for k, segment in enumerate(sorted_segments):

        if k == 0:
            continue

        m0 = segment[0, 3]

        if segment[0, 2] < sorted_segments[k-1][-1, 2]:
            
            # z0 = 0.5 * (segment[0, 2] + sorted_segments[k-1][-1, 2])
            z0 = sorted_segments[k-1][-1, 2]

            m1 = segment[-1, 3]
            z1 = segment[-1, 2]

            # linear interpolation between z0 and z1
            segment[:, 2] = (segment[:, 3] - m0) / (m1 - m0) * (z1 - z0) + z0

        m0 = sorted_segments[k-1][-1, 3]
        m1 = segment[0, 3]

        if m1 - m0 > 50.0:

            new_m = np.linspace(m1, m0, ceil((m1 - m0) / 10.0))
            new_segment = np.zeros((len(new_m), 4), dtype='float32')
            new_segment[:, 3] = new_m

            for j in range(3):
                
                c0 = sorted_segments[k-1][-1, j]
                c1 = segment[0, j]
                
                # linear interpolation between coordinate c0 and c1
                new_segment[:, j] = (new_m - m0) / (m1 - m0) * (c1 - c0) + c0

            missing.append(new_segment)

    print(len(missing))
    sorted_segments.extend(missing)

    return sorted(sorted_segments, key=measure)

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

    tileset = config.tileset('landcover')

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

        data = np.load(valley_floor_file, allow_pickle=True)
        valley_floor = data['valley_profile']
        valley_pixels = fct.worldtopixel(valley_floor[:, :2], ds.transform)
        valley_mask = np.zeros(valley_floor.shape[0], dtype=np.bool)

        for k, (i, j) in enumerate(valley_pixels):
            valley_mask[k] = accept_pixel(i, j)

        with rio.open(elevation_raster) as ds2:

            elevations = ds2.read(1)

        if np.sum(valley_mask) == 0:
            return

        reference, _ = nearest_value_and_distance(
            np.column_stack([
                valley_pixels[valley_mask],
                valley_floor[valley_mask, 2]
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
        elevation='tiled',
        valley_floor='ax_refaxis_valley_z',
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

    from ..metrics.Metrics import MetricSlopes

    config.default()

    _, _, seg2 = MetricSlopes(axis)
    seg3 = AdjustRefElevationGaps(seg2)

    valley_floor_file = config.filename('ax_refaxis_valley_z', axis=axis)
    np.savez(valley_floor_file, valley_profile=np.float32(np.concatenate(seg3, axis=0)))

    HeightAboveValleyFloor(axis, processes=5)
