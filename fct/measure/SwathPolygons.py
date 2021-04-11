# coding: utf-8

"""
Longitudinal swath generation :
discretize space along reference axis
"""

from operator import itemgetter
from multiprocessing import Pool
import logging

import numpy as np
import click

import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape, Polygon

from ..config import (
    LiteralParameter,
    DatasetParameter
)
# from ..tileio import ReadRasterTile
from ..tileio import as_window

from .. import transform as fct
from .. import speedup
from ..cli import starcall

from ..corridor.SwathDrainage import create_interpolate_drainage_fun
from ..corridor.ValleyBottomFeatures import (
    MASK_FLOOPLAIN_RELIEF,
    MASK_VALLEY_BOTTOM,
    MASK_HOLE
)

logger = logging.getLogger(__name__)

class Parameters:
    """
    Swath measurement parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')
    nearest = DatasetParameter(
        'nearest drainage axis (raster)',
        type='input')
    distance = DatasetParameter(
        'distance to drainage pixels (raster)',
        type='input')
    measure = DatasetParameter(
        'location along nearest reference axis (raster)',
        type='input')
    valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')

    swaths = DatasetParameter('swaths raster (discretized measures)', type='output')
    polygons = DatasetParameter('swaths polygons', type='output')
    
    swath_length = LiteralParameter(
        'swath disaggregation distance in measure unit (eg. meters)')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.nearest = 'nearest_drainage_axis'
            self.distance = 'nearest_distance'
            self.measure = 'axis_measure'
            self.valley_bottom = 'valley_bottom_final'
            self.swaths = 'swaths_refaxis'
            self.polygons = 'swaths_refaxis_polygons'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.nearest = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.distance = dict(key='ax_nearest_distance', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
            
            self.swaths = dict(key='ax_swaths_refaxis', axis=axis)
            self.polygons = dict(key='ax_swaths_refaxis_polygons', axis=axis)

        self.swath_length = 200.0


def SwathsTile(row, col, params, **kwargs):

    nearest_raster = params.nearest.tilename(row=row, col=col, **kwargs)
    measure_raster = params.measure.tilename(row=row, col=col, **kwargs)
    swaths_raster = params.swaths.tilename(row=row, col=col, **kwargs)
    valley_bottom_raster = params.valley_bottom.tilename(row=row, col=col, **kwargs)
    swath_length = params.swath_length

    maximinj = itemgetter(2, 1)
    minimaxj = itemgetter(0, 3)

    if not measure_raster.exists():
        return None, dict()

    with rio.open(nearest_raster) as ds:
        nearest = ds.read(1)

    with rio.open(valley_bottom_raster) as ds:
        
        valley_bottom = ds.read(1)
        valley_bottom_mask = (
            (valley_bottom == MASK_VALLEY_BOTTOM) |
            (valley_bottom == MASK_FLOOPLAIN_RELIEF) |
            (valley_bottom == MASK_HOLE)
        )

        del valley_bottom

    with rio.open(measure_raster) as ds:

        measure = ds.read(1)
        measure[~valley_bottom_mask] = ds.nodata

        if np.sum(measure[measure != ds.nodata]) == 0:
            return None, dict()

        measure_min = np.floor(np.min(measure[measure != ds.nodata]) / swath_length) * swath_length
        measure_max = np.ceil(np.max(measure[measure != ds.nodata]) / swath_length) * swath_length
        breaks = np.arange(measure_min, measure_max + swath_length, swath_length)
        measures = np.round(0.5 * (breaks[:-1] + breaks[1:]), 1)

        if measures.size == 0:
            return None, dict()

        swaths = np.uint32(np.digitize(measure, breaks))
        swaths_infos = dict()

        for axis in np.unique(nearest):

            if axis == 0:
                continue

            axis_mask = (nearest == axis)
            axis_swaths = np.zeros_like(swaths)
            axis_swaths[axis_mask] = swaths[axis_mask]
            boxes = speedup.flat_boxes(axis_swaths)

            if not boxes:
                continue

            lowerleft = fct.pixeltoworld(np.array([
                maximinj(box) for box in boxes.values()
            ], dtype='int32'), ds.transform)

            upperright = fct.pixeltoworld(np.array([
                minimaxj(box) for box in boxes.values()
            ], dtype='int32'), ds.transform)

            bounds = np.column_stack([lowerleft, upperright])

            swaths_infos.update({
                (axis, measures[swath-1]): bounds[k]
                for k, swath in enumerate(boxes)
                if 0 < swath < len(measures)+1
            })

        swaths_as_measures = np.float32(measures[0] + swath_length * (swaths - 1))
        swaths_as_measures[swaths == 0] = -99999.0

        profile = ds.profile.copy()
        profile.update(nodata=-99999.0, dtype='float32', compress='deflate')

        with rio.open(swaths_raster, 'w', **profile) as dst:
            dst.write(swaths_as_measures, 1)

        return measures[0], swaths_infos

def Swaths(params, processes=1, **kwargs):

    tilefile = params.tiles.filename()

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                SwathsTile,
                row,
                col,
                params,
                kwargs
            )

    g_attrs = dict()

    def merge_bounds(bounds1, bounds2):

        return (
            min(bounds1[0], bounds2[0]),
            min(bounds1[1], bounds2[1]),
            max(bounds1[2], bounds2[2]),
            max(bounds1[3], bounds2[3]),
        )

    def merge(attrs):

        if not attrs:
            # multiprocessing unpickles empty dict as list
            return

        g_attrs.update({
            key: merge_bounds(g_attrs[key], attrs[key])
            for key in attrs.keys() & g_attrs.keys()
        })

        g_attrs.update({
            key: attrs[key]
            for key in attrs.keys() - g_attrs.keys()
        })

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for measure_min, sw_infos in iterator:

                if measure_min is None:
                    continue

                merge(sw_infos)

    return g_attrs

def measure_to_swath_identifier(measures, swath_length):
    """
    Transform congruently metric measures to swath int identifier
    """

    return np.uint32(np.round(measures / swath_length + 0.5))

def VectorizeOneSwath(axis, gid, measure, bounds, params, **kwargs):
    """
    Vectorize swath polygon connected to talweg
    """

    # tileset = config.tileset()

    nearest_raster = params.nearest.filename(**kwargs)
    swaths_raster = params.swaths.filename(**kwargs)
    distance_raster = params.distance.filename(**kwargs)

    with rio.open(nearest_raster) as ds:

        window = as_window(bounds, ds.transform)
        nearest_axes = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(swaths_raster) as ds:

        window = as_window(bounds, ds.transform)
        swaths = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        # swaths = np.uint32(np.round((swaths - measure_min) / params.swath_length) + 1)
        swaths = measure_to_swath_identifier(swaths, params.swath_length)

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        state = np.full_like(swaths, 255, dtype='uint8')
        state[(nearest_axes == axis) & (swaths == gid)] = 0
        state[(state == 0) & (distance == 0)] = 1

        height, width = state.shape

        if height == 0 or width == 0:
            click.secho('Invalid swath %d with height = %d and width = %d' % (gid, height, width), fg='red')
            return axis, gid, measure, list()

        # out = np.full_like(state, 255, dtype='uint32')
        distance = np.zeros_like(state, dtype='float32')

        speedup.continuity_mask(
            state,
            # out,
            distance,
            jitter=0.4)

        # vectorize state => 0: outer space, 2: inner unit

        transform = ds.transform * ds.transform.translation(
            window.col_off,
            window.row_off)

        polygons = features.shapes(
            state,
            state != 255,
            connectivity=8,
            transform=transform)

        return axis, gid, measure, list(polygons)

def VectorizeSwaths(swaths_infos, drainage, params, processes=1, **kwargs):
    """
    Vectorize spatial units' polygons
    """

    interp_funs = dict()

    def interpolate_drainage(axis, measure):

        if axis not in interp_funs:
            
            try:

                fun = interp_funs[axis] = create_interpolate_drainage_fun(drainage, axis)

            except ValueError:

                logger.error('No drainage data for axis %d', axis)
                fun = interp_funs[axis] = lambda x: -99999.0

        else:

            fun = interp_funs[axis]

        return fun(measure)

    def arguments():

        for (axis, measure), bounds in swaths_infos.items():

            # gid = np.round((measure - measure_min) / params.swath_length) + 1
            gid = measure_to_swath_identifier(measure, params.swath_length)

            yield (
                VectorizeOneSwath,
                axis,
                gid,
                measure,
                # measure_min,
                bounds,
                params,
                kwargs
            )

    output = params.polygons.filename(tileset=None)
    # config.filename(params.output_swaths_shapefile, mod=False)

    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('GID', 'int'),
            ('AXIS', 'int:4'),
            ('VALUE', 'int:4'),
            # ('ROW', 'int:3'),
            # ('COL', 'int:3'),
            ('M', 'float:10.2'),
            ('DRAINAGE', 'float:10.3')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

    with fiona.open(output, 'w', **options) as dst:

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments())

            with click.progressbar(pooled, length=len(swaths_infos)) as iterator:
                for axis, gid, measure, polygons in iterator:

                    drainage_area = interpolate_drainage(axis, measure)

                    for (polygon, value) in polygons:

                        geom = asShape(polygon)
                        exterior = Polygon(geom.exterior).buffer(0)

                        feature = {
                            'geometry': exterior.__geo_interface__,
                            'properties': {
                                'GID': int(gid),
                                'AXIS': int(axis),
                                'VALUE': int(value),
                                # 'ROW': row,
                                # 'COL': col,
                                'M': float(measure),
                                'DRAINAGE': float(drainage_area)
                            }
                        }

                        dst.write(feature)

                        for ring in geom.interiors:

                            if not exterior.contains(ring):

                                feature = {
                                    'geometry': Polygon(ring).buffer(0).__geo_interface__,
                                    'properties': {
                                        'GID': int(gid),
                                        'AXIS': int(axis),
                                        'VALUE': int(value),
                                        # 'ROW': row,
                                        # 'COL': col,
                                        'M': float(measure),
                                        'DRAINAGE': float(drainage_area)
                                    }
                                }

                                dst.write(feature)
