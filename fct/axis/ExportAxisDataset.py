# coding: utf-8

"""
Export Network Rasters to per-axis Rasters
"""

from multiprocessing import Pool
import numpy as np
import xarray as xr
import rasterio as rio
import fiona
import pandas as pd
import click
from ..config import (
    config,
    LiteralParameter,
    DatasetParameter
)
from ..metadata import set_metadata
from ..tileio import buildvrt
from ..cli import starcall

class Parameters:
    """
    Network-to-axis export parameters
    """

    tilelist = DatasetParameter('domain tiles as CSV list')
    axis_nearest = DatasetParameter('nearest axis raster')
    swaths_index = DatasetParameter('swaths index/bounds')
    swaths_polygons = DatasetParameter('swaths polygons shapefile')

    output_tilelist = DatasetParameter('per-axis domain tiles as CSV list')
    output_axis_mask = DatasetParameter('per-axis mask raster')
    output_swaths_index = DatasetParameter('per-axis swaths index/bounds')
    output_swaths_polygons = DatasetParameter('per-axis swaths polygons shapefile')

    def __init__(self):
        """
        Default parameter values
        """

        self.tilelist = 'shortest_tiles'
        self.axis_nearest = 'axis_nearest'
        self.swaths_index = 'swaths_refaxis_bounds'
        self.swaths_polygons = 'swaths_refaxis_polygons'

        self.output_tilelist = 'ax_shortest_tiles'
        self.output_axis_mask = 'ax_axis_mask'
        self.output_swaths_index = 'ax_swaths_refaxis_bounds'
        self.output_swaths_polygons = 'ax_swaths_refaxis_polygons'

def CreateAxisMask(axis, params):

    tilefile = params.tilelist.filename()
    # config.tileset().filename('shortest_tiles')
    tiles = pd.read_csv(tilefile, names=('row', 'col'))
    ax_tiles = set()

    for row, col in tiles.values:

        rasterfile = params.axis_nearest.tilename(row=row, col=col)
        # config.tileset().tilename('axis_nearest', row=row, col=col)
        output = params.output_axis_mask.tilename(axis=axis, row=row, col=col)
        #config.tileset().tilename('ax_axis_mask', axis=axis, row=row, col=col)

        with rio.open(rasterfile) as ds:

            mask = np.uint8(ds.read(1) == axis)
            profile = ds.profile.copy()

        if np.sum(mask) > 0:

            profile.update(dtype='uint8', nodata=0, compress='deflate')

            with rio.open(output, 'w', **profile) as dst:
                dst.write(mask, 1)

            ax_tiles.add((row, col))

    output = params.output_tilelist.filename(axis=axis)
    # config.tileset().filename('ax_shortest_tiles', axis=axis)

    with open(output, 'w') as fp:
        for row, col in sorted(ax_tiles):
            fp.write(f'{row},{col}\n')

    buildvrt('default', params.output_axis_mask.name, axis=axis)

def ExportSwathBounds(axis, params):

    filename = params.swaths_index.filename(tileset=None)
    # config.filename('swaths_refaxis_bounds')
    output = params.output_swaths_index.filename(tileset=None, axis=axis)
    # config.filename('ax_swaths_refaxis_bounds', axis=axis)
    
    data = (
        xr
        .open_dataset(filename)
        .set_index(unit=['axis', 'swath'])
        .sel(axis=axis)
    )

    dataset = xr.Dataset(
        {
            'measure': (('swath',), data.measure.values),
            'bounds': (('swath', 'coord'), data.bounds.values),
            'delta_measure': data.delta_measure
        },
        coords={
            'axis': axis,
            'swath': data.swath.values,
            'coord': ['minx', 'miny', 'maxx', 'maxy']
        })

    set_metadata(dataset, 'swath_bounds')

    dataset.attrs['geographic_object'] = data.attrs['geographic_object']
    dataset.attrs['reference_axis'] = data.attrs['reference_axis']

    dataset.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'bounds': dict(zlib=True, complevel=9, least_significant_digit=2),
            'swath': dict(zlib=True, complevel=9)
        })

def ExportSwathPolygons(axis, params):

    shapefile = params.swaths_polygons.filename(tileset=None)
    # config.filename('swaths_refaxis_polygons')
    output = params.output_swaths_polygons.filename(axis=axis, tileset=None)
    # config.filename('ax_swaths_refaxis_polygons', axis=axis)

    with fiona.open(shapefile) as fs:

        options = dict(
            driver=fs.driver,
            schema=fs.schema,
            crs=fs.crs)

        with fiona.open(output, 'w', **options) as fst:

            with click.progressbar(fs) as iterator:
                for feature in iterator:
                    if feature['properties']['AXIS'] == axis:
                        fst.write(feature)

def ExportRasterTile(axis, params, src, dst, row, col, **kwargs):
    
    rasterfile = config.tileset().tilename(src, row=row, col=col, **kwargs)
    maskfile = params.output_axis_mask.tilename(axis=axis, row=row, col=col, **kwargs)
    # config.tileset().tilename('ax_axis_mask', axis=axis, row=row, col=col, **kwargs)
    output = config.tileset().tilename(dst, axis=axis, row=row, col=col, **kwargs)

    with rio.open(maskfile) as ds:
        mask = ds.read(1)

    with rio.open(rasterfile) as ds:

        data = ds.read(1)
        profile = ds.profile.copy()

    profile.update(compress='deflate')

    with rio.open(output, 'w', **profile) as dst:
        
        data[mask == 0] = ds.nodata
        dst.write(data, 1)

def ExportRasters(axis, params, rastermap, processes=1):
    
    tilefile = params.output_tilelist.filename(axis=axis)
    # config.tileset().filename('ax_shortest_tiles', axis=axis)
    tiles = pd.read_csv(tilefile, names=('row', 'col'))

    def arguments():

        for src, dst, kwargs in rastermap:
            for row, col in tiles.values:
                yield (
                    ExportRasterTile,
                    axis,
                    params,
                    src,
                    dst,
                    row,
                    col,
                    kwargs
                )

    length = len(tiles)*len(rastermap)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        with click.progressbar(pooled, length=length) as iterator:
            for _ in iterator:
                pass

    for _, dst, kwargs in rastermap:
        click.echo(f'Build VRT for {dst} ...')
        buildvrt('default', dst, axis=axis, **kwargs)

def DefaultRasterMap():

    return [
        ('axis_measure', 'ax_axis_measure', {}),
        ('axis_distance', 'ax_axis_distance', {}),
        ('swaths_refaxis', 'ax_swaths_refaxis', {}),
        ('nearest_height', 'ax_nearest_height', {}),
        ('nearest_distance', 'ax_nearest_distance', {}),
        ('valley_mask', 'ax_valley_mask', {}),
        # ('continuity_distance', 'ax_continuity_distance', {}),
        # ('continuity_state', 'ax_continuity_state', {}),
        ('continuity_variant', 'ax_continuity_variant', dict(variant='MAX')),
        ('continuity_variant_remapped', 'ax_continuity_variant_remapped', dict(variant='MAX')),
        ('continuity_variant', 'ax_continuity_variant', dict(variant='WEIGHTED')),
        ('continuity_variant_remapped', 'ax_continuity_variant_remapped', dict(variant='WEIGHTED'))
    ]
