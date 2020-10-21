import os
from multiprocessing import Pool
import numpy as np
import click
import rasterio as rio
from rasterio.features import sieve
import fiona
from ..config import config
from ..cli import starcall
from ..tileio import buildvrt

def RestoreReferenceAxis(axis):

    sourcefile = config.filename('backup_medialaxis', axis=axis)
    destination = config.filename('ax_refaxis', axis=axis)

    with fiona.open(sourcefile) as fs:

        options = dict(
            driver=fs.driver,
            crs=fs.crs,
            schema=fs.schema
        )

        with fiona.open(destination, 'w', **options) as fst:
            with click.progressbar(fs) as iterator:
                for feature in iterator:
                    fst.write(feature)

def MaskHeightAboveNearestDrainageTile(axis, row, col):

    tileset = config.tileset()

    mask_tile = tileset.tilename(
        'backup_valley_mask',
        axis=axis,
        row=row,
        col=col
    )

    raster_tile = tileset.tilename(
        'backup_nearest_height_undelimited',
        axis=axis,
        row=row,
        col=col
    )

    output = tileset.tilename(
        'ax_nearest_height',
        axis=axis,
        row=row,
        col=col
    )

    if not (os.path.exists(raster_tile) and os.path.exists(mask_tile)):
        return

    with rio.open(raster_tile) as ds:

        data = ds.read(1)
        nodata = ds.nodata
        profile = ds.profile.copy()

    with rio.open(mask_tile) as ds:

        mask = ds.read(1)
        data[mask == ds.nodata] = nodata

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data, 1)

def MaskNearestDistanceTile(axis, row, col):

    tileset = config.tileset()

    mask_tile = tileset.tilename(
        'backup_valley_mask',
        axis=axis,
        row=row,
        col=col
    )

    raster_tile = tileset.tilename(
        'backup_nearest_distance_undelimited',
        axis=axis,
        row=row,
        col=col
    )

    output = tileset.tilename(
        'ax_nearest_distance',
        axis=axis,
        row=row,
        col=col
    )

    if not (os.path.exists(raster_tile) and os.path.exists(mask_tile)):
        return

    with rio.open(raster_tile) as ds:

        data = ds.read(1)
        nodata = ds.nodata
        profile = ds.profile.copy()

    with rio.open(mask_tile) as ds:

        mask = ds.read(1)
        data[mask == ds.nodata] = nodata

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data, 1)

def MaskHeightAboveNearestDrainage(axis, processes=1, **kwargs):

    tileset = config.tileset()
    tilefile = tileset.filename('ax_shortest_tiles', axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    MaskHeightAboveNearestDrainageTile,
                    axis,
                    row,
                    col,
                    {}
                )

                yield (
                    MaskNearestDistanceTile,
                    axis,
                    row,
                    col,
                    {}
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

    buildvrt('default', 'ax_nearest_height', axis=axis)
    buildvrt('default', 'ax_nearest_distance', axis=axis)

def MaskLandcoverTile(axis, row, col):

    tileset = config.tileset()

    mask_tile = tileset.tilename(
        'backup_valley_mask',
        axis=axis,
        row=row,
        col=col
    )

    raster_tile = tileset.tilename(
        'landcover-bdt',
        axis=axis,
        row=row,
        col=col
    )

    output = tileset.tilename(
        'ax_landcover',
        axis=axis,
        row=row,
        col=col
    )

    if not (os.path.exists(raster_tile) and os.path.exists(mask_tile)):
        return

    with rio.open(raster_tile) as ds:

        data = ds.read(1)
        nodata = ds.nodata
        profile = ds.profile.copy()

    with rio.open(mask_tile) as ds:

        mask = ds.read(1)
        data[mask == ds.nodata] = nodata

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data, 1)

def MaskLandcover(axis, processes=1, **kwargs):

    tileset = config.tileset()
    tilefile = tileset.filename('ax_shortest_tiles', axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    MaskLandcoverTile,
                    axis,
                    row,
                    col,
                    {}
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

    buildvrt('default', 'ax_landcover', axis=axis)
