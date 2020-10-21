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

def BackupMedialAxis(axis):

    sourcefile = config.filename('ax_medialaxis', axis=axis)
    destination = config.filename('backup_medialaxis', axis=axis)

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

def BackupSwathPolygons(axis):

    sourcefile = config.filename('ax_swaths_medialaxis_polygons', axis=axis)
    destination = config.filename('backup_swaths_polygons', axis=axis)

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

    sourcefile = config.filename('ax_swaths_medialaxis_polygons_simplified', axis=axis)
    destination = config.filename('backup_swaths_polygons_simplified', axis=axis)

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

def BackupValleyMask(axis):

    tileset = config.tileset()
    tilefile = tileset.filename('ax_shortest_tiles', axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def tiles():

        with open(tilefile) as fp:
            for line in fp:
                row, col = tuple(int(x) for x in line.split(','))
                yield row, col

    with click.progressbar(tiles(), length=length()) as iterator:
        for row, col in iterator:

            sourcetile = tileset.tilename(
                'ax_swaths_medialaxis',
                axis=axis,
                row=row,
                col=col
            )

            destination = tileset.tilename(
                'backup_valley_mask',
                axis=axis,
                row=row,
                col=col
            )

            if not os.path.exists(sourcetile):
                continue

            with rio.open(sourcetile) as ds:

                profile = ds.profile.copy()
                mask = np.uint8(ds.read(1) != ds.nodata)

                # mask = sieve(mask, 400)

                profile.update(
                    dtype='uint8',
                    nodata=0,
                    compress='deflate'
                )

                with rio.open(destination, 'w', **profile) as dst:
                    dst.write(mask, 1)

    buildvrt('default', 'backup_valley_mask', axis=axis)

def BackupHeightAboveNearestDrainageTile(axis, row, col):

    tileset = config.tileset()
    sourcetile = tileset.tilename(
        'ax_nearest_height',
        axis=axis,
        row=row,
        col=col
    )

    destination = tileset.tilename(
        'backup_nearest_height_undelimited',
        axis=axis,
        row=row,
        col=col
    )

    if not os.path.exists(sourcetile):
        return

    with rio.open(sourcetile) as ds:

        profile = ds.profile.copy()
        hand = ds.read(1)

        with rio.open(destination, 'w', **profile) as dst:
            dst.write(hand, 1)

def BackupNearestDistanceTile(axis, row, col):

    tileset = config.tileset()
    sourcetile = tileset.tilename(
        'ax_nearest_distance',
        axis=axis,
        row=row,
        col=col
    )

    destination = tileset.tilename(
        'backup_nearest_distance_undelimited',
        axis=axis,
        row=row,
        col=col
    )

    if not os.path.exists(sourcetile):
        return

    with rio.open(sourcetile) as ds:

        profile = ds.profile.copy()
        hand = ds.read(1)

        with rio.open(destination, 'w', **profile) as dst:
            dst.write(hand, 1)

def BackupHeightAboveNearestDrainage(axis, processes=1):

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
                    BackupHeightAboveNearestDrainageTile,
                    axis,
                    row,
                    col,
                    {}
                )

                yield (
                    BackupNearestDistanceTile,
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

    buildvrt('default', 'backup_nearest_height_undelimited', axis=axis)
    buildvrt('default', 'backup_nearest_distance_undelimited', axis=axis)
