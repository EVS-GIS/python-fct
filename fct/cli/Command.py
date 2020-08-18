#!/usr/bin/env python
# coding: utf-8

"""
Fluvial Corridor Toolbox
"""

import os
import glob
from shutil import copyfile
import click

from ..config import config
from ..tileio import buildvrt
from .. import __version__ as version

from .Tiles import DatasourceToTiles
from .Options import (
    overwritable
)

@click.group()
def info():
    """
    Fluvial Corridor Toolbox
    """
    pass

# def TestTile(row, col, delta, **kwargs):
#     import os
#     import time
#     print(os.getpid(), row, col, delta)
#     time.sleep(1)

# @parallel(cli, TestTile)
# @click.option('--delta', default=-1.0)
# def test():
#     """
#     Print arguments and exit
#     """
#     tiles = dict()
#     for i in range(10):
#         tiles[0, i] = i
#     return tiles

@info.command()
def citation():

    click.secho('Fluvial Corridor Toolbox', fg='green')
    click.secho('Version %s' % version)
    click.secho('Description ...')
    click.secho('Cite me ...')
    click.secho('GitHub Link ...')

@click.group()
def cli():
    """
    Files and tilesets utilities
    """
    pass

@cli.group()
def files():
    """
    Manage filesets with ancillary files, eg. shapefiles
    """
    pass

@files.command('rename')
@click.argument('source')
@click.argument('destination')
@overwritable
def rename_fileset(source, destination, overwrite):
    """
    Rename fileset
    """

    config.default()

    src = config.filename(source)
    dest = config.filename(destination)

    if not os.path.exists(src):
        click.echo('Not found %s' % os.path.basename(src))
        return

    if os.path.exists(destination) and not overwrite:
        click.secho('Not overwriting %s' % destination)
        return

    src = os.path.splitext(src)[0]
    dest = os.path.splitext(dest)[0]

    for name in glob.glob(src + '.*'):

        extension = os.path.splitext(name)[1]

        if os.path.exists(dest + extension) and not overwrite:
            click.secho('Not overwriting %s' % dest)
            return

        os.rename(src + extension, dest + extension)

@files.command('copy')
@click.argument('source')
@click.argument('destination')
@overwritable
def copy_fileset(source, destination, overwrite):
    """
    Rename fileset
    """

    config.default()

    src = config.filename(source)
    dest = config.filename(destination)

    if not os.path.exists(src):
        click.echo('Not found %s' % os.path.basename(src))
        return

    if os.path.exists(destination) and not overwrite:
        click.secho('Not overwriting %s' % destination)
        return

    src = os.path.splitext(src)[0]
    dest = os.path.splitext(dest)[0]

    for name in glob.glob(src + '.*'):

        extension = os.path.splitext(name)[1]

        if os.path.exists(dest + extension) and not overwrite:
            click.secho('Not overwriting %s' % dest)
            return

        copyfile(src + extension, dest + extension)

@files.command('delete')
@click.argument('name')
def delete_fileset(name):
    """
    Delete fileset
    """

    if not click.confirm('Delete tile dataset %s ?' % name):
        return

    src = config.filename(name)

    if not os.path.exists(src):
        click.echo('Not found %s' % os.path.basename(src))
        return

    src = os.path.splitext(src)[0]
    for match in glob.glob(src + '.*'):
        os.unlink(match)

@cli.group()
def tiles():
    """
    Manage tile dataset defined in config.ini
    """
    pass

@tiles.command('rename')
@click.argument('source')
@click.argument('destination')
@click.option('--ext', '-e', default=False, is_flag=True, help='Glob extension')
@overwritable
def rename_tileset(source, destination, ext, overwrite):
    """
    Rename tile dataset
    """

    for row, col in config.tileset('default').tileindex:

        src = config.filename(source, row=row, col=col)
        dest = config.filename(destination, row=row, col=col)

        if not os.path.exists(src):
            click.echo('Not found %s' % os.path.basename(src))
            continue

        if ext:

            src = os.path.splitext(src)[0]
            dest = os.path.splitext(dest)[0]

            for name in glob.glob(src + '.*'):

                extension = os.path.splitext(name)[1]

                if os.path.exists(dest + extension) and not overwrite:
                    click.secho('Not overwriting %s' % dest)
                    continue

                os.rename(src + extension, dest + extension)

        else:

            if os.path.exists(dest) and not overwrite:
                click.echo('Not overwriting %s' % dest)
                return

            os.rename(src, dest)

@tiles.command('delete')
@click.argument('name')
@click.option('--ext', '-e', default=False, is_flag=True, help='Glob extension')
def delete(name, ext):
    """
    Delete tile dataset
    """

    if not click.confirm('Delete tile dataset %s ?' % name):
        return

    for row, col in config.tileset('default').tileindex:

        src = config.filename(name, row=row, col=col)

        if not os.path.exists(src):
            click.echo('Not found %s' % os.path.basename(src))
            continue

        if ext:

            src = os.path.splitext(src)[0]
            for match in glob.glob(src + '.*'):
                os.unlink(match)

        else:

            os.unlink(src)

@tiles.command('extract')
@click.argument('datasource')
@click.argument('tileset')
@click.argument('dataset')
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def extract(datasource, tileset, dataset, processes=1):
    """
    Extract Tiles from Datasource for tiles defined in Tileset,
    and store as Dataset.
    """

    config.default()
    DatasourceToTiles(datasource, tileset, dataset, processes)

@tiles.command('buildvrt')
@click.argument('tileset')
@click.argument('dataset')
def vrt(tileset, dataset):
    """
    Build GDAL Virtual Raster (VRT) from dataset tiles
    """

    config.default()
    buildvrt(tileset, dataset)
