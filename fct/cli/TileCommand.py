# coding: utf-8

"""
Generic Tile Management Utilities

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
import glob
import click

from .. import __version__ as version
from ..config import config
from ..tileio import (
    buildvrt,
    translate
)
from ..cli import fct_entry_point

from .Tiles import DatasourceToTiles
from .Options import overwritable

@fct_entry_point
def cli(env):
    """
    Generic tile management utilities
    """

@cli.command('rename')
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

@cli.command('delete')
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

@cli.command()
@click.argument('datasource')
@click.argument('tileset')
@click.argument('dataset')
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
@overwritable
def extract(datasource, tileset, dataset, processes=1, overwrite=False):
    """
    Extract Tiles from Datasource for tiles defined in Tileset,
    and store as Dataset.
    """
    
    DatasourceToTiles(datasource, tileset, dataset, processes, overwrite=overwrite)

@cli.command('buildvrt')
@click.argument('tileset')
@click.argument('dataset')
@click.option('--suffix/--no-suffix', default=True, help='Append tileset suffix')
def vrt(tileset, dataset, suffix):
    """
    Build GDAL Virtual Raster (VRT) from dataset tiles
    """

    buildvrt(tileset, dataset, suffix)

@cli.command()
@click.argument('dataset')
@click.option('--driver', default='gtiff', help='Output format: gtiff or netcdf')
def export(dataset, driver):
    """
    Export Virtual Raster (VRT) dataset to solid format GTiff or NetCDF
    """

    translate(dataset, driver)
