# coding: utf-8

"""
Generic File Management Utilities

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
import glob
from shutil import copyfile
import click

from ..config import config
from .. import __version__ as version

from .Options import (
    overwritable
)

@click.group()
def cli():
    """
    Generic file management utilities
    """

@cli.command('rename')
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

@cli.command('copy')
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

@cli.command('delete')
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
