# coding: utf-8

"""
Plot Commands

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import click
from ..config import config

# pylint: disable=import-outside-toplevel

@click.group()
def cli():
    """
    Preconfigured plots for visualizing FCT data
    """

@cli.command('swath')
@click.argument('axis', type=int)
@click.argument('swath', type=int)
@click.option(
    '--kind',
    type=click.Choice(['absolute', 'hand', 'havf'], case_sensitive=True),
    default='absolute',
    help="""select plot variant,
    absolute elevation,
    height above nearest drainage
    or height above valley floor""")
@click.option(
    '--clip',
    default=None,
    type=float,
    help='clip data at given height above nearest drainage')
@click.option(
    '--filename', '-f',
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, writable=True, exists=False),
    help='save output to file')
def plot_elevation_swath(axis, swath, kind, clip, filename):
    """
    Elevation swath profile
    """

    from .PlotElevationSwath import PlotSwath

    config.default()

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    PlotSwath(axis, swath, kind=kind, clip=clip, output=filename)

    if filename is None:
        plt.show(block=True)
