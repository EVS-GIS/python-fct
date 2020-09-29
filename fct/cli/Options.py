# coding: utf-8

"""
Common Click Options

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import multiprocessing as mp
import click

def set_processes(ctx, param, value):
    """
    Callback for --processes option.
    Set default to mp.cpu_count()
    """

    if value is None or ctx.resilient_parsing:
        return

    if value == 0:
        return mp.cpu_count()

    return value

arg_axis = click.argument(
    'axis',
    type=int,
    envvar='FCT_AXIS')

overwritable = click.option(
    '--overwrite', '-w',
    default=False,
    help='Overwrite existing output ?',
    is_flag=True)

verbosable = click.option(
    '--verbose', '-v',
    default=False,
    help='Print verbose messages ?',
    is_flag=True)

quiet_opt = click.option(
    '--quiet/--no-quiet', '-q',
    default=False,
    help='Suppress message output ?')

parallel_opt = click.option(
    '--processes', '-j',
    default=1,
    callback=set_processes,
    help="Execute j parallel processes")
