# coding: utf-8

"""
DOCME

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import time
from datetime import datetime
from functools import wraps
from multiprocessing import Pool
import click

from ..config import config
from .. import __version__ as version

def pretty_time_delta(delta):
    """
    See https://gist.github.com/thatalextaylor/7408395
    """

    days, seconds = divmod(int(delta), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days > 0:
        return '%d d %d h %d min %.0f s' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%d h %d min %.0f s' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%d min %d s' % (minutes, seconds)
    else:
        return '%.1f s' % (delta,)

def command_info(command, ntiles, kwargs):

    start_time = time.time()
    tileset = config.tileset('default')
    workdir = config.workdir

    processes = kwargs.get('processes', 0)
    overwrite = kwargs.get('overwrite', False)

    click.secho('Command        : %s' % command, fg='green')
    click.secho('FCT version    : %s' % version)

    click.secho('Tileset        : %s' % tileset.name)
    click.secho('# of tiles     : %d' % ntiles)
    click.secho('Tile Directory : %s' % workdir)
    
    parameters = {
        k: v for k, v in kwargs.items()
        if k not in ['progress', 'overwrite', 'processes', 'verbose', 'tile']
    }

    if parameters:
        click.secho('--  Parameters :')
        for param, value in parameters.items():
            click.secho('%14s : %s' % (param, value), fg='cyan')

    if processes == 1:
        click.echo('Run single process')
    elif processes > 1:
        click.secho('Run %d parallel processes' % processes, fg='yellow')

    if overwrite:
        click.secho('Overwrite existing files', fg='yellow')

    click.secho('Start time     : %s' % datetime.fromtimestamp(start_time))

    return start_time

def aggregate(group, name=None):
    """
    Define a new aggregate command within `group`.
    """

    def decorate(fun):

        @group.command(name)
        @wraps(fun)
        def decorated(*args, **kwargs):

            tile_index = config.tileset('default').tileindex
            start_time = command_info(name or fun.__name__, len(tile_index), kwargs)

            fun(*args, **kwargs)

            elapsed = time.time() - start_time
            click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

        return decorated

    return decorate

def starcall(args):
    """
    Invoke first arg function with all other arguments.
    """

    fun = args[0]
    return fun(*args[1:-1], **args[-1])

def parallel(group, tilefun, name=None):
    """
    Define a new command within `group` as a Multiprocessing wrapper.
    This command will process tiles in parallel
    using function `tilefun`.
    """

    def decorate(fun):
        """
        Decorator function
        """

        @group.command(name)
        @click.option('--tile', type=(int, int), default=(None, None), help='Process only one tile')
        @click.option('--processes', '-j', default=1, help="Execute j parallel processes")
        @click.option('--progress', '-p', default=False, help="Display progress bar", is_flag=True)
        @wraps(fun)
        def decorated(**kwargs):
            """
            Multiprocessing wrapper
            See https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
            """

            tile_index = fun()
            tile = kwargs['tile']
            processes = kwargs['processes']
            progress = kwargs['progress']
            start_time = command_info(name or fun.__name__, len(tile_index), kwargs)

            kwargs = {k: v for k, v in kwargs.items() if k not in ('progress', 'processes', 'tile')}

            if tile != (None, None):

                row, col = tile
                if (row, col) in tile_index:
                    click.secho('Processing tile (%d, %d)' % (row, col), fg='cyan')
                    tilefun(row, col, **kwargs)
                else:
                    click.secho('No such tile (%d, %d)' % (row, col), fg='red')

            else:

                arguments = ([tilefun, row, col, kwargs] for row, col in tile_index)

                with Pool(processes=processes) as pool:

                    pooled = pool.imap_unordered(starcall, arguments)

                    if progress:

                        with click.progressbar(pooled, length=len(tile_index)) as bar:
                            for _ in bar:
                                # click.echo('\n\r')
                                pass

                    else:

                        for _ in pooled:
                            pass

            elapsed = time.time() - start_time
            click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

        return decorated

    return decorate
