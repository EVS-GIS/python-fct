#!/usr/bin/env python
# coding: utf-8

"""
Command Line Interface Tools for Tiled Processing of Large DEM
"""

import os
import glob
from functools import wraps
from multiprocessing import Pool
import click

import time
from datetime import datetime, timedelta
from config import tileindex, workdir, filename, parameter

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
    tileset = parameter('workspace.tileset')
    workdir = parameter('workspace.workdir')

    processes = kwargs.get('processes', 0)
    overwrite = kwargs.get('overwrite', False)

    click.secho('Command        : %s' % command, fg='green')
    click.secho('FCT version    : %s' % '1.0.5')

    click.secho('Tileset        : %s' % tileset)
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

            tile_index = tileindex()
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

overwritable = click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
verbosable = click.option('--verbose', '-v', default=False, help='Print verbose messages ?', is_flag=True)
quiet_opt = click.option('--quiet/--no-quiet', '-q', default=False, help='Suppress message output ?')

@click.group()
def cli():
    """
    Tiled Processing of Large DEM
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

@cli.command()
def citation():

    click.secho('Fluvial Corridor Toolbox', fg='green')
    click.secho('Description ...')
    click.secho('Version ...')
    click.secho('Cite me ...')
    click.secho('GitHub Link ...')

@cli.group()
def fileset():
    """
    Manage filesets with ancillary files, eg. shapefiles
    """
    pass

@fileset.command('rename')
@click.argument('source')
@click.argument('destination')
@overwritable
def rename_fileset(source, destination, overwrite):
    """
    Rename fileset
    """

    src = filename(source)
    dest = filename(destination)

    if not os.path.exists(src):
        click.echo('Not found %s' % os.path.basename(src))
        return

    src = os.path.splitext(src)[0]
    dest = os.path.splitext(dest)[0]

    for name in glob.glob(src + '.*'):

        extension = os.path.splitext(name)[1]

        if os.path.exists(dest + extension) and not overwrite:
            click.secho('Not overwriting %s' % dest)
            return

        os.rename(src + extension, dest + extension)

@fileset.command('delete')
@click.argument('name')
def delete_fileset(name):
    """
    Delete fileset
    """

    if not click.confirm('Delete tile dataset %s ?' % name):
        return

    src = filename(name)

    if not os.path.exists(src):
        click.echo('Not found %s' % os.path.basename(src))
        return

    src = os.path.splitext(src)[0]
    for match in glob.glob(src + '.*'):
        os.unlink(match)

@cli.group()
def tileset():
    """
    Manage tile dataset defined in config.ini
    """
    pass

@tileset.command('rename')
@click.argument('source')
@click.argument('destination')
@click.option('--ext', '-e', default=False, is_flag=True, help='Glob extension')
@overwritable
def rename_tileset(source, destination, ext, overwrite):
    """
    Rename tile dataset
    """

    for row, col in tileindex():

        src = filename(source, row=row, col=col)
        dest = filename(destination, row=row, col=col)

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

@tileset.command('delete')
@click.argument('name')
@click.option('--ext', '-e', default=False, is_flag=True, help='Glob extension')
def delete(name, ext):
    """
    Delete tile dataset
    """

    if not click.confirm('Delete tile dataset %s ?' % name):
        return

    for row, col in tileindex():

        src = filename(name, row=row, col=col)

        if not os.path.exists(src):
            click.echo('Not found %s' % os.path.basename(src))
            continue

        if ext:

            src = os.path.splitext(src)[0]
            for match in glob.glob(src + '.*'):
                os.unlink(match)

        else:

            os.unlink(src)

@cli.group()
def prepare():
    """
    First-pass DEM Reconditioning
    """
    pass

from Burn import BurnTile
from PreProcessing import (
    TileExtendedBoundingBox,
    ExtractAndPatchTile,
    FillDepressions,
    Spillover,
    ApplyFlatZ
)

@parallel(prepare, BurnTile)
@overwritable
@click.option('--delta', default=-1.0)
def burn():
    """
    Burn Stream Network
    """
    return tileindex()

@parallel(prepare, ExtractAndPatchTile)
@verbosable
@overwritable
@click.option('--smooth', default=5, help='Smooth window')
def mktiles():
    """
    Extract and Patch DEM Tiles
    """
    return tileindex()

@prepare.command()
@click.option('--padding', default=20, help='Number of pixels of padding')
def boxes(padding):
    """
    Calculate Padded Tile Bounding Boxes
    """
    tile_index = tileindex()

    output = os.path.join(workdir(), '../TILEBOXES.shp')

    if os.path.exists(output):
        for name in glob.glob(os.path.join(workdir(), '../TILEBOXES.*')):
            if os.path.exists(name):
                os.unlink(name)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:
            TileExtendedBoundingBox(row, col, padding)

@parallel(prepare, FillDepressions)
@click.option('--burn', '-b', default=-1.0, help='Burn delta >= 0 (in meters)')
@overwritable
@verbosable
def fill():
    """
    Priority-Flood Depressions Filling
    """
    return tileindex()

@aggregate(prepare)
@overwritable
def spillover(overwrite):
    """
    Build Spillover Graph and Resolve Watersheds' Minimum Z
    """
    Spillover(overwrite)

@parallel(prepare, ApplyFlatZ)
@overwritable
def applyz():
    """
    Raise Flat Regions to their Minimum Z
    (Flat Filling)
    """
    return tileindex()

@cli.group()
def flats():
    """
    Tile-border flats processing
    """
    pass

from BorderFlats import (
    LabelBorderFlats,
    ResolveFlatSpillover,
    ApplyMinimumZ
)

from FlatMap import FlatDepth

@parallel(flats, LabelBorderFlats)
def labelflats():
    """
    Label tile border flats
    """
    return tileindex()

@aggregate(flats, 'spillover')
@click.option('--epsilon', '-e', default=0.0005, help='epsilon gradient')
def flatspillover(epsilon):
    """
    Build border flats spillover graph,
    and resolve watershed's minimum z
    in order to ensure epsilon gradient
    between connected watersheds.
    """
    ResolveFlatSpillover(epsilon)

@parallel(flats, ApplyMinimumZ)
@overwritable
def applyminz():
    """
    Apply flat spillover minimum Z to DEM tiles
    """
    return tileindex()

@parallel(flats, FlatDepth)
@overwritable
def depthmap():
    """
    Calculate raster map
    of how much flat cells have been raised
    after DEM depression filling
    """
    return tileindex()

@cli.group()
def flow():
    """
    Flow Direction Processing
    """
    pass

from FlowDirection import (
    FlowDirection,
    Outlets,
    AggregateOutlets,
)

@parallel(flow, FlowDirection, 'calculate')
@overwritable
def flowdir():
    """
    Calculate Flow Direction Tiles
    """
    return tileindex()

@parallel(flow, Outlets, 'outlets')
@verbosable
def extract_outlets():
    """
    Extract Tiles' Outlets from Flow Direction
    """
    return tileindex()

@aggregate(flow, 'aggregate')
def aggregate_outlets():
    """
    Aggregate Tile Outlets
    """
    AggregateOutlets()

@cli.group()
def drainage():
    """
    Accumulation Raster and Stream Network Processing
    """
    pass

from StreamNetwork import (
    InletAreas,
    FlowAccumulation,
    StreamToFeature,
    NoFlowPixels,
    AggregateNoFlowPixels,
    AggregateStreams
)

@aggregate(drainage, 'dispatch')
def dispatch():
    """
    Dispatch Drainage Contribution accross Tiles
    """
    InletAreas()

@parallel(drainage, FlowAccumulation)
@overwritable
def accumulate():
    """
    Calculate Accumulation Raster Tiles
    """
    return tileindex()

@parallel(drainage, NoFlowPixels)
@click.option('--min_drainage', '-a', default=5.0, help='Minimum Drainage Area in km²')
def noflow():
    """
    Find Problematic No Flow Pixels on Stream Network
    """
    return tileindex()

@aggregate(drainage, 'aggregate-noflow')
def aggregate_noflow():
    """
    Aggregate No Flow Shapefiles
    """
    AggregateNoFlowPixels()

@parallel(drainage, StreamToFeature)
@click.option('--min_drainage', '-a', default=5.0, help='Minimum Drainage Area in km²')
def vectorize():
    """
    Vectorize Stream Network Tiles
    """
    return tileindex()

@aggregate(drainage, 'aggregate')
def aggregate_streams():
    """
    Aggregate Stream Network
    """
    AggregateStreams()

@cli.group()
def streams():
    """
    Create Stream Network from Mapped Sources
    """
    pass

from StreamSources import (
    InletSources,
    StreamToFeatureFromSources,
    AggregateStreamsFromSources
)

@aggregate(streams, 'sources')
def sources():
    """
    Map Sources accross Tiles
    """
    InletSources()

@parallel(streams, StreamToFeatureFromSources, 'vectorize')
@click.option('--min_drainage', '-a', default=50.0, help='Minimum Drainage Area in km²')
def vectorize_from_sources():
    """
    Vectorize Stream Network From Mapped Sources
    """
    return tileindex()

@aggregate(streams, 'aggregate')
def aggregate_from_sources():
    """
    Aggregate Stream Network
    """
    AggregateStreamsFromSources()

if __name__ == '__main__':
    cli()
