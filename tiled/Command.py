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

from config import tileindex, workdir

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
        @click.option('--processes', '-j', default=1, help="Execute j parallel processes")
        @click.option('--progress', '-p', default=False, help="Display progress bar", is_flag=True)
        @wraps(fun)
        def decorated(processes, progress, **kwargs):
            """
            Multiprocessing wrapper
            See https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
            """

            click.secho('Running %d processes ...' % processes, fg='yellow')

            tile_index = fun()
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

        return decorated

    return decorate

overwritable = click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
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
@overwritable
@quiet_opt
def patch():
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
        for filename in glob.glob(os.path.join(workdir(), '../TILEBOXES.*')):
            if os.path.exists(filename):
                os.unlink(filename)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:
            TileExtendedBoundingBox(row, col, padding)

@parallel(prepare, FillDepressions)
@overwritable
@quiet_opt
def fill():
    """
    Priority-Flood Depressions Filling
    """
    return tileindex()

@prepare.command()
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

@parallel(flats, LabelBorderFlats)
def labelflats():
    """
    Label tile border flats
    """
    return tileindex()

@flats.command('spillover')
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

@cli.group()
def flow():
    """
    Flow Direction Processing
    """
    pass

from FlowDirection import (
    FlowDirection,
    AggregateOutlets,
)

@parallel(flow, FlowDirection, 'calculate')
def flowdir():
    """
    Calculate Flow Direction Tiles
    """
    return tileindex()

@flow.command('aggregate')
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
    AggregateStreams
)

@drainage.command()
def inletareas():
    """
    Calculate Tiles Inlets Drainage Contribution
    """
    InletAreas()

@parallel(drainage, FlowAccumulation)
def accumulate():
    """
    Calculate Accumulation Raster Tiles
    """
    return tileindex()

@parallel(drainage, StreamToFeature)
@click.option('--min_drainage', '-a', default=5.0, help='Minimum Drainage Area in km²')
def vectorize():
    """
    Vectorize Stream Network Tiles
    """
    return tileindex()

@drainage.command('aggregate')
def aggregate_streams():
    """
    Aggregate Stream Network
    """
    AggregateStreams()

if __name__ == '__main__':
    cli()
