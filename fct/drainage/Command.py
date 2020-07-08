# coding: utf-8

"""
Command Line Interface Tools for Tiled Processing of Large DEM
"""

import os
import glob
import click

from ..config import config

from ..cli import (
    aggregate,
    parallel,
    overwritable,
    verbosable
)

from .Burn import BurnTile
from .PreProcessing import (
    TileExtendedBoundingBox,
    ExtractAndPatchTile,
    FillDepressions,
    Spillover,
    ApplyFlatZ
)

from .BorderFlats import (
    LabelBorderFlats,
    ResolveFlatSpillover,
    ApplyMinimumZ
)

from .FlatMap import FlatDepth

from .FlowDirection import (
    FlowDirection,
    Outlets,
    AggregateOutlets,
)

from .StreamNetwork import (
    InletAreas,
    FlowAccumulation,
    StreamToFeature,
    NoFlowPixels,
    AggregateNoFlowPixels,
    AggregateStreams
)

from .StreamSources import (
    InletSources,
    StreamToFeatureFromSources,
    AggregateStreamsFromSources
)

config.default()

def workdir():
    """
    Return default working directory
    """
    return config.workdir

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

@click.group()
def cli():
    """
    Tiled Processing of Large DEM
    """
    pass

@cli.group()
def prepare():
    """
    First-pass DEM Reconditioning
    """
    pass

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
