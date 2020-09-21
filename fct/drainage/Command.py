# coding: utf-8

"""
Command Line Interface Tools for Tiled Processing of Large DEM
"""

import os
import glob
import click

from ..config import config

from ..cli import (
    fct_entry_point,
    aggregate,
    parallel,
    overwritable,
    verbosable
)

from .Burn import (
    BurnTile,
    DrapeNetworkAndAdjustElevations,
    SplitStreamNetworkIntoTiles
)

from .PrepareDEM import (
    TileExtendedBoundingBox,
    ExtractAndPatchTile,
    MeanFilter
)

from .DepressionFill import (
    LabelWatersheds,
    ResolveWatershedSpillover,
    DispatchWatershedMinimumZ
)

from .BorderFlats import (
    LabelBorderFlats,
    ResolveFlatSpillover,
    DispatchFlatMinimumZ
)

from .FlatMap import DepressionDepthMap

from .FlowDirection import (
    FlowDirection,
    Outlets,
    AggregateOutlets,
)

from .StreamNetwork import (
    InletAreas,
    FlowAccumulation,
    StreamToFeature,
    AggregateStreams,
    NoFlowPixels,
    AggregateNoFlowPixels
)

from .StreamSources import (
    InletSources,
    StreamToFeatureFromSources,
    AggregateStreamsFromSources,
    NoFlowPixels as NoFlowPixelsFromSources,
    AggregateNoFlowPixels as AggregateNoFlowPixelsFromSources,
)

from .JoinNetworkAttributes import (
    JoinNetworkAttributes,
    UpdateLengthOrder
)

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

@fct_entry_point
def cli(env):
    """
    Tiled Processing of Large DEM
    """

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
@click.option('--smooth', default=0, help='Smooth window')
@click.option('--exterior', default='exterior-domain', help='Exterior domain logical name')
@click.option('--exterior-data', default=9000.0, help='Value for exterior domain area')
def mktiles():
    """
    Extract and Patch DEM Tiles
    """
    return tileindex()

@parallel(prepare, MeanFilter)
@overwritable
@click.option('--window', default=5, help='Smooth window (pixels)')
def smooth():
    """
    Smooth DEM by applying a mean filter on a window defined by size
    """
    return tileindex()

@prepare.command()
@click.option('--dataset', '-ds', default='smoothed', help='DEM dataset logical name')
def drape(dataset):
    """
    Drape hydrography on DEM and split network into tiles
    """

    DrapeNetworkAndAdjustElevations(dataset)
    SplitStreamNetworkIntoTiles()

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

@cli.group()
def watershed():
    """
    Depression filling procedure.
    Part 1: generate cross-tile watershed graph
            and identify flat areas
    """

@parallel(watershed, LabelWatersheds, 'labels')
@overwritable
@verbosable
@click.option('--dataset', '-ds', default='smoothed', help='DEM dataset logical name')
@click.option('--burn', '-b', default=-1.0, help='Burn delta >= 0 (in meters)')
@click.option('--exterior-data', default=9000.0, help='Value for exterior domain area')
def watershed_labels():
    """
    Priority-Flood Depressions Filling
    """
    return tileindex()

@aggregate(watershed, 'resolve')
@overwritable
def watershed_resolve(overwrite):
    """
    Build Spillover Graph and Resolve Watersheds' Minimum Z
    """
    ResolveWatershedSpillover(overwrite)

@parallel(watershed, DispatchWatershedMinimumZ, 'dispatch')
@overwritable
def watershed_dispatch():
    """
    Raise Flat Regions to their Minimum Z
    (Flat Filling)
    """
    return tileindex()

@cli.group()
def flat():
    """
    Depression filling procedure.
    Part 2: flat drainage resolution
    """
    pass

@parallel(flat, LabelBorderFlats, 'labels')
def flat_labels():
    """
    Label tile border flats
    """
    return tileindex()

@aggregate(flat, 'resolve')
@click.option('--epsilon', '-e', default=0.0005, help='epsilon gradient')
def flat_resolve(epsilon):
    """
    Build border flats spillover graph,
    and resolve watershed's minimum z
    in order to ensure epsilon gradient
    between connected watersheds.
    """
    ResolveFlatSpillover(epsilon)

@parallel(flat, DispatchFlatMinimumZ, 'dispatch')
@overwritable
def flat_dispacth():
    """
    Apply flat spillover minimum Z to DEM tiles
    """
    return tileindex()

@parallel(flat, DepressionDepthMap)
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
@click.option('--exterior', default='exterior-domain', help='Exterior domain logical name')
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
@click.option('--exterior', default='exterior-inlets', help='Exterior flow logical name')
def dispatch(exterior):
    """
    Dispatch Drainage Contribution accross Tiles
    """
    InletAreas(exterior)

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

@streams.command()
@click.argument('sourcefile')
@click.argument('networkfile')
@click.argument('destination')
def join(sourcefile, networkfile, destination):
    """
    Join source attributes to network segments,
    and update axis length
    """

    temp = config.temporary_dataset(destination)
    click.echo('Intermediate result: %s -> %s' % (temp.name, temp.filename()))
    click.echo('JoinNetworkAttributes')
    JoinNetworkAttributes(sourcefile, networkfile, temp.name)
    click.echo('UpdateLengthOrder')
    UpdateLengthOrder(temp.name, destination)

@aggregate(streams)
def from_sources():
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

@parallel(streams, NoFlowPixelsFromSources, 'noflow')
@click.option('--min_drainage', '-a', default=5.0, help='Minimum Drainage Area in km²')
def noflow_from_sources():
    """
    Find Problematic No Flow Pixels on Stream Network
    """
    return tileindex()

@aggregate(streams, 'aggregate-noflow')
def aggregate_noflow_from_sources():
    """
    Aggregate No Flow Shapefiles
    """
    AggregateNoFlowPixelsFromSources()

from .FixNoFlow import DrainageRaster

@parallel(streams, DrainageRaster, 'rasterize')
@click.option('--min_drainage', '-a', default=5.0, help='Minimum Drainage Area in km²')
def drainage_raster():
    """
    Rasterize back drainage network
    """

    return tileindex()

@cli.command('watershed')
@click.argument('axis', type=click.INT)
@click.option('--processes', '-j', default=1)
def delineate_watershed(axis, processes):
    """
    Delineate watershed from drainage network/flow raster
    """

    #pylint:disable=import-outside-toplevel

    from .Watersheds import (
        Watershed,
        VectorizeWatershed
    )

    Watershed(axis, processes)
    VectorizeWatershed(axis, processes)
