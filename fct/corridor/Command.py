# coding: utf-8

"""
Command Line Interface for Corridor Module
"""

import os
import glob
import time
from datetime import datetime
import click

from .. import __version__ as version
from ..config import config
from ..cli import (
    fct_entry_point,
    parallel_opt
)
from ..cli.Decorators import pretty_time_delta
from ..tileio import buildvrt

# pylint: disable=import-outside-toplevel

def PrintCommandInfo(command, axis, processes, parameters):
    """
    Print command info
    """

    start_time = time.time()

    tileset = config.tileset()

    click.secho('Command        : %s' % command, fg='green')
    click.secho('FCT version    : %s' % version)

    click.secho('Tileset        : %s' % tileset.name)
    click.secho('# of tiles     : %d' % len(tileset))

    click.secho('--%16s:' % 'Parameters', fg='cyan')

    if axis:
        click.secho('  %16s: %d' % ('axis', axis))

    for parameter, value in parameters.items():
        click.echo('  %16s: %s' % (parameter, value))

    if processes > 0:
        click.secho('-- Start time     : %s' % datetime.fromtimestamp(start_time))
        click.secho('Running %d processes' % processes, fg='yellow')

    return start_time

@fct_entry_point
def cli(env):
    """
    Fluvial corridor delineation module
    """

@cli.command('setup')
def setup_axes():
    """
    Create axes subdirectory
    and copy input data for each axis.
    """

    from .SetupAxes import SetupAxes

    PrintCommandInfo('setup axes data', None, 0, dict())
    SetupAxes()

@cli.command()
@click.argument('axis', type=int)
@click.option('--vrt/--no-vrt', default=True, help='Build VRT after processing')
@parallel_opt
def flow_height(axis, vrt, processes):
    """
    Relative heights based on flow direction
    """

    from .FlowHeight import FlowHeight

    FlowHeight(axis=axis, processes=processes)

    if vrt:

        buildvrt('default', 'ax_flow_height', axis=axis)
        buildvrt('default', 'ax_flow_distance', axis=axis)

@cli.command()
@click.argument('axis', type=int)
@click.option('--vrt/--no-vrt', default=True, help='Build VRT after processing')
@parallel_opt
def shortest_height(axis, vrt, processes):
    """
    Relative heights following shortest path to drainage/reference
    """

    from .ShortestHeight import (
        ShortestHeight,
        ShortestHeightDefaultParameters
    )

    parameters = ShortestHeightDefaultParameters()
    start_time = PrintCommandInfo('valley bottom shortest', axis, processes, parameters)

    ShortestHeight(axis=axis, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    if vrt:

        click.secho('Building output VRTs', fg='cyan')
        buildvrt('default', parameters['dataset_height'], axis=axis)
        buildvrt('default', parameters['dataset_distance'], axis=axis)

@cli.command()
@click.argument('axis', type=int)
@click.option('--vrt/--no-vrt', default=True, help='Build VRT after processing')
@parallel_opt
def hand(axis, vrt, processes):
    """
    Refine shortest height output with HAND
    (height above nearest drainage)
    """

    from .HeightAboveNearestDrainage import (
        HeightAboveNearestDrainage,
        HeightAboveTalwegDefaultParameters
    )

    parameters = HeightAboveTalwegDefaultParameters()
    start_time = PrintCommandInfo('height above nearest drainage', axis, processes, parameters)

    HeightAboveNearestDrainage(axis=axis, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    if vrt:

        click.secho('Building output VRTs', fg='cyan')
        buildvrt('default', parameters['height'], axis=axis)
        buildvrt('default', parameters['distance'], axis=axis)

# @cli.command('cliphand')
# @click.argument('axis', type=int)
# @parallel_opt
# def clip_hand(axis, processes):
#     """
#     Clip height above threshold in HAND raster
#     """

#     from .HeightAboveNearestDrainage import ClipHeight

#     start_time = PrintCommandInfo('height above nearest drainage', axis, processes, {})

#     ClipHeight(axis, processes=processes)

#     elapsed = time.time() - start_time
#     click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
@click.option('--vrt/--no-vrt', default=True, help='Build VRT after processing')
# @click.option('--buf', default=40.0, help='buffer width in pixels')
@parallel_opt
def valleymask(axis, vrt, processes):
    """
    Clip height above threshold in HAND raster
    """

    from .ValleyBottomMask import (
        ValleyBottomMask,
        ValleyBottomMaskDefaultParameters
    )

    parameters = ValleyBottomMaskDefaultParameters()
    start_time = PrintCommandInfo('height above nearest drainage', axis, processes, parameters)

    ValleyBottomMask(axis, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    if vrt:

        click.secho('Building output VRTs', fg='cyan')
        buildvrt('default', parameters['output'], axis=axis)

@cli.group()
def disaggregate():
    """
    Disaggregate spatial object into longitudinal units
    """

@disaggregate.command('valleybottom')
@click.argument('axis', type=int)
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@parallel_opt
def disaggregate_valley_bottom(axis, length, processes):
    """
    Disaggregate valley bottom (from nearest height raster)
    into longitudinal units
    """

    from ..swath.SpatialReferencing import (
        SpatialReference,
        ValleyBottomParameters,
        ExportSpatialUnitDefs,
        VectorizeContinuousAll
    )

    parameters = ValleyBottomParameters()
    parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    start_time = PrintCommandInfo('valley bottom disaggregation', axis, processes, parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Disaggregate valley bottom', fg='cyan')
    swaths = SpatialReference(
        axis=axis,
        processes=processes,
        **parameters)

    ExportSpatialUnitDefs(axis, swaths, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    buildvrt('default', 'ax_valley_swaths', axis=axis)
    buildvrt('default', 'ax_axis_measure', axis=axis)
    buildvrt('default', 'ax_axis_distance', axis=axis)

    click.secho('Polygonize spatial units', fg='cyan')
    VectorizeContinuousAll(
        axis=axis,
        processes=processes,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@disaggregate.command('natural')
@click.argument('axis', type=int)
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@parallel_opt
def disaggregate_natural(axis, length, processes):
    """
    Disaggregate natural corridor into longitudinal units
    """

    from ..swath.SpatialReferencing import (
        SpatialReference,
        AggregateSpatialUnits,
        NaturalCorridorParameters
    )

    parameters = NaturalCorridorParameters()
    parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    start_time = PrintCommandInfo('natural corridor disaggregation', axis, processes, parameters)

    SpatialReference(
        axis=axis,
        processes=processes,
        **parameters)

    AggregateSpatialUnits(axis, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))
