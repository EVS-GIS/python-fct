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
from ..cli import parallel_opt
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

@click.group()
def cli():
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

    config.default()
    PrintCommandInfo('setup axes data', None, 0, dict())
    SetupAxes()

@cli.command()
@click.argument('axis', type=int)
@click.option('--vrt/--no--vrt', default=True, help='Build VRT after processing')
@parallel_opt
def shortest(axis, vrt, processes):
    """
    Relative heights following shortest path to drainage/reference
    """

    # pylint: disable=import-outside-toplevel
    from .ValleyBottomShortest import (
        ValleyBottom,
        ValleyBottomDefaultParameters
    )

    config.default()
    parameters = ValleyBottomDefaultParameters()
    start_time = PrintCommandInfo('valley bottom shortest', axis, processes, parameters)

    ValleyBottom(axis=axis, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    if vrt:

        click.secho('Building output VRTs', fg='cyan')
        buildvrt('default', parameters['dataset_height'], axis=axis)
        buildvrt('default', parameters['dataset_distance'], axis=axis)

@cli.command()
@click.argument('axis', type=int)
@click.option('--vrt/--no--vrt', default=True, help='Build VRT after processing')
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

    config.default()
    parameters = HeightAboveTalwegDefaultParameters()
    start_time = PrintCommandInfo('height above nearest drainage', axis, processes, parameters)

    HeightAboveNearestDrainage(axis=axis, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    if vrt:

        click.secho('Building output VRTs', fg='cyan')
        buildvrt('default', parameters['height'], axis=axis)
        buildvrt('default', parameters['distance'], axis=axis)

@cli.command('cliphand')
@click.argument('axis', type=int)
@parallel_opt
def clip_hand(axis, processes):
    """
    Clip height above threshold in HAND raster
    """

    from .HeightAboveNearestDrainage import ClipHeight

    config.default()
    start_time = PrintCommandInfo('height above nearest drainage', axis, processes, {})

    ClipHeight(axis, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
@parallel_opt
@click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
@click.option('--infra/--no-infra', default=True, help='Account for infrastructures in space fragmentation')
def natural(axis, processes, maxiter, infra):
    """
    Extract natural corridor from landcover
    within valley bottom
    """

    # pylint: disable=import-outside-toplevel
    from .LayeredContinuityAnalysis import (
        LandcoverContinuityAnalysis,
        NaturalCorridorDefaultParameters
    )

    config.default()
    parameters = NaturalCorridorDefaultParameters()
    parameters.update(with_infra=infra)

    start_time = PrintCommandInfo('natural corridor', axis, processes, parameters)

    LandcoverContinuityAnalysis(
        axis=axis,
        processes=processes,
        maxiter=maxiter,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
@parallel_opt
@click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
@click.option('--infra/--no-infra', default=True, help='Account for infrastructures in space fragmentation')
def continuity(axis, processes, maxiter, infra):
    """
    Extract natural corridor from landcover
    within valley bottom
    """

    # pylint: disable=import-outside-toplevel
    # from .LandcoverContinuityAnalysis import (
    from .LayeredContinuityAnalysis import (
        LandcoverContinuityAnalysis,
        ContinuityDefaultParameters,
        NoInfrastructureParameters
    )

    config.default()

    if infra:
        parameters = ContinuityDefaultParameters()
    else:
        parameters = NoInfrastructureParameters()

    start_time = PrintCommandInfo('landcover continuity analysis', axis, processes, parameters)

    LandcoverContinuityAnalysis(
        axis=axis,
        processes=processes,
        maxiter=maxiter,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.group()
def disaggregate():
    """
    Disaggregate spatial object into longitudinal units
    """

@disaggregate.command('valleybottom')
@click.argument('axis', type=int)
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@click.option('--buf', default=40.0, help='buffer width in pixels')
@parallel_opt
def disaggregate_valley_bottom(axis, length, buf, processes):
    """
    Disaggregate valley bottom (from nearest height raster)
    into longitudinal units
    """

    # pylint: disable=import-outside-toplevel

    from .RasterBuffer import HANDBuffer

    from ..swath.SpatialReferencing import (
        SpatialReference,
        ValleyBottomParameters,
        ExportSpatialUnitDefs,
        VectorizeContinuousAll
    )

    config.default()

    parameters = ValleyBottomParameters()
    parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    start_time = PrintCommandInfo('valley bottom disaggregation', axis, processes, parameters)

    click.secho('Create valley bottom mask with buffer', fg='cyan')
    HANDBuffer(axis, buf, ax_tiles=parameters['ax_tiles'], processes=processes)

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

    # pylint: disable=import-outside-toplevel
    from ..swath.SpatialReferencing import (
        SpatialReference,
        AggregateSpatialUnits,
        NaturalCorridorParameters
    )

    config.default()

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
