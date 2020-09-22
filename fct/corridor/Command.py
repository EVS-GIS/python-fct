# coding: utf-8

"""
Command Line Interface for Corridor Module
"""

import os
import glob
import time
from datetime import datetime
import numpy as np
import click
import xarray as xr

from .. import __version__ as version
from ..config import config
from ..cli import (
    fct_entry_point,
    parallel_opt
)
from ..cli.Decorators import pretty_time_delta
from ..tileio import buildvrt

# pylint: disable=import-outside-toplevel

def PrintCommandInfo(command, axis, processes, parameters=None):
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

    if parameters:
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

@cli.command()
@click.argument('axis', type=int)
@click.argument('name')
def vrt(axis, name):
    """
    Build Virtual Raster (VRT) for axis tileset
    """

    buildvrt('default', name, axis=axis)

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

    from ..swath.SwathMeasurement import (
        DisaggregateIntoSwaths,
        ValleyBottomParameters,
        WriteSwathsBounds,
        VectorizeSwathPolygons,
        UpdateSwathRaster
    )

    parameters = ValleyBottomParameters()
    parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    start_time = PrintCommandInfo('valley bottom swaths', axis, processes, parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Disaggregate valley bottom', fg='cyan')
    swaths = DisaggregateIntoSwaths(
        axis=axis,
        processes=processes,
        **parameters)

    WriteSwathsBounds(axis, swaths, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    buildvrt('default', 'ax_valley_swaths', axis=axis)
    buildvrt('default', 'ax_axis_measure', axis=axis)
    buildvrt('default', 'ax_axis_distance', axis=axis)

    click.secho('Vectorize swath polygons', fg='cyan')
    VectorizeSwathPolygons(
        axis=axis,
        processes=processes,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Update swath raster', fg='cyan')
    UpdateSwathRaster(
        axis=axis,
        processes=processes,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@disaggregate.command('medialaxis')
@click.argument('axis', type=int)
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@parallel_opt
def disaggregate_medial_axis(axis, length, processes):
    """
    Disaggregate valley bottom (from nearest height raster)
    into longitudinal units
    """

    from ..swath.SwathMeasurement import (
        DisaggregateIntoSwaths,
        ValleyMedialAxisParameters,
        WriteSwathsBounds,
        VectorizeSwathPolygons,
        UpdateSwathRaster
    )

    parameters = ValleyMedialAxisParameters()
    parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    start_time = PrintCommandInfo('valley bottom swaths, using medial axis', axis, processes, parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Disaggregate valley bottom', fg='cyan')
    swaths = DisaggregateIntoSwaths(
        axis=axis,
        processes=processes,
        **parameters)

    WriteSwathsBounds(axis, swaths, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    buildvrt('default', 'ax_valley_swaths', axis=axis)
    buildvrt('default', 'ax_axis_measure', axis=axis)
    buildvrt('default', 'ax_axis_distance', axis=axis)

    click.secho('Vectorize swath polygons', fg='cyan')
    VectorizeSwathPolygons(
        axis=axis,
        processes=processes,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Update swath raster', fg='cyan')
    UpdateSwathRaster(
        axis=axis,
        processes=processes,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
@parallel_opt
def swath_axes(axis, processes):
    """
    Generate cross-profile swath axes
    """

    from ..swath.SwathAxes import SwathAxes

    start_time = PrintCommandInfo('generate swath axes', axis, processes)

    SwathAxes(axis=axis, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
@click.option('--threshold', '-t', default=5.0, help='height threshold in meters')
@parallel_opt
def refine_valley_mask(axis, threshold, processes):
    """
    Refine valley mask, using talweg depth relative to valley floor
    """

    from .ValleyBottomRefined import ValleyMask

    start_time = PrintCommandInfo(
        'valley bottom swaths, using medial axis',
        axis,
        processes,
        dict(threshold=threshold))

    ValleyMask(axis=axis, threshold=threshold, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    buildvrt('default', 'ax_valley_mask_refined', axis=axis)

@disaggregate.command('natural')
@click.argument('axis', type=int)
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@parallel_opt
def disaggregate_natural(axis, length, processes):
    """
    Disaggregate natural corridor into longitudinal units
    """

    from ..swath.SwathMeasurement import (
        DisaggregateIntoSwaths,
        AggregateSpatialUnits,
        NaturalCorridorParameters
    )

    parameters = NaturalCorridorParameters()
    parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    start_time = PrintCommandInfo('natural corridor disaggregation', axis, processes, parameters)

    DisaggregateIntoSwaths(
        axis=axis,
        processes=processes,
        **parameters)

    AggregateSpatialUnits(axis, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
@parallel_opt
def elevation_swath_profiles(axis, processes):
    """
    Calculate elevation swath profiles
    """

    from ..swath.ElevationSwathProfile import SwathProfiles

    start_time = PrintCommandInfo('elevation swath profiles', axis, processes, {})

    click.secho('Calculate swath profiles', fg='cyan')
    SwathProfiles(axis=axis, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
def talweg_height(axis):
    """
    Calculate talweg height relative to valley floor
    """

    from fct.corridor.ValleyBottomRefined import (
        TalwegHeightBySwathUnit,
        WriteTalwegHeights
    )

    swaths, values = TalwegHeightBySwathUnit(axis)
    WriteTalwegHeights(axis, swaths, values)

@cli.command('medialaxis')
@click.argument('axis', type=int)
def valley_medial_axis(axis):
    """
    Calculate valley medial axis
    """

    from .ValleyMedialAxis import (
        ValleyMedialAxis,
        ExportValleyMedialAxisToShapefile,
        unproject
    )

    medialaxis = ValleyMedialAxis(axis, processes=6)
    data = xr.Dataset({'dist': ('measure', medialaxis[:, 1])}, coords={'measure': medialaxis[:, 0]})
    smoothed = data.rolling(measure=5, center=True, min_periods=1).mean()
    transformed = unproject(axis, np.column_stack([smoothed.measure, smoothed.dist]))
    ExportValleyMedialAxisToShapefile(axis, transformed[~np.isnan(transformed[:, 1])])

@cli.command()
@click.argument('axis', type=int)
def valley_profile(axis):
    """
    Calculate idealized/smoothed valley elevation profile
    """

    from .ValleyElevationProfile import ValleyElevationProfile

    start_time = PrintCommandInfo('smoothed valley elevation profile', axis, 1, {})

    ValleyElevationProfile(axis)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@click.argument('axis', type=int)
@parallel_opt
def height_above_valley_floor(axis, processes):
    """
    Calculate height raster relative to valley floor
    """

    from .HeightAboveValleyFloor import HeightAboveValleyFloor

    start_time = PrintCommandInfo('height above valley floor', axis, processes, {})

    HeightAboveValleyFloor(axis=axis, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    buildvrt('default', 'ax_valley_height', axis=axis)

# @cli.command()
# @click.argument('axis', type=int)
# @parallel_opt
# @click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
# @click.option('--infra/--no-infra', default=True, help='Account for infrastructures in space fragmentation')
# def natural(axis, processes, maxiter, infra):
#     """
#     Extract natural corridor from landcover
#     within valley bottom
#     """

#     from ..continuity.LayeredContinuityAnalysis import (
#         LandcoverContinuityAnalysis,
#         NaturalCorridorDefaultParameters
#     )

#     parameters = NaturalCorridorDefaultParameters()
#     parameters.update(with_infra=infra)

#     start_time = PrintCommandInfo('natural corridor', axis, processes, parameters)

#     LandcoverContinuityAnalysis(
#         axis=axis,
#         processes=processes,
#         maxiter=maxiter,
#         **parameters)

#     elapsed = time.time() - start_time
#     click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

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

    from ..continuity.LayeredContinuityAnalysis import (
        LandcoverContinuityAnalysis,
        ContinuityDefaultParameters,
        NoInfrastructureParameters
    )

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

@cli.command()
@click.argument('axis', type=int)
@click.option('--width', '-b', default=40.0, help='buffer width in pixels')
@parallel_opt
def corridor_mask(axis, width, processes):
    """
    Calculate natural/reversible corridor
    """

    from .CorridorMask import CorridorMask

    start_time = PrintCommandInfo(
        'natural corridor mask',
        axis,
        processes,
        dict(buffer_width=width))

    CorridorMask(
        axis=axis,
        buffer_width=width,
        ax_tiles='ax_shortest_tiles',
        processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))
