# coding: utf-8

"""
Command Line Interface for Corridor Module
"""

import time
from datetime import datetime
import numpy as np
import click
import xarray as xr

from .. import __version__ as version
from ..config import config
from ..cli import (
    fct_entry_point,
    arg_axis,
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
@arg_axis
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
@arg_axis
@parallel_opt
def prepare_from_backup(axis, processes):
    """
    Restore Height above nearest drainage (HAND)
    and reference axis from first iteration HAND & valley medial axis.
    """

    from .Prepare import (
        MaskHeightAboveNearestDrainage,
        RestoreReferenceAxis
    )

    MaskHeightAboveNearestDrainage(axis, processes)
    RestoreReferenceAxis(axis)

@cli.command()
@arg_axis
@parallel_opt
def landcover(axis, processes):
    """
    Restore Height above nearest drainage (HAND)
    and reference axis from first iteration HAND & valley medial axis.
    """

    from .Prepare import MaskLandcover
    MaskLandcover(axis, processes)

@cli.command()
@arg_axis
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
@arg_axis
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
@arg_axis
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
# @arg_axis
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
@arg_axis
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

@cli.command()
@arg_axis
def medialaxis(axis):
    """
    Calculate corridor medial axis
    """

    from .MedialAxis import MedialAxis

    start_time = PrintCommandInfo(
        'corridor medial axis',
        axis,
        1,
        {})

    MedialAxis(axis)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@arg_axis
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

@cli.command()
@arg_axis
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
@arg_axis
def talweg_profile(axis):
    """
    Calculate idealized/smoothed talweg elevation profile
    """

    from .TalwegElevationProfile import TalwegElevationProfile

    start_time = PrintCommandInfo('smoothed talweg elevation profile', axis, 1, {})

    TalwegElevationProfile(axis)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@arg_axis
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
# @arg_axis
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
@arg_axis
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

    parameters.update(
        output='ax_continuity_variant',
        variant='MAX'
    )

    start_time = PrintCommandInfo('landcover continuity analysis', axis, processes, parameters)

    LandcoverContinuityAnalysis(
        axis=axis,
        processes=processes,
        maxiter=maxiter,
        **parameters)

    # buildvrt('default', 'ax_continuity_variant', axis=axis, variant='MAX')

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@arg_axis
@parallel_opt
# @click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
# @click.option('--infra/--no-infra', default=True, help='Account for infrastructures in space fragmentation')
def continuity_weighted(axis, processes):
    """
    Extract natural corridor from landcover
    within valley bottom
    """

    from ..continuity.LateralContinuity import LateralContinuity

    parameters = dict(
        tileset='default',
        landcover='landcover-bdt',
        output='ax_continuity_variant',
        variant='WEIGHTED'
    )

    start_time = PrintCommandInfo('distance weighted landcover continuity analysis', axis, processes, parameters)

    LateralContinuity(
        axis=axis,
        processes=processes,
        **parameters
    )

    buildvrt('default', 'ax_continuity_variant', axis=axis, variant='WEIGHTED')

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

# @cli.command()
# @arg_axis
# @click.option('--width', '-b', default=40.0, help='buffer width in pixels')
# @parallel_opt
# def corridor_mask(axis, width, processes):
#     """
#     Calculate natural/reversible corridor
#     """

#     from .CorridorMask import CorridorMask

#     start_time = PrintCommandInfo(
#         'natural corridor mask',
#         axis,
#         processes,
#         dict(buffer_width=width))

#     CorridorMask(
#         axis=axis,
#         buffer_width=width,
#         ax_tiles='ax_shortest_tiles',
#         processes=processes)

#     elapsed = time.time() - start_time
#     click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

# @cli.command()
# @arg_axis
# def valley_bottom_boundary(axis):
#     """
#     Pseudo valley bottom boundary
#     """

#     from .ValleyBottomBoundary import ValleyBottomBoundary

#     start_time = PrintCommandInfo(
#         'valley bottom boundary',
#         axis,
#         1)

#     ValleyBottomBoundary(axis)

#     elapsed = time.time() - start_time
#     click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@arg_axis
@parallel_opt
# @click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
# @click.option('--infra/--no-infra', default=True, help='Account for infrastructures in space fragmentation')
def continuity_remap(axis, processes):

    from ..continuity.RemapContinuityRaster import RemapContinuityRaster

    RemapContinuityRaster(axis, processes, variant='MAX')
    buildvrt(
        'default',
        'ax_continuity_variant_remapped',
        axis=axis,
        variant='MAX'
    )

    RemapContinuityRaster(axis, processes, variant='WEIGHTED')
    buildvrt(
        'default',
        'ax_continuity_variant_remapped',
        axis=axis,
        variant='WEIGHTED'
    )
