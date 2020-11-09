# coding: utf-8

"""
Command Line Interface for Corridor Module
"""

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
    Network-scale swath measurement and procedures
    """

@cli.command()
@parallel_opt
def landcover(processes):
    """
    Restore Height above nearest drainage (HAND)
    and reference axis from first iteration HAND & valley medial axis.
    """

    from .ValleyBottomLandcover import ValleyBottomLandcover
    ValleyBottomLandcover(processes)

@cli.command()
@click.option('--max-height', default=20.0, help='Max height limit (meters)')
@click.option('--max-distance', default=2000, help='Max distance limit (pixels)')
@parallel_opt
def shortest_height(max_height, max_distance, processes):
    """
    Relative heights following shortest path to drainage/reference
    """

    from .ShortestHeight import (
        ShortestHeight,
        ShortestHeightDefaultParameters
    )

    parameters = ShortestHeightDefaultParameters()
    parameters.update(max_dz=max_height, max_distance=max_distance)
    start_time = PrintCommandInfo('valley bottom shortest', None, processes, parameters)

    ShortestHeight(processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', parameters['dataset_height'])
    buildvrt('default', parameters['dataset_distance'])

@cli.command()
@parallel_opt
def hand(processes):
    """
    Refine shortest height output with HAND
    (height above nearest drainage)
    """

    from .HeightAboveNearestDrainage import (
        HeightAboveNearestDrainage,
        HeightAboveTalwegDefaultParameters
    )

    parameters = HeightAboveTalwegDefaultParameters()
    start_time = PrintCommandInfo('height above nearest drainage', None, processes, parameters)

    HeightAboveNearestDrainage(axis=None, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', parameters['height'])
    buildvrt('default', parameters['distance'])


@cli.command()
@click.option('--buffer-width', default=20.0, help='buffer width in pixels')
@click.option('--max-height', default=12.0, help='Max height limit (meters)')
@parallel_opt
def valleymask(buffer_width, max_height, processes):
    """
    Clip height above threshold in HAND raster
    """

    from .ValleyBottomMask import (
        ValleyBottomMask,
        ValleyBottomMaskDefaultParameters
    )

    parameters = ValleyBottomMaskDefaultParameters()
    parameters.update(max_height=max_height, buffer_width=buffer_width)
    start_time = PrintCommandInfo('extract valley mask', None, processes, parameters)

    ValleyBottomMask(None, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', parameters['output'])

@cli.command()
@parallel_opt
def valleymask_hand(processes):
    """
    Refine shortest height output with HAND
    (height above nearest drainage)
    """

    from .HeightAboveNearestDrainage import (
        HeightAboveNearestDrainage,
        HeightAboveTalwegDefaultParameters
    )

    parameters = HeightAboveTalwegDefaultParameters()
    parameters.update(mask='valley_mask')
    start_time = PrintCommandInfo('update valley mask HAND', None, processes, parameters)

    HeightAboveNearestDrainage(axis=None, processes=processes, **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', parameters['height'])
    buildvrt('default', parameters['distance'])

@cli.command()
@parallel_opt
@click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
def continuity(processes, maxiter):
    """
    Extract natural corridor from landcover
    within valley bottom
    """

    from .LayeredContinuityAnalysis import (
        LandcoverContinuityAnalysis,
        ContinuityDefaultParameters
    )

    parameters = ContinuityDefaultParameters()

    parameters.update(
        output='continuity_variant',
        variant='MAX'
    )

    start_time = PrintCommandInfo('landcover continuity analysis', None, processes, parameters)

    LandcoverContinuityAnalysis(
        processes=processes,
        maxiter=maxiter,
        **parameters)

    # buildvrt('default', 'ax_continuity_variant', axis=axis, variant='MAX')

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@parallel_opt
# @click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
# @click.option('--infra/--no-infra', default=True, help='Account for infrastructures in space fragmentation')
def continuity_weighted(processes):
    """
    Extract natural corridor from landcover
    within valley bottom
    """

    from .WeightedContinuityAnalysis import WeightedContinuityAnalysis

    parameters = dict(
        tileset='default',
        landcover='landcover-bdt',
        output='continuity_variant',
        variant='WEIGHTED'
    )

    start_time = PrintCommandInfo('distance weighted landcover continuity analysis', None, processes, parameters)

    WeightedContinuityAnalysis(
        processes=processes,
        **parameters
    )

    buildvrt('default', 'continuity_variant', variant='WEIGHTED')

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))


@cli.command()
@parallel_opt
def continuity_remap(processes):

    from .RemapContinuityRaster import RemapContinuityRaster

    RemapContinuityRaster(processes, variant='MAX')
    buildvrt(
        'default',
        'continuity_variant_remapped',
        variant='MAX'
    )

    RemapContinuityRaster(processes, variant='WEIGHTED')
    buildvrt(
        'default',
        'continuity_variant_remapped',
        variant='WEIGHTED'
    )

@cli.group()
def swath():
    """
    Create longitudinal swaths
    """

@swath.command()
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@parallel_opt
def measure(length, processes):
    """
    Disaggregate valley bottom (from nearest height raster)
    into longitudinal units
    """

    from .SwathMeasurement import (
        DisaggregateIntoSwaths,
        ValleyBottomParameters,
        WriteSwathsBounds
    )

    parameters = ValleyBottomParameters()
    parameters.update(mdelta=length, ax_tiles='shortest_tiles')

    start_time = PrintCommandInfo('disaggregate valley bottom into longitudinal units', None, processes, parameters)

    swaths = DisaggregateIntoSwaths(
        processes=processes,
        **parameters)

    WriteSwathsBounds(swaths, **parameters)

    buildvrt('default', 'swaths_refaxis')
    buildvrt('default', 'axis_measure')
    buildvrt('default', 'axis_distance')
    buildvrt('default', 'axis_nearest')

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@swath.command()
@parallel_opt
def vectorize(processes):
    """
    Vectorize swath polygons
    """

    from .SwathMeasurement import VectorizeSwathPolygons, ValleyBottomParameters

    parameters = ValleyBottomParameters()
    parameters.update(ax_tiles='ax_shortest_tiles')

    start_time = PrintCommandInfo('vectorize swath polygons', None, processes, parameters)

    VectorizeSwathPolygons(
        processes=processes,
        **parameters)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))
