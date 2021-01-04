# coding: utf-8

"""
Command Line Interface for Network Module
"""

import time
from datetime import datetime
import click

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

    def params_to_dict(params):
        """
        Enumerate values in parameter class instance
        """

        if isinstance(params, dict):

            for key, value in params.items():
                yield key, value

        else:

            for key, value in params.__dict__.items():
                if key.startswith('_'):
                    yield key[1:], value

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
        for parameter, value in params_to_dict(parameters):
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
        Parameters,
        ShortestHeight
    )

    # parameters = ShortestHeightDefaultParameters()
    # parameters.update(max_dz=max_height, max_distance=max_distance)
    params = Parameters()
    params.max_height = max_height
    params.max_distance = max_distance
    start_time = PrintCommandInfo('shortest height', None, processes, params)

    ShortestHeight(params, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', params.height.name)
    buildvrt('default', params.distance.name)

@cli.command()
@parallel_opt
def hand(processes):
    """
    Refine shortest height output with HAND
    (height above nearest drainage)
    """

    from .HeightAboveNearestDrainage import (
        Parameters,
        HeightAboveNearestDrainage
    )

    params = Parameters()
    start_time = PrintCommandInfo('height above nearest drainage', None, processes, params)

    HeightAboveNearestDrainage(axis=None, params=params, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', params.height.name)
    buildvrt('default', params.distance.name)


@cli.command()
@click.option('--buffer-width', default=20.0, help='buffer width in pixels')
@click.option('--max-height', default=12.0, help='Max height limit (meters)')
@parallel_opt
def valleymask(buffer_width, max_height, processes):
    """
    Clip height above threshold in HAND raster
    """

    from .ValleyBottomMask import (
        Parameters,
        ValleyBottomMask
    )

    # parameters = ValleyBottomMaskDefaultParameters()
    # parameters.update(max_height=max_height, buffer_width=buffer_width)

    params = Parameters()
    params.max_height = max_height
    params.buffer_width = buffer_width
    start_time = PrintCommandInfo('extract valley mask', None, processes, params)

    ValleyBottomMask(axis=None, params=params, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', params.output.name)

@cli.command()
@parallel_opt
def valleymask_hand(processes):
    """
    Refine shortest height output with HAND
    (height above nearest drainage)
    """

    from .HeightAboveNearestDrainage import (
        Parameters,
        HeightAboveNearestDrainage
    )

    # parameters = HeightAboveTalwegDefaultParameters()
    # parameters.update(mask='valley_mask')
    params = Parameters()
    params.mask = 'valley_mask'
    start_time = PrintCommandInfo('update valley mask HAND', None, processes, params)

    HeightAboveNearestDrainage(axis=None, params=params, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

    click.secho('Building output VRTs', fg='cyan')
    buildvrt('default', params.height.name)
    buildvrt('default', params.distance.name)

@cli.command()
@parallel_opt
@click.option('--maxiter', '-it', default=10, help='Stop after max iterations')
def continuity(processes, maxiter):
    """
    Extract natural corridor from landcover
    within valley bottom
    """

    from .LayeredContinuityAnalysis import (
        Parameters,
        LandcoverContinuityAnalysis
    )

    # parameters = ContinuityDefaultParameters()

    # parameters.update(
    #     output='continuity_variant',
    #     variant='MAX'
    # )

    params = Parameters()
    params.output = 'continuity_variant'
    parameters = dict(variant='MAX')

    start_time = PrintCommandInfo('landcover continuity analysis', None, processes, params)

    LandcoverContinuityAnalysis(
        params,
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

    from .WeightedContinuityAnalysis import (
        Parameters,
        WeightedContinuityAnalysis
    )

    # parameters = dict(
    #     tileset='default',
    #     landcover='landcover-bdt',
    #     output='continuity_variant',
    #     variant='WEIGHTED'
    # )

    params = Parameters()
    params.output = 'continuity_variant'
    parameters = dict(variant='WEIGHTED')

    start_time = PrintCommandInfo('distance weighted landcover continuity analysis', None, processes, params)

    WeightedContinuityAnalysis(
        params,
        processes=processes,
        **parameters
    )

    buildvrt('default', 'continuity_variant', variant='WEIGHTED')

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))


@cli.command()
@parallel_opt
def continuity_remap(processes):

    from .RemapContinuityRaster import (
        Parameters,
        RemapContinuityRaster
    )

    params = Parameters()
    params.continuity = 'continuity_variant'
    params.output = 'continuity_variant_remapped'

    RemapContinuityRaster(params, processes, variant='MAX')
    buildvrt(
        'default',
        params.output.name,
        variant='MAX'
    )

    RemapContinuityRaster(params, processes, variant='WEIGHTED')
    buildvrt(
        'default',
        params.output.name,
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
        Parameters,
        DisaggregateIntoSwaths,
        WriteSwathsBounds
    )

    # parameters = ValleyBottomParameters()
    # parameters.update(mdelta=length, ax_tiles='shortest_tiles')

    params = Parameters()
    params.mdelta = length
    params.ax_tiles = 'shortest_tiles'

    start_time = PrintCommandInfo('disaggregate valley bottom into longitudinal units', None, processes, params)

    swaths = DisaggregateIntoSwaths(params, processes=processes)

    WriteSwathsBounds(params, swaths)

    buildvrt('default', params.output_swaths_raster.name)
    buildvrt('default', params.output_measure.name)
    buildvrt('default', params.output_distance.name)
    buildvrt('default', params.output_nearest.name)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@swath.command()
@parallel_opt
def vectorize(processes):
    """
    Vectorize swath polygons
    """

    from .SwathMeasurement import (
        Parameters,
        VectorizeSwathPolygons
    )

    params = Parameters()
    params.ax_tiles = 'shortest_tiles'

    start_time = PrintCommandInfo('vectorize swath polygons', None, processes, params)

    VectorizeSwathPolygons(params, processes=processes)

    elapsed = time.time() - start_time
    click.secho('Elapsed time   : %s' % pretty_time_delta(elapsed))

@cli.command()
@arg_axis
@parallel_opt
def export_axis(axis, processes):
    """
    Export network datasets for given axis
    """

    from .ExportAxisDataset import (
        Parameters,
        CreateAxisMask,
        ExportSwathBounds,
        ExportSwathPolygons,
        ExportRasters,
        DefaultRasterMap
    )

    params = Parameters()

    CreateAxisMask(axis, params)
    ExportSwathBounds(axis, params)
    ExportSwathPolygons(axis, params)
    ExportRasters(axis, params, DefaultRasterMap(), processes=processes)
