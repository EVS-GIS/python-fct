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
    Landcover/lateral continuity analysis
    """

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

    # from .LandcoverContinuityAnalysis import (
    from .LayeredContinuityAnalysis import (
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
