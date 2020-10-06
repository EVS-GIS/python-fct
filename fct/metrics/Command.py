# coding: utf-8

"""
Metrics Calculation Commands

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import click

from .LandCover import MkLandCoverTiles
from .Population import DisaggregatePopulation
from ..subgrid.SubGrid import (
    DefineSubGridMask,
    AggregatePopulation,
    AggregateLandCover,
    DominantLandCover
)

from ..cli import (
    fct_entry_point,
    fct_command,
    arg_axis,
    parallel_opt
)

# pylint: disable=import-outside-toplevel,unused-argument

@fct_entry_point
def cli(env):
    """
    Metrics extraction module
    """

@fct_command(cli)
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def data_landcover(processes=1):
    """
    Reclass landcover data and create landcover tiles
    """

    MkLandCoverTiles(processes)

@fct_command(cli)
@click.argument('variable')
@click.argument('destination')
@click.option('--landcoverset', '-lc', default='landcover-bdt')
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def data_population(variable, destination, landcoverset, processes=1):
    """
    Disaggregate population data to match the resolution of landcover data
    """

    DisaggregatePopulation(
        processes=processes,
        variable=variable,
        destination=destination,
        landcoverset=landcoverset)

@cli.group()
def subgrid():
    """
    SubGrid Aggregates
    """

@subgrid.command('mask')
def subgrid_mask():
    """
    Define SubGrid Mask
    """

    DefineSubGridMask()

@subgrid.command('population')
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def subgrid_population(processes=1):
    """
    Aggregate population data
    """

    AggregatePopulation(processes)

@subgrid.command('landcover')
@click.option(
    '--dataset', '-d',
    default='landcover-bdt',
    help='Select land cover dataset by logical name')
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def subgrid_landcover(dataset, processes=1):
    """
    Aggregate landcover data
    """

    click.secho('Using %s lancover dataset' % dataset, fg='cyan')
    AggregateLandCover(processes, dataset=dataset)

@subgrid.command('dominant')
def subgrid_dominant_landcover():
    """
    Calculate dominant landcover at subgrid's resolution
    """

    DominantLandCover()

@fct_command(cli)
@parallel_opt
def hypsometry_global(processes):
    """
    Calculate elevation distributions (hypsometer)
    """

    from .Hypsometry import Hypsometry

    Hypsometry(axis=None, processes=processes)

@fct_command(cli)
@arg_axis
@parallel_opt
def hypsometry(axis, processes):
    """
    Calculate elevation distributions (hypsometer)
    """

    from .Hypsometry import Hypsometry

    Hypsometry(axis=axis, processes=processes)

@fct_command(cli)
@arg_axis
@parallel_opt
def drainage_area(axis, processes):
    """
    Calculate drainage area
    """

    from .DrainageArea import MetricDrainageArea

    MetricDrainageArea(axis, processes)

@fct_command(cli)
@arg_axis
def talweg(axis):
    """
    Calculate talweg-related metrics :
    depth relative to floodplain, intercepted length,
    mean slope, representative elevation
    """

    from .TalwegMetrics import (
        TalwegMetrics,
        WriteTalwegMetrics
    )

    dataset = TalwegMetrics(axis)
    WriteTalwegMetrics(axis, dataset)

@fct_command(cli)
@arg_axis
def planform(axis):
    """
    Calculate talweg-related metrics :
    depth relative to floodplain, intercepted length,
    mean slope, representative elevation
    """

    from .PlanformShift import (
        PlanformShift,
        WritePlanforMetrics
    )

    dataset = PlanformShift(axis)
    WritePlanforMetrics(axis, dataset)

@fct_command(cli)
@arg_axis
def valleybottom_width(axis):
    """
    Calculate valley bottom width metrics
    """

    from .ValleyBottomWidth import (
        ValleyBottomWidth,
        WriteValleyBottomWidth
    )

    width = ValleyBottomWidth(axis)
    WriteValleyBottomWidth(axis, width)

# @fct_command(cli)
# @arg_axis
# def corridor_width(axis):
#     """
#     Calculate corridor width metrics
#     """

#     from .CorridorWidth import (
#         CorridorWidth,
#         WriteCorridorWidth
#     )

#     width = CorridorWidth(axis)
#     WriteCorridorWidth(axis, width)

@fct_command(cli)
@arg_axis
def landcover_width(axis):
    """
    Calculate landcover width metrics
    """

    from fct.metrics.LandCoverWidth import (
        DatasetParameter,
        LandCoverWidth,
        WriteLandCoverWidth
    )

    datasets = DatasetParameter(
        landcover='landcover-bdt',
        swath_features='ax_valley_swaths_polygons',
        swath_data='ax_swath_landcover_npz'
    )
    method = 'total landcover width'
    subset = 'TOTAL_BDT'
    data = LandCoverWidth(axis, method, datasets, subset=subset)
    WriteLandCoverWidth(axis, data, output='metrics_lcw_variant', variant=subset)

    datasets = DatasetParameter(
        # landcover='ax_corridor_mask',
        landcover='ax_continuity',
        swath_features='ax_valley_swaths_polygons',
        swath_data='ax_swath_landcover_npz'
    )
    method = 'continuous buffer width from river channel'
    subset = 'CONT_BDT'
    data = LandCoverWidth(axis, method, datasets, subset=subset)
    WriteLandCoverWidth(axis, data, output='metrics_lcw_variant', variant=subset)
