# coding: utf-8

"""
Configuration Classes

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

from ..config import config
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
    parallel_opt
)

# pylint: disable=import-outside-toplevel

@fct_entry_point
def cli(env):
    """
    Metrics extraction module
    """

@cli.command()
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def data_landcover(processes=1):
    """
    Reclass landcover data and create landcover tiles
    """

    MkLandCoverTiles(processes)

@cli.command()
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
    pass

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

@cli.command()
@click.argument('axis', type=int)
@parallel_opt
def valleybottom_swath(axis, processes):
    """
    Calculate landcover swaths
    """

    from fct.swath.ValleyBottomSwathProfile import ValleyBottomSwathProfile

    ValleyBottomSwathProfile(
        axis,
        processes=processes,
        valley_bottom_mask='ax_valley_mask_refined'
    )

@cli.command()
@click.argument('axis', type=int)
@parallel_opt
def landcover_swath(axis, processes):
    """
    Calculate landcover swaths
    """

    from fct.swath.LandCoverSwathProfile import LandCoverSwathProfile

    LandCoverSwathProfile(
        axis,
        processes=processes,
        landcover='landcover-bdt',
        valley_bottom_mask='ax_valley_mask_refined',
        subset='TOTAL_BDT')

    LandCoverSwathProfile(
        axis,
        processes=processes,
        landcover='ax_corridor_mask',
        subset='CONT_BDT')

@cli.command()
@click.argument('axis', type=int)
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

@cli.command()
@click.argument('axis', type=int)
def corridor_width(axis):
    """
    Calculate corridor width metrics
    """

    from .CorridorWidth import (
        CorridorWidth,
        WriteCorridorWidth
    )

    width = CorridorWidth(axis)
    WriteCorridorWidth(axis, width)

@cli.command()
@click.argument('axis', type=int)
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
        swath_data='ax_swath_landcover'
    )
    method = 'total landcover width'
    subset = 'TOTAL_BDT'
    data = LandCoverWidth(axis, method, datasets, subset=subset)
    WriteLandCoverWidth(axis, data, output='metrics_lcw_variant', variant=subset)

    datasets = DatasetParameter(
        landcover='ax_corridor_mask',
        swath_features='ax_valley_swaths_polygons',
        swath_data='ax_swath_landcover'
    )
    method = 'continuous buffer width from river channel'
    subset = 'CONT_BDT'
    data = LandCoverWidth(axis, method, datasets, subset=subset)
    WriteLandCoverWidth(axis, data, output='metrics_lcw_variant', variant=subset)
