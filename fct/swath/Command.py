# coding: utf-8

"""
Command Line Interface for Corridor Module
"""


import click

from ..cli import (
    fct_entry_point,
    fct_command,
    parallel_opt
)

# pylint: disable=import-outside-toplevel,unused-argument

@fct_entry_point
def cli(env):
    """
    Fluvial corridor delineation module
    """

@fct_command(cli, 'elevation swath profiles', name='elevation')
@click.argument('axis', type=int)
@parallel_opt
def elevation_swath_profiles(axis, processes):
    """
    Calculate elevation swath profiles
    """

    from .ElevationSwathProfile import SwathProfiles

    click.secho('Calculate swath profiles', fg='cyan')
    SwathProfiles(axis=axis, processes=processes)

@fct_command(cli, 'export elevation swath profiles to netcdf', name='export-elevation')
@click.argument('axis', type=int)
def export_elevation_to_netcdf(axis):
    """
    Export elevation swath profiles to netCDF format
    """

    from .ElevationSwathProfile import ExportElevationSwathsToNetCDF

    ExportElevationSwathsToNetCDF(axis)

@fct_command(cli, 'valleybottom swath profiles', name='valleybottom')
@click.argument('axis', type=int)
@parallel_opt
def valleybottom_swath(axis, processes):
    """
    Calculate valleybottom swaths
    """

    from .ValleyBottomSwathProfile import ValleyBottomSwathProfile

    ValleyBottomSwathProfile(
        axis,
        processes=processes,
        valley_bottom_mask='ax_valley_mask_refined'
    )

@fct_command(cli, 'export valleybottom swath profiles to netcdf', name='export-valleybottom')
@click.argument('axis', type=int)
def export_valleybottom_to_netcdf(axis):
    """
    Export valleybottom swath profiles to netCDF format
    """

    from .ValleyBottomSwathProfile import ExportValleyBottomSwathsToNetCDF

    ExportValleyBottomSwathsToNetCDF(axis)

@fct_command(cli, 'landcover swath profiles', name='landcover')
@click.argument('axis', type=int)
@parallel_opt
def landcover_swath(axis, processes):
    """
    Calculate landcover swaths
    """

    from .LandCoverSwathProfile import LandCoverSwathProfile

    LandCoverSwathProfile(
        axis,
        processes=processes,
        landcover='landcover-bdt',
        valley_bottom_mask='ax_valley_mask_refined',
        subset='TOTAL_BDT')

    LandCoverSwathProfile(
        axis,
        processes=processes,
        # landcover='ax_corridor_mask',
        landcover='ax_continuity',
        subset='CONT_BDT')

@fct_command(cli, 'export landcover swath profiles to netcdf', name='export-landcover')
@click.argument('axis', type=int)
def export_landcover_to_netcdf(axis):
    """
    Export landcover swath profiles to netCDF format
    """

    from .LandCoverSwathProfile import ExportLandcoverSwathsToNetCDF

    ExportLandcoverSwathsToNetCDF(
        axis,
        landcover='landcover-bdt',
        subset='TOTAL_BDT'
    )

    ExportLandcoverSwathsToNetCDF(
        axis,
        landcover='ax_continuity',
        subset='CONT_BDT'
    )

@fct_command(cli, 'generate cross-profile swath axes', name='axes')
@click.argument('axis', type=int)
@parallel_opt
def swath_axes(axis, processes):
    """
    Generate cross-profile swath axes
    """

    from .SwathAxes import SwathAxes

    SwathAxes(axis=axis, processes=processes)
