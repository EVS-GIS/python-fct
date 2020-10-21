# coding: utf-8

"""
Command Line Interface for Corridor Module
"""

import numpy as np
import click
import xarray as xr

from ..cli import (
    fct_entry_point,
    fct_command,
    arg_axis,
    parallel_opt
)

from ..tileio import buildvrt

# pylint: disable=import-outside-toplevel,unused-argument

@fct_entry_point
def cli(env):
    """
    Fluvial corridor delineation module
    """

@fct_command(cli, 'discretize valley bottom into longitudinal units')
@arg_axis
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@click.option('--medialaxis', default=False, is_flag=True, help='use medial axis for reference')
@parallel_opt
def discretize(axis, length, medialaxis, processes):
    """
    Disaggregate valley bottom (from nearest height raster)
    into longitudinal units
    """

    from .SwathMeasurement import (
        DisaggregateIntoSwaths,
        ValleyBottomParameters,
        ValleyMedialAxisParameters,
        WriteSwathsBounds,
        VectorizeSwathPolygons
    )

    if medialaxis:

        parameters = ValleyMedialAxisParameters()
        parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    else:

        parameters = ValleyBottomParameters()
        parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

    click.secho('Disaggregate valley bottom', fg='cyan')
    swaths = DisaggregateIntoSwaths(
        axis=axis,
        processes=processes,
        **parameters)

    WriteSwathsBounds(axis, swaths, **parameters)

    if medialaxis:

        buildvrt('default', 'ax_swaths_medialaxis', axis=axis)
        buildvrt('default', 'ax_medialaxis_measure', axis=axis)
        buildvrt('default', 'ax_medialaxis_distance', axis=axis)

    else:

        buildvrt('default', 'ax_swaths_refaxis', axis=axis)
        buildvrt('default', 'ax_axis_measure', axis=axis)
        buildvrt('default', 'ax_axis_distance', axis=axis)

    click.secho('Vectorize swath polygons', fg='cyan')
    VectorizeSwathPolygons(
        axis=axis,
        processes=processes,
        **parameters)

@fct_command(cli, 'discretize valley bottom into longitudinal units')
@arg_axis
@click.option('--length', default=200.0, help='unit length / disaggregation step')
@parallel_opt
def create(axis, length, processes):
    """
    Disaggregate valley bottom (from nearest height raster)
    into longitudinal units
    """

    from .SwathMeasurement import (
        DisaggregateIntoSwaths,
        ValleyBottomParameters,
        WriteSwathsBounds,
        VectorizeSwathPolygons
    )

    parameters = ValleyBottomParameters()
    parameters.update(
        mdelta=length,
        ax_tiles='ax_shortest_tiles',
        ax_mask='ax_nearest_height')

    click.secho('Disaggregate valley bottom', fg='cyan')
    swaths = DisaggregateIntoSwaths(
        axis=axis,
        processes=processes,
        **parameters)

    WriteSwathsBounds(axis, swaths, **parameters)

    buildvrt('default', 'ax_swaths_refaxis', axis=axis)
    buildvrt('default', 'ax_axis_measure', axis=axis)
    buildvrt('default', 'ax_axis_distance', axis=axis)

    click.secho('Vectorize swath polygons', fg='cyan')
    VectorizeSwathPolygons(
        axis=axis,
        processes=processes,
        **parameters)

@fct_command(cli, 'update swath units')
@arg_axis
@click.option('--medialaxis', default=False, is_flag=True, help='use medial axis for reference')
@parallel_opt
def update(axis, medialaxis, processes):
    """
    Commit manual swath edits to swaths raster
    """

    from .SwathMeasurement import (
        ValleyBottomParameters,
        ValleyMedialAxisParameters,
        UpdateSwathRaster
    )

    if medialaxis:

        parameters = ValleyMedialAxisParameters()
        parameters.update(ax_tiles='ax_shortest_tiles')

    else:

        parameters = ValleyBottomParameters()
        parameters.update(ax_tiles='ax_shortest_tiles')

    click.secho('Update swath raster', fg='cyan')
    UpdateSwathRaster(
        axis=axis,
        processes=processes,
        **parameters)

# @fct_command(cli, 'valley bottom swaths, using medial axis', name='medialaxis')
# @arg_axis
# @click.option('--length', default=200.0, help='unit length / disaggregation step')
# @parallel_opt
# def discretize_medial_axis(axis, length, processes):
#     """
#     Disaggregate valley bottom (from nearest height raster)
#     into longitudinal units
#     """

#     from .SwathMeasurement import (
#         DisaggregateIntoSwaths,
#         ValleyMedialAxisParameters,
#         WriteSwathsBounds,
#         VectorizeSwathPolygons,
#         UpdateSwathRaster
#     )

#     parameters = ValleyMedialAxisParameters()
#     parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

#     click.secho('Disaggregate valley bottom', fg='cyan')
#     swaths = DisaggregateIntoSwaths(
#         axis=axis,
#         processes=processes,
#         **parameters)

#     WriteSwathsBounds(axis, swaths, **parameters)

#     buildvrt('default', 'ax_valley_swaths', axis=axis)
#     buildvrt('default', 'ax_axis_measure', axis=axis)
#     buildvrt('default', 'ax_axis_distance', axis=axis)

#     click.secho('Vectorize swath polygons', fg='cyan')
#     VectorizeSwathPolygons(
#         axis=axis,
#         processes=processes,
#         **parameters)

#     click.secho('Update swath raster', fg='cyan')
#     UpdateSwathRaster(
#         axis=axis,
#         processes=processes,
#         **parameters)

# @fct_command(cli, 'natural corridor disaggregation', name='natural')
# @arg_axis
# @click.option('--length', default=200.0, help='unit length / disaggregation step')
# @parallel_opt
# def discretize_natural(axis, length, processes):
#     """
#     Disaggregate natural corridor into longitudinal units
#     """

#     from .SwathMeasurement import (
#         DisaggregateIntoSwaths,
#         AggregateSpatialUnits,
#         NaturalCorridorParameters
#     )

#     parameters = NaturalCorridorParameters()
#     parameters.update(mdelta=length, ax_tiles='ax_shortest_tiles')

#     DisaggregateIntoSwaths(
#         axis=axis,
#         processes=processes,
#         **parameters)

#     AggregateSpatialUnits(axis, **parameters)

@fct_command(cli, 'simplify swath polygons', name='simplify')
@arg_axis
@click.option('--simplify', default=20.0, help='simplify distance tolerance (Douglas-Peucker)')
@click.option('--smooth', default=3, help='smoothing iterations (Chaikin)')
@click.option('--medialaxis', default=False, is_flag=True, help='use medial axis for reference')
def simplify_swath_polygons(axis, simplify, smooth, medialaxis):
    """
    Simplify and smooth swath polygons
    """

    from .SimplifySwathPolygons import (
        SimplifySwathPolygons
    )

    if medialaxis:

        SimplifySwathPolygons(
            axis,
            simplify,
            smooth,
            polygons='ax_swaths_medialaxis_polygons',
            output='ax_swaths_medialaxis_polygons_simplified')

    else:

        SimplifySwathPolygons(
            axis,
            simplify,
            smooth,
            polygons='ax_swaths_refaxis_polygons',
            output='ax_swaths_refaxis_polygons_simplified')

# @fct_command(cli, 'swath medial axis', 'medialaxis')
# @arg_axis
# def medial_axis(axis):
#     """
#     Calculate swath medial axis
#     """

#     from .SwathMedialAxis import (
#         SwathMedialAxis,
#         ExportSwathMedialAxisToShapefile,
#         unproject
#     )

#     medialaxis = SwathMedialAxis(axis, processes=6)
#     data = xr.Dataset({'dist': ('measure', medialaxis[:, 1])}, coords={'measure': medialaxis[:, 0]})
#     smoothed = data.rolling(measure=5, center=True, min_periods=1).mean()
#     transformed = unproject(axis, np.column_stack([smoothed.measure, smoothed.dist]))
#     ExportSwathMedialAxisToShapefile(axis, transformed[~np.isnan(transformed[:, 1])])

@fct_command(cli, 'generate cross-profile swath axes', name='axes')
@arg_axis
@parallel_opt
def swath_axes(axis, processes):
    """
    Generate cross-profile swath axes
    """

    from .SwathAxes import SwathAxes

    SwathAxes(axis=axis, processes=processes)

@cli.group()
def profile():
    """
    Calculate swath profiles
    """

@cli.group('export')
def export_profile():
    """
    Export swath profiles to netCDF format
    """

@fct_command(profile, 'elevation swath profiles', name='elevation')
@arg_axis
@parallel_opt
def elevation_swath_profiles(axis, processes):
    """
    Calculate elevation swath profiles
    """

    from .ElevationSwathProfile import SwathProfiles

    click.secho('Calculate swath profiles', fg='cyan')
    SwathProfiles(axis=axis, processes=processes)

@fct_command(
    export_profile,
    'export elevation swath profiles to netcdf',
    name='elevation')
@arg_axis
def export_elevation_to_netcdf(axis):
    """
    Export elevation swath profiles to netCDF format
    """

    from .ElevationSwathProfile import ExportElevationSwathsToNetCDF

    ExportElevationSwathsToNetCDF(axis)

@fct_command(profile, 'valleybottom swath profiles', name='valleybottom')
@arg_axis
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

@fct_command(
    export_profile,
    'export valleybottom swath profiles to netcdf',
    name='valleybottom')
@arg_axis
def export_valleybottom_to_netcdf(axis):
    """
    Export valleybottom swath profiles to netCDF format
    """

    from .ValleyBottomSwathProfile import ExportValleyBottomSwathsToNetCDF

    ExportValleyBottomSwathsToNetCDF(axis)

@fct_command(profile, 'landcover swath profiles', name='landcover')
@arg_axis
@parallel_opt
def landcover_swath(axis, processes):
    """
    Calculate landcover swaths
    """

    from .LandCoverSwathProfile import LandCoverSwathProfile

    LandCoverSwathProfile(
        axis,
        processes=processes,
        landcover='ax_landcover',
        # valley_bottom_mask='ax_valley_mask_refined',
        subset='TOTAL')

    # LandCoverSwathProfile(
    #     axis,
    #     processes=processes,
    #     # landcover='ax_corridor_mask',
    #     landcover='ax_continuity_variant',
    #     variant='MAX',
    #     subset='MAX')

    # LandCoverSwathProfile(
    #     axis,
    #     processes=processes,
    #     # landcover='ax_corridor_mask',
    #     landcover='ax_continuity_variant',
    #     variant='WEIGHTED',
    #     subset='WEIGHTED')

@fct_command(
    export_profile,
    'export landcover swath profiles to netcdf',
    name='landcover')
@arg_axis
def export_landcover_to_netcdf(axis):
    """
    Export landcover swath profiles to netCDF format
    """

    from .LandCoverSwathProfile import ExportLandcoverSwathsToNetCDF

    ExportLandcoverSwathsToNetCDF(
        axis,
        landcover='ax_landcover',
        subset='TOTAL'
    )

    # ExportLandcoverSwathsToNetCDF(
    #     axis,
    #     landcover='ax_continuity',
    #     subset='MAX'
    # )

    # ExportLandcoverSwathsToNetCDF(
    #     axis,
    #     landcover='ax_continuity',
    #     subset='WEIGHTED'
    # )

@fct_command(profile, 'continuity swath profiles', name='continuity')
@arg_axis
def continuity_swath(axis):
    """
    Calculate continuity swaths
    """

    from fct.swath.LandCoverSwathProfile import (
        LandCoverSwathProfile,
        ExportContinuitySwathsToNetCDF
    )

    LandCoverSwathProfile( 
        axis,
        processes=6,
        # landcover='ax_corridor_mask',
        landcover='ax_continuity_variant_remapped',
        variant='MAX',
        subset='MAX')

    ExportContinuitySwathsToNetCDF(
        axis,
        landcover='ax_continuity_variant_remapped',
        variant='MAX',
        subset='MAX'
    )

    LandCoverSwathProfile( 
        axis,
        processes=6,
        # landcover='ax_corridor_mask',
        landcover='ax_continuity_variant_remapped',
        variant='WEIGHTED',
        subset='WEIGHTED')

    ExportContinuitySwathsToNetCDF(
        axis,
        landcover='ax_continuity_variant_remapped',
        variant='WEIGHTED',
        subset='WEIGHTED'
    )
