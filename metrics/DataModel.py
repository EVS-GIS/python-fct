# coding: utf-8
# pylint:disable=invalid-name,protected-access,line-too-long

"""
NetCDF4 Fluvial Corridor Data Model

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import numpy as np
import click
from netCDF4 import Dataset

# template = Dataset('/home/crousson/projects/fct-cli/metrics/lamb93.nc', 'w')
# srs = rootgrp['lambert_conformal_conic']
# template.createVariable(srs.name, srs.datatype, srs.dimensions)
# template[srs.name].setncatts(srs.__dict__)

def createFluvialCorridorDataset(filename):
    """
    Creates an empty NetCDF data structure
    """

    rootgrp = Dataset(filename, 'w')
    rootgrp.Conventions = 'CF-1.5'
    rootgrp.FCT = '1.0.5'
    rootgrp.description = 'Lorem etc.'

    # Common variable options

    options = dict(zlib=True)

    xym_options = dict(
        zlib=True,
        least_significant_digit=2
    )

    elevation_options = dict(
        zlib=True,
        fill_value=-99999,
        least_significant_digit=1
    )

    measurement_options = dict(
        zlib=True,
        fill_value=-99999,
        least_significant_digit=1
    )

    landcover_options = dict(
        zlib=True,
        fill_value=-99999,
        least_significant_digit=0
    )

    # Groups

    grp_entity = rootgrp.createGroup('entity')
    grp_profile = rootgrp.createGroup('entity/profile')
    grp_corridor = rootgrp.createGroup('entity/corridor')
    grp_swath = rootgrp.createGroup('entity/swath')
    grp_swath_elevation = rootgrp.createGroup('entity/swath/elevation')
    grp_swath_landcover = rootgrp.createGroup('entity/swath/landcover')

    # Dimensions

    rootgrp.createDimension('axdim', None)
    rootgrp.createDimension('mdim', None)
    rootgrp.createDimension('profile', None)
    rootgrp.createDimension('landcover', 10)

    grp_swath.createDimension('sdim', None)
    grp_swath.createDimension('swath', None)
    grp_swath_elevation.createDimension('quantile', 5)
    # rootgrp.createDimension('string8', 8)
    # rootgrp.createDimension('string20', 20)

    # Spatial Reference System

    crs = rootgrp.createVariable('crs', '|S1')
    crs.long_name = 'Coordinate reference system (CRS) definition'
    crs.grid_mapping_name = 'lambert_conformal_conic'
    crs.projected_coordinate_system_name = "RGF93 / Lambert-93"
    crs.geographic_coordinate_system_name = "RGF93 / Reseau_Geodesique_Francais_1993"
    crs.vertical_datum_name = 'Nivellement General de la France - IGN69'
    crs.longitude_of_central_meridian = 3.0
    crs.false_easting = 700000.0
    crs.false_northing = 6600000.0
    crs.latitude_of_projection_origin = 46.5
    crs.standard_parallel = np.array([49., 44.])
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257222101
    crs.spatial_ref = """
        PROJCS["RGF93 / Lambert-93",
            GEOGCS["RGF93"
                DATUM["Reseau_Geodesique_Francais_1993",
                    SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],
                    TOWGS84[0,0,0,0,0,0,0],
                    AUTHORITY["EPSG","6171"]]
                PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4171"]],
            PROJECTION["Lambert_Conformal_Conic_2SP"],
            PARAMETER["standard_parallel_1",49],
            PARAMETER["standard_parallel_2",44],
            PARAMETER["latitude_of_origin",46.5],
            PARAMETER["central_meridian",3],
            PARAMETER["false_easting",700000],
            PARAMETER["false_northing",6600000],
            UNIT["metre",1,AUTHORITY["EPSG","9001"]],
            AXIS["X",EAST],
            AXIS["Y",NORTH],
            AUTHORITY["EPSG","2154"]]
    """
    crs.vertical_ref = """
        VERT_CS["NGF-IGN69 height",
            VERT_DATUM["Nivellement General de la France - IGN69",2005,
                AUTHORITY["EPSG","5119"]],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Up",UP],
            AUTHORITY["EPSG","5720"]]
    """
    crs.crs_wkt = crs.spatial_ref

    # Coordinates

    p = grp_profile.createVariable('profile', 'u4', ('profile',))
    p.long_name = 'long profile coordinate'
    p.compress = 'axdim mdim'

    s = grp_swath.createVariable('swath', 'u4', ('swath',))
    s.long_name = 'swath profile coordinate'
    s.compress = 'axdim mdim sdim'

    # Entity Attributes

    axdim = grp_entity.createVariable('axdim', 'u4', ('axdim',))
    axdim.long_name = 'long profile identifier'
    # unique identifier, unitless

    ox = grp_entity.createVariable('ox', 'f8', ('axdim',), **xym_options)
    ox.long_name = 'projected x location at long profile origin (most downstream point)'
    ox.units = 'm'

    oy = grp_entity.createVariable('oy', 'f8', ('axdim',), **xym_options)
    oy.long_name = 'projected y location at long profile origin (most downstream point)'
    oy.units = 'm'

    om = grp_entity.createVariable('om', 'f8', ('axdim',), **xym_options)
    om.long_name = 'measure (distance) from network outlet at long profile origin'
    om.units = 'm'

    srcx = grp_entity.createVariable('srcx', 'f8', ('axdim',), **xym_options)
    srcx.long_name = 'projected x location of stream source'
    srcx.units = 'm'

    srcy = grp_entity.createVariable('srcy', 'f8', ('axdim',), **xym_options)
    srcy.long_name = 'projected y location of stream source'
    srcy.units = 'm'

    srcm = grp_entity.createVariable('srcm', 'f8', ('axdim',), **xym_options)
    srcm.long_name = 'measure (distance) from origin at stream source'
    srcm.units = 'm'

    cdentitehy = grp_entity.createVariable('cdentitehy', 'S8', ('axdim',), **options)
    # cdentitehy = rootgrp.createVariable('cdentitehy', str, ('axdim', 'string8'))
    cdentitehy.long_name = 'hydrographic entity identifier'
    cdentitehy.source = 'BD Carthage'
    cdentitehy._Encoding = 'ascii'
    cdentitehy.coordinates = 'ox oy'
    cdentitehy.grid_mapping = 'crs: ox oy'
    # unique identifier, unitless

    hydronym = grp_entity.createVariable('hydronym', 'S20', ('axdim',), **options)
    # hydronym = rootgrp.createVariable('hydronym', str, ('axdim', 'string20'))
    hydronym.long_name = 'river toponym'
    hydronym.source = 'BD Topo'
    hydronym._Encoding = 'utf-8'
    hydronym.coordinates = 'ox oy'
    hydronym.grid_mapping = 'crs: ox oy'
    # name identifier, unitless

    hack = grp_entity.createVariable('hack', 'u4', ('axdim',), **options)
    hack.long_name = 'Hack order of long profile'
    hack.coordinates = 'ox oy'
    hack.grid_mapping = 'crs: ox oy'
    # unitless

    ostrahler = grp_entity.createVariable('ostrahler', 'u4', ('axdim',), **options)
    ostrahler.long_name = 'Strahler order at long profile origin'
    ostrahler.coordinates = 'ox oy'
    ostrahler.grid_mapping = 'crs: ox oy'
    # unitless

    # Long Profile Coordinates

    ax = grp_profile.createVariable('ax', 'u4', ('profile',), **options)
    ax.long_name = 'long profile identifier of LP unit'

    m = grp_profile.createVariable('m', 'f8', ('profile',), **xym_options)
    m.long_name = 'measure (distance) from origin of LP unit'
    m.units = 'm'

    # Long Profile General Characteristics

    px = grp_profile.createVariable('px', 'f8', ('profile',), **xym_options)
    px.long_name = 'projected x location of LP unit'
    px.units = 'm'
    px.coordinates = 'ax m'

    py = grp_profile.createVariable('py', 'f8', ('profile',), **xym_options)
    py.long_name = 'projected y location of LP unit'
    py.units = 'm'
    py.coordinates = 'ax m'

    strahler = grp_profile.createVariable('strahler', 'u4', ('profile',), **options)
    strahler.long_name = 'Strahler order at LP unit location'
    # unitless
    strahler.coordinates = 'ax m'

    # Long Profile Metrics

    drainage_options = dict(zlib=True, least_significant_digit=1, fill_value=0)

    drainage = grp_profile.createVariable('A', 'f8', ('profile',), **drainage_options)
    drainage.long_name = 'drainage area'
    drainage.units = 'km^2'
    drainage.coordinates = 'ax m'

    z = grp_profile.createVariable('z', 'f4', ('profile',), **elevation_options)
    z.long_name = 'elevation'
    z.units = 'm'
    z.vertical_ref = crs.vertical_ref
    z.source = 'RGE Alti 5 m'
    z.coordinates = 'ax m'

    dz = grp_profile.createVariable('dz', 'f4', ('profile',), **elevation_options)
    dz.long_name = 'slope, derived from z variable'
    dz.units = 'percent'
    dz.coordinates = 'ax m'

    dzv = grp_profile.createVariable('dzv', 'f4', ('profile',), **elevation_options)
    dzv.long_name = 'slope of valley floor'
    dzv.units = 'percent'
    dzv.coordinates = 'ax m'

    dzt = grp_profile.createVariable('dzt', 'f4', ('profile',), **elevation_options)
    dzt.long_name = 'slope of talweg'
    dzt.units = 'percent'
    dzt.coordinates = 'ax m'

    acw = grp_profile.createVariable('acw', 'f4', ('profile',), **measurement_options)
    acw.long_name = 'active channel width'
    acw.units = 'm'
    acw.coordinates = 'ax m'

    lfw = grp_profile.createVariable('lfw', 'f4', ('profile',), **measurement_options)
    lfw.long_name = 'low-flow water channel width'
    lfw.units = 'm'
    lfw.coordinates = 'ax m'

    axl = grp_profile.createVariable('axl', 'f4', ('profile',), **measurement_options)
    axl.long_name = 'intercepted reference axis length'
    axl.units = 'm'
    axl.coordinates = 'ax m'

    twl = grp_profile.createVariable('twl', 'f4', ('profile',), **measurement_options)
    twl.long_name = 'intercepted talweg length'
    twl.units = 'm'
    twl.coordinates = 'ax m'

    tcl = grp_profile.createVariable('tcl', 'f4', ('profile',), **measurement_options)
    tcl.long_name = 'intercepted water channels length'
    tcl.units = 'm'
    tcl.coordinates = 'ax m'

    # Fluvial corridor widths

    fcw = grp_corridor.createVariable('fcw', 'f4', ('profile',), **measurement_options)
    fcw.long_name = 'fluvial corridor width'
    fcw.units = 'm'
    fcw.coordinates = 'ax m'

    fcwk = grp_corridor.createVariable('fcwk', 'f4', ('profile', 'landcover'), **landcover_options)
    fcwk.long_name = 'fluvial corridor width per land cover class'
    fcwk.units = 'm'
    fcwk.coordinates = 'ax m'

    fcwc = grp_corridor.createVariable('fcwc', 'f4', ('profile', 'landcover'), **landcover_options)
    fcwc.long_name = 'fluvial corridor width per land cover continuity class'
    fcwc.units = 'm'
    fcwc.coordinates = 'ax m'

    # Swath Profile Coordinates

    # sx = grp_swath.createVariable('sx', 'u4', ('swath',), **options)
    # sx.long_name = 'long profile identifier of LP unit'

    # sm = grp_swath.createVariable('sm', 'f8', ('swath',), **xym_options)
    # sm.long_name = 'measure (distance) from origin of LP unit'
    # sm.units = 'm'

    sp = grp_swath.createVariable('sp', 'u4', ('swath',), **options)
    sp.long_name = 'long profile identifier of LP unit'
    sp.coordinates = 'ax m'

    sx = grp_swath.createVariable('sx', 'f8', ('swath',), **xym_options)
    sx.long_name = 'distance from reference axis'
    sx.positive = 'left bank to right bank'
    sx.units = 'm'

    # Swath slice general characteristics

    ds = grp_swath.createVariable('ds', 'u4', ('swath',), **options)
    ds.long_name = 'swath slice density, ie. number of pixels used in slice values'
    ds.units = 'pixels'
    ds.coordinates = 'sp sx'

    # Elevation per swath slice

    sz = grp_swath_elevation.createVariable('sz', 'f4', ('swath', 'quantile'), **elevation_options)
    sz.long_name = 'swath elevation'
    sz.units = 'm'
    sz.coordinates = 'sp sx'
    sz.vertical_ref = crs.vertical_ref

    hand = grp_swath_elevation.createVariable('hand', 'f4', ('swath', 'quantile'), **elevation_options)
    hand.long_name = 'height above nearest drainage'
    hand.units = 'm'
    hand.coordinates = 'sp sx'

    hvf = grp_swath_elevation.createVariable('hvf', 'f4', ('swath', 'quantile'), **elevation_options)
    hvf.long_name = 'height above valley floor'
    hvf.units = 'm'
    hvf.coordinates = 'sp sx'

    # Land cover metrics per swath slice

    lck = grp_swath_landcover.createVariable('lck', 'uint8', ('profile', 'landcover'), **options)
    lck.long_name = 'land cover classes represented at profile location'

    scwk = grp_swath_landcover.createVariable('scwk', 'uint32', ('swath', 'landcover'), **landcover_options)
    scwk.long_name = 'unit width for land cover class'
    scwk.units = 'pixels'
    scwk.coordinates = 'sp sx'

    lcc = grp_swath_landcover.createVariable('lcc', 'uint8', ('profile', 'landcover'), **options)
    lcc.long_name = 'land cover continuity classes represented at profile location'

    scwc = grp_swath_landcover.createVariable('scwc', 'uint32', ('swath', 'landcover'), **landcover_options)
    scwc.long_name = 'unit width for land cover continuity class'
    scwc.units = 'pixels'
    scwc.coordinates = 'sp sx'

    rootgrp.close()

@click.command('create')
@click.argument('filename')
@click.option('--overwrite', '-w', is_flag=True, default=False)
@click.pass_context
def cli(ctx, filename, overwrite):
    """
    Creates an empty NetCDF data structure
    """

    if os.path.exists(filename) and not overwrite:
        ctx.fail('File %s already exists' % filename)

    createFluvialCorridorDataset(filename)

if __name__ == '__main__':
    #pylint:disable=no-value-for-parameter
    cli()
