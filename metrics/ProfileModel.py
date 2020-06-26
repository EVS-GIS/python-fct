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
import re
import numpy as np
import click
from netCDF4 import Dataset

# template = Dataset('/home/crousson/projects/fct-cli/metrics/lamb93.nc', 'w')
# srs = rootgrp['lambert_conformal_conic']
# template.createVariable(srs.name, srs.datatype, srs.dimensions)
# template[srs.name].setncatts(srs.__dict__)

def strip(s):
    # return re.sub(' {2,}', ' ', s.strip())
    return s.rstrip()

def createFluvialCorridorProfileDataset(filename, description=None):
    """
    Creates an empty NetCDF data structure
    """

    rootgrp = Dataset(filename, 'w')
    rootgrp.Conventions = 'CF-1.8'
    rootgrp.FCT = 'Fluvial Corridor Toolbox Profile Model 1.0.5'

    if description:
        rootgrp.description = description

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
    crs.spatial_ref = strip("""
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
    """)
    crs.vertical_ref = strip("""
        VERT_CS["NGF-IGN69 height",
            VERT_DATUM["Nivellement General de la France - IGN69",2005,
                AUTHORITY["EPSG","5119"]],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Up",UP],
            AUTHORITY["EPSG","5720"]]
    """)
    crs.crs_wkt = crs.spatial_ref

    rootgrp.featureType = 'profile'
    rootgrp.grid_mapping = 'crs: x, y'

    # Coordinates

    p = grp_profile.createVariable('profile', 'uint32', ('profile',))
    p.long_name = 'long profile coordinate'
    p.compress = 'axdim mdim'
    p.cf_role = 'profile_id'

    s = grp_swath.createVariable('swath', 'uint32', ('swath',))
    s.long_name = 'swath coordinate'
    s.compress = 'profile sdim'
    s.cf_role = 'profile_id' # swath ID

    # Entity Attributes

    ax = grp_entity.createVariable('ax', 'uint32', ('axdim',))
    ax.long_name = 'entity index'
    # unique identifier, unitless

    x = grp_entity.createVariable('x', 'float64', ('axdim',), **xym_options)
    x.long_name = 'projected x location at long profile origin (most downstream point)'
    x.units = 'm'

    y = grp_entity.createVariable('y', 'float64', ('axdim',), **xym_options)
    y.long_name = 'projected y location at long profile origin (most downstream point)'
    y.units = 'm'

    m = grp_entity.createVariable('m', 'float64', ('axdim',), **xym_options)
    m.long_name = 'measure (distance) from network outlet at long profile origin'
    m.units = 'm'
    m.grid_mapping = 'crs: x, y'

    xs = grp_entity.createVariable('xs', 'float64', ('axdim',), **xym_options)
    xs.long_name = 'projected x location of stream source'
    xs.units = 'm'

    ys = grp_entity.createVariable('ys', 'float64', ('axdim',), **xym_options)
    ys.long_name = 'projected y location of stream source'
    ys.units = 'm'

    ms = grp_entity.createVariable('ms', 'float64', ('axdim',), **xym_options)
    ms.long_name = 'measure (distance) from origin at stream source'
    ms.units = 'm'
    ms.grid_mapping = 'crs: xs, ys'

    cdh = grp_entity.createVariable('cdh', 'S8', ('axdim',), **options)
    cdh.long_name = 'hydrographic entity identifier'
    cdh.source = 'BD Carthage CdEntiteHy'
    cdh._Encoding = 'ascii'
    cdh.coordinates = 'x y'
    cdh.grid_mapping = 'crs: x, y'
    # unique identifier, unitless

    nym = grp_entity.createVariable('nym', 'S20', ('axdim',), **options)
    nym.long_name = 'river toponym'
    nym.source = 'BD Topo'
    nym._Encoding = 'utf-8'
    nym.coordinates = 'x y'
    nym.grid_mapping = 'crs: x, y'
    # name identifier, unitless

    hk = grp_entity.createVariable('hk', 'uint8', ('axdim',), **options)
    hk.long_name = 'Hack order of long profile'
    hk.coordinates = 'xo yo'
    hk.grid_mapping = 'crs: x, y'
    # unitless

    sto = grp_entity.createVariable('sto', 'uint8', ('axdim',), **options)
    sto.long_name = 'Strahler order at long profile origin'
    sto.coordinates = 'x y'
    sto.grid_mapping = 'crs: x, y'
    # unitless

    join = grp_entity.createVariable('join', 'uint32', ('axdim',), **options)
    join.long_name = 'junction profile coordinate'
    join.description = strip("""
        This variable contains an index on the profile dimension,
        which can be dereferenced to coordinate (axis, m)
        where :
            - `axis` is the downstream entity identifier
            - and `m` the location of the junction on `axis`
        """)

    # Long Profile Location Coordinates

    ap = grp_profile.createVariable('ap', 'uint32', ('profile',), **options)
    ap.long_name = 'long profile identifier of LP unit'

    mp = grp_profile.createVariable('mp', 'float64', ('profile',), **xym_options)
    mp.long_name = 'measure (distance) from origin of LP unit'
    mp.units = 'm'

    # Long Profile Location Characteristics

    xp = grp_profile.createVariable('xp', 'float64', ('profile',), **xym_options)
    xp.long_name = 'projected x location of LP unit'
    xp.units = 'm'
    xp.coordinates = 'ap mp'

    yp = grp_profile.createVariable('yp', 'float64', ('profile',), **xym_options)
    yp.long_name = 'projected y location of LP unit'
    yp.units = 'm'
    yp.coordinates = 'ap mp'

    st = grp_profile.createVariable('st', 'uint8', ('profile',), **options)
    st.long_name = 'Strahler order at LP unit location'
    # unitless
    st.coordinates = 'ap mp'

    # Long Profile Metrics

    drainage_options = dict(zlib=True, least_significant_digit=2, fill_value=0)

    drainage = grp_profile.createVariable('A', 'float32', ('profile',), **drainage_options)
    drainage.long_name = 'drainage area'
    drainage.units = 'km^2'
    drainage.coordinates = 'ap mp'

    z = grp_profile.createVariable('z', 'float32', ('profile',), **elevation_options)
    z.long_name = 'elevation'
    z.units = 'm'
    z.vertical_ref = crs.vertical_ref
    z.source = 'RGE Alti 5 m'
    z.coordinates = 'ap mp'

    dz = grp_profile.createVariable('dz', 'float32', ('profile',), **elevation_options)
    dz.long_name = 'slope, derived from z variable'
    dz.units = 'percent'
    dz.coordinates = 'ap mp'

    dzv = grp_profile.createVariable('dzv', 'float32', ('profile',), **elevation_options)
    dzv.long_name = 'slope of valley floor'
    dzv.units = 'percent'
    dzv.coordinates = 'ap mp'

    dzt = grp_profile.createVariable('dzt', 'float32', ('profile',), **elevation_options)
    dzt.long_name = 'slope of talweg'
    dzt.units = 'percent'
    dzt.coordinates = 'ap mp'

    acw = grp_profile.createVariable('acw', 'float32', ('profile',), **measurement_options)
    acw.long_name = 'active channel width'
    acw.units = 'm'
    acw.coordinates = 'ap mp'

    lfw = grp_profile.createVariable('lfw', 'float32', ('profile',), **measurement_options)
    lfw.long_name = 'low-flow water channel width'
    lfw.units = 'm'
    lfw.coordinates = 'ax m'

    axl = grp_profile.createVariable('axl', 'float32', ('profile',), **measurement_options)
    axl.long_name = 'intercepted reference axis length'
    axl.units = 'm'
    axl.coordinates = 'ap mp'

    twl = grp_profile.createVariable('twl', 'float32', ('profile',), **measurement_options)
    twl.long_name = 'intercepted talweg length'
    twl.units = 'm'
    twl.coordinates = 'ap mp'

    tcl = grp_profile.createVariable('tcl', 'float32', ('profile',), **measurement_options)
    tcl.long_name = 'intercepted water channels length'
    tcl.units = 'm'
    tcl.coordinates = 'ap mp'

    # Fluvial corridor widths

    fcw = grp_corridor.createVariable('fcw', 'float32', ('profile',), **measurement_options)
    fcw.long_name = 'fluvial corridor width'
    fcw.units = 'm'
    fcw.coordinates = 'ap mp'

    fcwk = grp_corridor.createVariable('fcwk', 'float32', ('profile', 'landcover'), **landcover_options)
    fcwk.long_name = 'fluvial corridor width per land cover class'
    fcwk.units = 'm'
    fcwk.coordinates = 'ap mp'

    fcwc = grp_corridor.createVariable('fcwc', 'float32', ('profile', 'landcover'), **landcover_options)
    fcwc.long_name = 'fluvial corridor width per land cover continuity class'
    fcwc.units = 'm'
    fcwc.coordinates = 'ap mp'

    # Swath Slice Coordinates

    # sx = grp_swath.createVariable('sx', 'uint32', ('swath',), **options)
    # sx.long_name = 'long profile identifier of LP unit'

    # sm = grp_swath.createVariable('sm', 'f8', ('swath',), **xym_options)
    # sm.long_name = 'measure (distance) from origin of LP unit'
    # sm.units = 'm'

    sp = grp_swath.createVariable('sp', 'uint32', ('swath',), **options)
    sp.long_name = 'long profile identifier of LP unit'
    sp.coordinates = 'ap mp'

    sx = grp_swath.createVariable('sx', 'float32', ('swath',), **xym_options)
    sx.long_name = 'distance from reference axis'
    sx.positive = 'left bank to right bank'
    sx.units = 'm'

    # Swath Slice Characteristics

    ds = grp_swath.createVariable('ds', 'uint32', ('swath',), **options)
    ds.long_name = 'swath slice density, ie. number of pixels used in slice values'
    ds.units = 'pixels'
    ds.coordinates = 'sp sx'

    # Swath Slice Elevations

    sz = grp_swath_elevation.createVariable('sz', 'float32', ('swath', 'quantile'), **elevation_options)
    sz.long_name = 'swath elevation'
    sz.units = 'm'
    sz.coordinates = 'sp sx'
    sz.vertical_ref = crs.vertical_ref

    hand = grp_swath_elevation.createVariable('hand', 'float32', ('swath', 'quantile'), **elevation_options)
    hand.long_name = 'height above nearest drainage'
    hand.units = 'm'
    hand.coordinates = 'sp sx'

    hvf = grp_swath_elevation.createVariable('hvf', 'float32', ('swath', 'quantile'), **elevation_options)
    hvf.long_name = 'height above valley floor'
    hvf.units = 'm'
    hvf.coordinates = 'sp sx'

    # Land cover metrics

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

    createFluvialCorridorProfileDataset(filename)

if __name__ == '__main__':
    #pylint:disable=no-value-for-parameter
    cli()
