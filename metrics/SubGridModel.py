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

def createFluvialCorridorSubGridDataset(filename, description=None):
    """
    Creates an empty NetCDF data structure
    """

    rootgrp = Dataset(filename, 'w')
    rootgrp.Conventions = 'CF-1.5'
    rootgrp.FCT = 'Fluvial Corridor Toolbox SubGrid Model 1.0.5'

    if description:
        rootgrp.description = description

    # Common variable options

    options = dict(zlib=True)

    xy_options = dict(
        least_significant_digit=2,
        zlib=True
    )

    area_options = dict(
        least_significant_digit=3,
        fill_value=-99999.0,
        zlib=True
    )

    length_options = dict(
        least_significant_digit=3,
        fill_value=-99999.0,
        zlib=True
    )

    pop_options = dict(
        least_significant_digit=3,
        fill_value=-99999.0,
        zlib=True
    )

    # Groups

    grp_landcover = rootgrp.createGroup('landcover')
    grp_population = rootgrp.createGroup('population')

    # Dimensions

    rootgrp.createDimension('Y', None)
    rootgrp.createDimension('X', None)
    grp_landcover.createDimension('GRID', None)
    grp_landcover.createDimension('K', 10)

    # Spatial Reference System

    crs = rootgrp.createVariable('lambert_conformal_conic', '|S1')
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
    crs.crs_wkt = crs.spatial_ref
    crs.GeoTransform = '829997.5 5 0 6680002.5 0 -5'

    vertical_ref = strip("""
        VERT_CS["NGF-IGN69 height",
            VERT_DATUM["Nivellement General de la France - IGN69",2005,
                AUTHORITY["EPSG","5119"]],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Up",UP],
            AUTHORITY["EPSG","5720"]]
    """)

    # Fixed time variables

    time_landcover = rootgrp.createVariable('time_landcover', 'int16', ())
    time_landcover.long_name = 'temporality of landcover data'
    time_landcover.units = 'year'
    time_landcover.calendar = 'gregorian'
    time_landcover[...] = 2018

    time_pop = rootgrp.createVariable('time_pop', 'int16', ())
    time_pop.long_name = 'temporality of population data'
    time_pop.units = 'year'
    time_pop.calendar = 'gregorian'
    time_pop[...] = 2015

    # Coordinates

    x = rootgrp.createVariable('x', 'float64', ('X',), **xy_options)
    x.standard_name = 'projection_x_coordinate'
    x.long_name = 'x coordinate'
    x.units = 'm'

    y = rootgrp.createVariable('y', 'float64', ('Y',), **xy_options)
    y.standard_name = 'projection_y_coordinate'
    y.long_name = 'y coordinate'
    y.units = 'm'

    grid = grp_landcover.createVariable('idx', 'uint32', ('GRID',), **options)
    grid.long_name = 'raster grid index'
    grid.compress = 'Y X'

    xg = grp_landcover.createVariable('xg', 'float64', ('GRID',), **options)
    xg.long_name = 'x coordinate of grid cell (index)'
    xg.units = 'm'

    yg = grp_landcover.createVariable('yg', 'float64', ('GRID',), **options)
    yg.long_name = 'y coordinate of grid cell (index)'
    yg.units = 'm'

    # Variables

    drainage = rootgrp.createVariable('A', 'float32', ('Y', 'X'), **area_options)
    drainage.long_name = 'drainage area'
    drainage.units = 'km^2'
    drainage.grid_mapping = 'lambert_conformal_conic: x, y'

    dlen = rootgrp.createVariable('dlen', 'float32', ('Y', 'X'), **length_options)
    dlen.long_name = 'drainage length'
    dlen.units = 'km'
    dlen.grid_mapping = 'lambert_conformal_conic: x, y'

    xo = rootgrp.createVariable('xo', 'uint16', ('Y', 'X'), **options)
    xo.long_name = 'cell outlet x index'
    xo.units = 'm'

    yo = rootgrp.createVariable('yo', 'uint16', ('Y', 'X'), **options)
    yo.long_name = 'cell outlet y index'
    yo.units = 'm'

    zo = rootgrp.createVariable('zo', 'float32', ('Y', 'X'), **xy_options)
    zo.standard_name = 'height'
    zo.long_name = 'cell outlet elevation'
    zo.units = 'm'
    zo.vertical_ref = vertical_ref
    zo.coordinates = 'x(xo) y(yo)'
    zo.grid_mapping = 'lambert_conformal_conic: x, y'

    jnext = rootgrp.createVariable('jnext', 'uint32', ('Y', 'X'), **options)
    jnext.long_name = 'downstream cell x coordinate'
    jnext.units = 'm'

    inext = rootgrp.createVariable('inext', 'uint32', ('Y', 'X'), **options)
    inext.long_name = 'downstream cell y coordinate'
    inext.units = 'm'

    # Land Cover Metrics

    lck = grp_landcover.createVariable('lck', 'S20', ('K',))
    lck.long_name = 'land cover class description'

    lc = grp_landcover.createVariable('lc', 'float32', ('K', 'Y', 'X'), **area_options)
    lc.long_name = 'area of land cover class in cell (x, y)'
    lc.units = 'ha'
    lc.grid_mapping = 'lambert_conformal_conic: x, y'

    lcum = grp_landcover.createVariable('lcum', 'float32', ('K', 'GRID'), **area_options)
    lcum.long_name = 'cumulative upstream area of land cover class k'
    lcum.units = 'km^2'
    lcum.coordinates = 'xg yg'
    lcum.grid_mapping = 'lambert_conformal_conic: xg, yg'

    # Population Variables

    # uint16 ? max = 65536
    pop = grp_population.createVariable(
        'pop', 'uint32', ('Y', 'X'),
        fill_value=np.iinfo('uint32').max,
        **options)
    pop.long_name = 'population from 2015 census'
    pop.units = 'inhabitants'
    pop.source = 'INSEE, Filosofi 2015, Données au carreau de 200 m (y compris données imputées)'
    pop.references = 'https://www.insee.fr/fr/statistiques/4176290'

    popcum = grp_population.createVariable('popcum', 'float32', ('Y', 'X'), **pop_options)
    popcum.long_name = 'cumulative upstream population from 2015 census'
    popcum.units = 'thousands'
    popcum.source = 'INSEE, Filosofi 2015, Données au carreau de 200 m (y compris données imputées)'
    popcum.references = 'https://www.insee.fr/fr/statistiques/4176290'


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

    createFluvialCorridorSubGridDataset(filename)

if __name__ == '__main__':
    #pylint:disable=no-value-for-parameter
    cli()
