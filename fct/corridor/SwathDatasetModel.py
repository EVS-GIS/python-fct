# coding: utf-8

"""
NetCDF Data Format for Swath Profiles

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np
from netCDF4 import Dataset

import fiona
import click

from ..config import config

def strip(s):
    # return re.sub(' {2,}', ' ', s.strip())
    return s.rstrip()

def CreateSwathDataset(axis, dataset, **kwargs):
    """
    Creates an empty NetCDF data structure
    """

    filename = config.filename(dataset, axis=axis, **kwargs)

    rootgrp = Dataset(filename, 'w')
    rootgrp.Conventions = 'CF-1.8'
    rootgrp.FCT = 'Fluvial Corridor Toolbox Swath Profile 1.0.5'

    # Dimensions

    # do not create dimension for fixed coordinate
    # rootgrp.createDimension('axis', None)
    rootgrp.createDimension('measure', None)
    rootgrp.createDimension('distance', None)
    rootgrp.createDimension('i', None)

    # Coordinate reference system

    coord_options = dict(
        zlib=True,
        complevel=9,
        least_significant_digit=2
    )

    index_options = dict(
        zlib=True,
        complevel=9
    )

    # Coordinates

    var_axis = rootgrp.createVariable('axis', 'uint32')
    var_axis.long_name = 'axis identifier'
    var_axis[...] = axis

    measure = rootgrp.createVariable('measure', 'float32', ('measure',), **coord_options)
    measure.long_name = 'measure dimension'
    measure.units = 'm'

    distance = rootgrp.createVariable('distance', 'float32', ('distance',), **coord_options)
    distance.long_name = 'distance dimension'
    distance.units = 'm'

    index = rootgrp.createVariable('i', 'uint64', ('i',), **index_options)
    index.long_name = 'swath slice index'
    index.compress = 'measure distance'
    index.cf_role = 'profile_id'

    # Variables

    swath = rootgrp.createVariable('swath', 'uint32', ('measure',))
    swath.long_name = 'swath identifier'
    swath.coordinates = 'axis measure'

    mi = rootgrp.createVariable('mi', 'float32', ('i',), **coord_options)
    mi.long_name = 'measure coordinate of swath slice i'
    mi.units = 'm'

    di = rootgrp.createVariable('di', 'float32', ('i',), **coord_options)
    di.long_name = 'distance coordinate of swath slice i'
    di.units = 'm'

    return rootgrp

def CreateElevationVariables(grp_elevation):

    # Dimensions

    grp_elevation.createDimension('quantile', 5)
    quantile = grp_elevation.createVariable('quantile', 'float32', ('quantile',))
    quantile.long_name = 'quantile dimension'
    quantile[:] = [.05, .25, .5, .75, .95]

    heights = np.arange(5.0, 15.5, 0.5)
    grp_elevation.createDimension('height', len(heights))
    height = grp_elevation.createVariable('height', 'float32', ('height',))
    height.long_name = 'height dimension'
    height[:] = heights

    # Variables

    count_options = dict(
        zlib=True,
        complevel=9
    )

    z_options = dict(
        zlib=True,
        complevel=9,
        least_significant_digit=2
    )

    area = grp_elevation.createVariable('vba', 'uint32', ('measure', 'height'), **count_options)
    area.long_name = 'valley bottom area of swath measured at height h'
    area.coordinates = 'axis measure'
    area.units = 'pixels'

    density = grp_elevation.createVariable('density', 'uint32', ('i',), **count_options)
    density.long_name = 'swath slice total area'
    density.coordinates = 'axis mi di'
    density.units = 'pixels'

    absz = grp_elevation.createVariable('absz', 'float32', ('i', 'quantile'), **z_options)
    absz.long_name = 'absolute elevation'
    # absz.standard_name = 'z'
    absz.units = 'm'
    absz.vertical_ref= strip("""
        VERT_CS["NGF-IGN69 height",
            VERT_DATUM["Nivellement General de la France - IGN69",2005,
                AUTHORITY["EPSG","5119"]],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Up",UP],
            AUTHORITY["EPSG","5720"]]
    """)
    absz.coordinates = 'axis mi di'

    hand = grp_elevation.createVariable('hand', 'float32', ('i', 'quantile'), **z_options)
    hand.long_name = 'height above nearest drainage'
    hand.units = 'm'
    hand.coordinates = 'axis mi di'

    havf = grp_elevation.createVariable('havf', 'float32', ('i', 'quantile'), **z_options)
    havf.long_name = 'height above valley floor'
    havf.units = 'm'
    havf.coordinates = 'axis mi di'

def CreateLandCoverVariables(grp):

    # Dimensions

    grp.createDimension('landcover', 9)
    landcover = grp.createVariable('landcover', str, ('landcover',))
    landcovers = [
        'Water Channel',
        'Gravel Bars',
        'Natural Open',
        'Forest',
        'Grassland',
        'Crops',
        'Diffuse Urban',
        'Dense Urban',
        'Infrastructures'
    ]

    for k, value in enumerate(landcovers):
        landcover[k] = value

    grp.createDimension('ref', 2)
    ref = grp.createVariable('ref', str, ('ref',))
    refs = [
        'refaxis',
        'talweg'
    ]

    for k, value in enumerate(refs):
        ref[k] = value

    # Variables

    count_options = dict(
        zlib=True,
        complevel=9
    )

    density = grp.createVariable('density', 'uint32', ('i', 'ref'), **count_options)
    density.long_name = 'swath slice total area, at distance di from ref axis'
    density.coordinates = 'axis mi di'
    density.units = 'pixels'

    ak = grp.createVariable('ak', 'uint16', ('i', 'landcover', 'ref'), **count_options)
    ak.long_name = 'area of landcover class k, at distance di from ref axis'
    ak.coordinates = 'axis mi di'
    ak.units = 'pixels'


def PackElevationSwathProfiles(
        axis,
        destination='ax_swath_elevation_pack',
        **kwargs):
    """
    Pack elevation swath profiles into one netCDF archive

    Parameters
    ----------

    axis: int

        Axis identifier

    Keyword Parameters
    ------------------

    destination: str, logical name

        Output file
    """

    dgo_shapefile = config.filename('ax_swath_features', axis=axis)

    dataset = CreateSwathDataset(axis, destination, **kwargs)
    CreateElevationVariables(dataset)

    measures = list()
    distances = set()

    with fiona.open(dgo_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for k, feature in enumerate(iterator):

                gid = feature['properties']['GID']
                measure = feature['properties']['M']

                swathfile = config.filename('ax_swath_elevation', axis=axis, gid=gid)
                data = np.load(swathfile, allow_pickle=True)

                distance = data['x']
                density = data['density']
                varea = data['varea']
                absz = data['sz']
                hand = data['hand']
                havf = data['hvf']

                index = dataset['i'].shape[0]
                size = len(distance)

                dataset['swath'][k] = gid
                dataset['i'][index:index+size] = np.arange(index, index+size)
                dataset['mi'][index:index+size] = measure
                dataset['di'][index:index+size] = distance

                dataset['vba'][k, :] = varea
                dataset['density'][index:index+size] = density
                dataset['absz'][index:index+size, :] = absz
                dataset['hand'][index:index+size, :] = hand

                if havf.shape == hand.shape:
                    dataset['havf'][index:index+size, :] = havf
                else:
                    dataset['havf'][index:index+size, :] = np.full_like(hand, np.nan)

                measures.append(measure)
                distances.update(distance)

    dataset['measure'][:] = measures
    dataset['distance'][:] = list(sorted(distances))

    dataset.close()

def PackLandCoverSwathProfiles(
        axis,
        destination='ax_swath_landcover_pack',
        subset='lancover',
        **kwargs):
    """
    Pack landcover swath profiles into one netCDF archive

    Parameters
    ----------

    axis: int

        Axis identifier

    Keyword Parameters
    ------------------

    destination: str, logical name

        Output file

    subset: str

        Landcover swath subset to export,
        `landcover` (total landcover area) or
        `continuity` (continuous landcover buffer area)

    Other keyword parameters are passed to filename templates.
    """

    kwargs.update(subset=subset.upper())

    dgo_shapefile = config.filename('ax_swath_features', axis=axis)

    dataset = CreateSwathDataset(axis, destination, **kwargs)
    CreateLandCoverVariables(dataset)

    measures = list()
    distances = set()

    with fiona.open(dgo_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for k, feature in enumerate(iterator):

                gid = feature['properties']['GID']
                measure = feature['properties']['M']

                swathfile = config.filename('ax_swath_landcover', axis=axis, gid=gid, **kwargs)
                data = np.load(swathfile, allow_pickle=True)

                distance = data['x']
                density = data['density']
                classes = data['classes']
                swath = data['swath']

                index = dataset['i'].shape[0]
                size = len(distance)

                dataset['swath'][k] = gid
                dataset['i'][index:index+size] = np.arange(index, index+size)
                dataset['mi'][index:index+size] = measure
                dataset['di'][index:index+size] = distance

                dataset['density'][index:index+size, :] = density
                dataset['ak'][index:index+size, :, :] = np.zeros((size, 9, 2), dtype='uint16')

                for i, klass in enumerate(classes):

                    if klass == 255:
                        continue

                    dataset['ak'][index:index+size, klass-1, :] = swath[:, i, :]

                measures.append(measure)
                distances.update(distance)

    dataset['measure'][:] = measures
    dataset['distance'][:] = list(sorted(distances))

    dataset.close()

def test(axis):

    config.default()

    PackElevationSwathProfiles(axis)
    PackLandCoverSwathProfiles(axis, subset='TOTAL_BDT')
    PackLandCoverSwathProfiles(axis, subset='CONT_CESBIO')
    PackLandCoverSwathProfiles(axis, subset='CONT_BDT')
    PackLandCoverSwathProfiles(axis, subset='CONT_NOINFRA')
