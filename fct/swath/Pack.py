# coding: utf-8
# pylint:disable=invalid-name

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

from netCDF4 import Dataset
import os
import click
import numpy as np

import fiona


workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def WriteEntities():
    pass

def WriteProfile(entity):
    pass

def WriteStdMetrics(entity):
    pass

def WriteCorridorMetrics(entity):
    pass

def WriteElevationSwath(dst, entity, swath):
    """
    DOCME
    """

    sp = dst['entity/swath/sp']
    sx = dst['entity/swath/sx']
    ds = dst['entity/swath/ds']
    sz = dst['entity/swath/elevation/sz']
    hand = dst['entity/swath/elevation/hand']
    hvf = dst['entity/swath/elevation/hvf']

    # if parallel:

    #     sp.set_collective(True)
    #     sx.set_collective(True)
    #     ds.set_collective(True)
    #     sz.set_collective(True)
    #     hand.set_collective(True)
    #     hvf.set_collective(True)

    swathfile = os.path.join(workdir, 'SWATH', 'AX%03d_SWATH_%04d.npz' % (entity, swath))

    data = np.load(swathfile, allow_pickle=True)
    sz_values = data['sz']
    hand_values = data['hand']
    hvf_values = data['hvf']

    idx0 = sp.shape[0]
    idx1 = idx0 + sz_values.shape[0]

    # TODO map swath to ID in profile dimension
    sp[idx0:idx1] = swath
    sx[idx0:idx1] = data['x']
    ds[idx0:idx1] = data['density']

    sz[idx0:idx1, :] = sz_values
    hand[idx0:idx1, :] = hand_values

    if hvf_values.size > 0:
        hvf[idx0:idx1, :] = hvf_values
    else:
        hvf[idx0:idx1, :] = hvf._FillValue

def WriteLandCoverSwath(dst, entity, swath, landcover='continuity'):
    """
    DOCME
    """

    sp = dst['entity/swath/sp']
    sx = dst['entity/swath/sx']
    ds = dst['entity/swath/ds']

    if landcover == 'continuity':
        swathfile = os.path.join(workdir, 'OCS', 'AX%03d_SWATH_CONTINUITY_%04d.npz' % (entity, swath))
        lc = dst['entity/swath/landcover/lcc']
        scw = dst['entity/swath/landcover/scwc']
    elif landcover == 'raw':
        swathfile = os.path.join(workdir, 'OCS', 'AX%03d_SWATH_RAW_%04d.npz' % (entity, swath))
        lc = dst['entity/swath/landcover/lck']
        scw = dst['entity/swath/landcover/scwk']
    else:
        return

    # if parallel:

    #     sp.set_collective(True)
    #     sx.set_collective(True)
    #     ds.set_collective(True)
    #     lc.set_collective(True)
    #     scw.set_collective(True)

    data = np.load(swathfile, allow_pickle=True)

    classes = data['classes']
    values = data['swath']

    idx0 = sp.shape[0]
    idx1 = idx0 + values.shape[0]

    # TODO map swath to ID in profile dimension
    sp[idx0:idx1] = swath
    sx[idx0:idx1] = data['x']
    ds[idx0:idx1] = data['density']

    lc[swath, :len(classes)] = classes
    scw[idx0:idx1, :len(classes)] = values[:, :]

def test(entity):

    dst = Dataset('Test.nc', 'r+')

    dgo_shapefile = os.path.join(workdir, 'AX%03d_DGO.shp' % entity)

    with fiona.open(dgo_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                swath = feature['properties']['GID']
                WriteElevationSwath(dst, entity, swath)
                WriteLandCoverSwath(dst, entity, swath, 'raw')
                WriteLandCoverSwath(dst, entity, swath, 'continuity')

    dst.close()

# def test_parallel(entity):

#     #pylint:disable=import-outside-toplevel,c-extension-no-member
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()

#     dst = Dataset('Test2.nc', 'r+', parallel=True)

#     dgo_shapefile = os.path.join(workdir, 'AX%03d_DGO.shp' % entity)

#     with fiona.open(dgo_shapefile) as fs:
#         # with click.progressbar(fs) as iterator:
#         for k, feature in enumerate(fs):
#             if k % size == rank:

#                 swath = feature['properties']['GID']
#                 WriteElevationSwath(dst, entity, swath, True)
#                 WriteLandCoverSwath(dst, entity, swath, 'raw', True)
#                 WriteLandCoverSwath(dst, entity, swath, 'continuity', True)

#     dst.close()

if __name__ == '__main__':
    test(1044)
    # test_parallel(1044)
