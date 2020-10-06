# coding: utf-8

"""
Planform signal, talweg shift with respect to given reference axis

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import numpy as np
from scipy.spatial import cKDTree

import fiona
import xarray as xr

from .. import terrain_analysis as ta
from ..config import config
from ..metadata import set_metadata
from ..plotting.PlotCorridor import (
    SetupPlot,
    SetupMeasureAxis,
    FinalizePlot
)

def PlanformShift(axis, refaxis_name='ax_refaxis'):
    """
    Project talweg linestring on linear reference axis,
    yielding a long profile signal of the amplitude of talweg shift.
    """

    refaxis_shapefile = config.filename(refaxis_name, axis=axis)
    talweg_shapefile = config.filename('ax_talweg', axis=axis)

    with fiona.open(refaxis_shapefile) as fs:
        refaxis = np.concatenate([f['geometry']['coordinates'] for f in fs])

    with fiona.open(talweg_shapefile) as fs:
        talweg = np.concatenate([f['geometry']['coordinates'] for f in fs])

    midpoints = 0.5 * (refaxis[1:, :] + refaxis[:-1, :])
    index = cKDTree(midpoints[:, :2], balanced_tree=True)
    _, nearest = index.query(talweg[:, :2], k=1)

    talweg = np.float32(talweg[:, :2])
    refaxis = np.float32(refaxis[:, :2])

    talweg_measure = np.cumsum(np.linalg.norm(talweg[1:] - talweg[:-1], axis=1))
    talweg_measure = np.concatenate([np.zeros(1), talweg_measure])

    # x = refaxis measure
    x = np.cumsum(np.linalg.norm(refaxis[1:] - refaxis[:-1], axis=1))
    x = np.concatenate([np.zeros(1), x])

    _, signed_distance, location = ta.signed_distance(
        refaxis[nearest],
        refaxis[nearest+1],
        talweg)

    xt = x[nearest] + location * (x[nearest+1] - x[nearest])

    dataset = xr.Dataset(
        {
            'talweg_measure': ('measure', talweg_measure),
            'talweg_shift': ('measure', signed_distance)
        },
        coords={
            'axis': axis,
            'measure': np.max(x) - xt
        }
    )

    set_metadata(dataset, 'metrics_planform')
    dataset['talweg_shift'].attrs['reference'] = refaxis_name

    return dataset

def PlanformAmplitude(planform_shift, window=20):
    """
    see :
    - https://fr.wikipedia.org/wiki/Valeur_efficace
    - https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal

    >>> def window_rms(a, window_size):
    >>>   a2 = np.power(a,2)
    >>>   window = np.ones(window_size)/float(window_size)
    >>>   return np.sqrt(np.convolve(a2, window, 'valid'))
    """

    return np.sqrt(
        2 * np.square(planform_shift)
        .rolling(measure=window, min_periods=1, center=True)
        .mean()
    )

def WritePlanforMetrics(axis, dataset):

    output = config.filename('metrics_planform', axis=axis)

    dataset.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'talweg_measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'talweg_shift': dict(zlib=True, complevel=9, least_significant_digit=2),
        }
    )

def PlotPlanformShift(axis, data, filename=None):

    x = data['measure']
    y = data['talweg_shift']

    fig, ax = SetupPlot()
    ax.plot(x, y)
    SetupMeasureAxis(ax, x, title='Location from source (m)')
    FinalizePlot(fig, ax, filename=filename)
