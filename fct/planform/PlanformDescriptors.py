"""
Planform signal,
talweg shift with respect to given reference axis
"""

import numpy as np
from scipy.spatial import cKDTree

import click
import fiona
from shapely.geometry import LineString
import xarray as xr

from .. import terrain_analysis as ta
from ..config import (
    DatasetParameter,
    LiteralParameter
)
from ..metadata import set_metadata
from ..plotting.PlotCorridor import (
    SetupPlot,
    SetupMeasureAxis,
    FinalizePlot
)

class Parameters:
    """
    Planform metrics parameters
    """

    talweg = DatasetParameter(
        'stream/talweg polyline',
        type='input')

    refaxis = DatasetParameter(
        'reference axis used to measure talweg shift',
        type='input')

    output = DatasetParameter(
        'destination netcdf dataset',
        type='output')

    sample_distance = LiteralParameter(
        'distance between sampled talweg points')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.talweg = 'network-cartography-ready'
            self.refaxis = 'refaxis'
            self.output = 'metrics_planform'

        else:

            self.talweg = dict(key='ax_talweg', axis=axis)
            self.refaxis = dict(key='ax_refaxis', axis=axis)
            self.output = dict(key='metrics_planform', axis=axis)

        self.sample_distance = 20.0

def PlanformCurvature(talweg):
    """
    Direction angle change between two talweg points
    (TODO: normalize by distance in order to get dφ/ds
    rather than dφ)

    Parameter:

    - talweg:
        numpy array of points (x, y) with dimension (n, 2)
        representing the talweg linestring
    """

    dxy = np.diff(talweg, axis=0)
    dx = dxy[:, 0]
    dy = dxy[:, 1]
    phi = np.arctan2(dy, dx)

    # phi2 = np.concatenate([phi[1:], np.zeros(1)])

    # curvature = np.abs(phi2 - phi)

    # change = (np.sign(phi) != np.sign(phi2))
    # curvature2 = np.abs(phi) + np.abs(phi2)
    # complement2 = 2*np.pi - curvature2
    
    # set1 = change & (curvature2 > complement2)
    # curvature[set1] = complement2[set1]
    
    # set2 = change & (curvature2 <= complement2)
    # curvature[set2] = curvature2[set2]

    curvature = np.diff(phi)
    curvature[curvature < -np.pi] += 2*np.pi
    curvature[curvature > np.pi] -= 2*np.pi

    return np.concatenate([np.zeros(1), curvature, np.zeros(1)])

def DirectionAngle(talweg, refaxis, nearest):
    """
    Direction angle in radians of the curve
    with respect to reference axis
    """

    dxy = np.diff(talweg, axis=0)
    dx = dxy[:, 0]
    dy = dxy[:, 1]
    phi = np.arctan2(dy, dx)

    refdxy = refaxis[nearest+1] - refaxis[nearest]
    refdx = refdxy[:, 0]
    refdy = refdxy[:, 1]
    refdirection = np.arctan2(refdy, refdx)

    phi = phi - refdirection[:-1]
    phi[phi < -np.pi] += 2*np.pi
    phi[phi > np.pi] -= 2*np.pi

    return np.concatenate([phi, phi[[-1]]])

# def DirectionAngle2(talweg_samples, talweg, refaxis, ref_nearest):
#     """
#     Direction angle in radians of the curve
#     with respect to reference axis
#     """

#     midpoints = 0.5 * (talweg[1:, :] + talweg[:-1, :])
#     index = cKDTree(midpoints, balanced_tree=True)
#     _, talweg_nearest = index.query(talweg_samples[:, :2], k=1)

#     dxy = talweg[talweg_nearest+1] - talweg[talweg_nearest]
#     dx = dxy[:, 0]
#     dy = dxy[:, 1]
#     phi = np.arctan2(dy, dx)

#     refdxy = refaxis[ref_nearest+1] - refaxis[ref_nearest]
#     refdx = refdxy[:, 0]
#     refdy = refdxy[:, 1]
#     refdirection = np.arctan2(refdy, refdx)

#     phi = phi - refdirection
#     phi[phi < -np.pi] += 2*np.pi
#     phi[phi > np.pi] -= 2*np.pi

#     return phi

def calculate_planform_variables(axis, talweg, params: Parameters) -> xr.Dataset:
    """
    Per axis variables calculation
    """

    refaxis_shapefile = params.refaxis.filename(tileset=None)
    distance = params.sample_distance

    with fiona.open(refaxis_shapefile) as fs:

        refaxis = np.concatenate([
            f['geometry']['coordinates']
            for f in fs
            if f['properties']['AXIS'] == axis
        ])

    if distance > 0:

        talweg_geometry = LineString(talweg)
        talweg_measure = np.arange(0, talweg_geometry.length, distance)
        talweg_samples = np.array([
            talweg_geometry.interpolate(x).coords[0]
            for x in talweg_measure
        ])

    else:

        talweg_samples = np.float32(talweg[:, :2])
        talweg_measure = np.cumsum(np.linalg.norm(talweg_samples[1:] - talweg_samples[:-1], axis=1))
        talweg_measure = np.concatenate([np.zeros(1), talweg_measure])

    talweg = np.float32(talweg[:, :2])
    talweg_samples = np.float32(talweg_samples[:, :2])
    refaxis = np.float32(refaxis[:, :2])

    midpoints = 0.5 * (refaxis[1:, :] + refaxis[:-1, :])
    index = cKDTree(midpoints, balanced_tree=True)
    _, nearest = index.query(talweg_samples[:, :2], k=1)

    # talweg_measure = np.cumsum(np.linalg.norm(talweg[1:] - talweg[:-1], axis=1))
    # talweg_measure = np.concatenate([np.zeros(1), talweg_measure])

    # x = refaxis measure
    x = np.cumsum(np.linalg.norm(refaxis[1:] - refaxis[:-1], axis=1))
    x = np.concatenate([np.zeros(1), x])

    _, signed_distance, location = ta.signed_distance(
        refaxis[nearest],
        refaxis[nearest+1],
        talweg_samples)

    talweg_curvature = PlanformCurvature(talweg_samples)
    talweg_direction_angle = DirectionAngle(talweg_samples, refaxis, nearest)
    # talweg_direction_angle = DirectionAngle2(talweg_samples, talweg, refaxis, nearest)

    xt = x[nearest] + location * (x[nearest+1] - x[nearest])

    dataset = xr.Dataset(
        {
            'talweg_shift': ('sample', signed_distance),
            'talweg_curvature': ('sample', np.float32(talweg_curvature)),
            'talweg_direction_angle': ('sample', np.float32(talweg_direction_angle)),
            'ref_measure': ('sample', np.float32(np.max(x) - xt)),
        },
        coords={
            'axis': ('sample', np.full_like(talweg_measure, axis, dtype='uint32')),
            'talweg_measure': ('sample', np.float32(talweg_measure)),
        }
    )

    return dataset

def PlanformVariables(params: Parameters, ax=None) -> xr.Dataset:
    """
    Project talweg linestring on linear reference axis,
    yielding a long profile signal of the amplitude of talweg shift.

    >>> with open('/media/crousson/Backup2/RhoneMediterranee/GrandsAxes/AXES/AX0002/METRICS/MISC_TALWEG.csv', 'w') as fp:
    >>>     fp.write('x;shift;curv;direction\n')
    >>>     for k in range(len(data.talweg_measure)):
    >>>         fp.write('%f;%f;%f;%f\n' % (data['talweg_measure'][k], data['talweg_shift'][k], data['talweg_curvature'][k], data['talweg_direction_angle'][k]))
    """

    talweg_shapefile = params.talweg.filename(tileset=None)

    values = list()

    with fiona.open(talweg_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                axis = feature['properties']['AXIS']

                if ax is not None and axis != ax:
                    continue

                talweg = np.asarray(feature['geometry']['coordinates'])
                axis_values = calculate_planform_variables(axis, talweg, params)
                values.append(axis_values)

    dataset = xr.concat(values, 'sample', 'all')

    set_metadata(dataset, 'metrics_planform')
    dataset['talweg_shift'].attrs['reference'] = params.refaxis.name

    return dataset

def PlanformEnvelope(axis, measure, amplitude, params: Parameters):

    refaxis_shapefile = params.refaxis.filename(tileset=None)
    talweg_shapefile = params.talweg.filename(tileset=None)

    with fiona.open(refaxis_shapefile) as fs:
        refaxis = np.concatenate([f['geometry']['coordinates'] for f in fs])

    with fiona.open(talweg_shapefile) as fs:

        talweg = np.concatenate([f['geometry']['coordinates'] for f in fs])
        talweg_geometry = LineString(talweg)
        talweg_samples = np.array([
            talweg_geometry.interpolate(x).coords[0]
            for x in measure
        ])

    refaxis = np.float32(refaxis[:, :2])
    talweg_samples = np.float32(talweg_samples[:, :2])

    midpoints = 0.5 * (refaxis[1:, :] + refaxis[:-1, :])
    index = cKDTree(midpoints, balanced_tree=True)
    _, nearest = index.query(talweg_samples[:, :2], k=1)

    _, _, location = ta.signed_distance(
        refaxis[nearest],
        refaxis[nearest+1],
        talweg_samples)

    direction = refaxis[nearest+1] - refaxis[nearest]
    normal = (
        np.column_stack([-direction[:, 1], direction[:, 0]]) /
        np.linalg.norm(direction, axis=1)[:, np.newaxis]
    )

    projection = refaxis[nearest] + location[:, np.newaxis]*direction

    left = projection + amplitude[:, np.newaxis]*normal
    right = projection - amplitude[:, np.newaxis]*normal

    return projection, left, right

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

# def WritePlanforMetrics(axis, dataset):

#     output = config.filename('metrics_planform', axis=axis)

#     dataset.to_netcdf(
#         output, 'w',
#         encoding={
#             'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
#             'talweg_measure': dict(zlib=True, complevel=9, least_significant_digit=0),
#             'talweg_shift': dict(zlib=True, complevel=9, least_significant_digit=2),
#         }
#     )

def PlotPlanformShift(data: xr.Dataset, axis: int = None, filename: str = None):

    if axis is not None:
        data = data.sel(axis=axis)

    x = data.talweg_measure
    y = data.talweg_shift

    fig, ax = SetupPlot()
    ax.plot(x, y)
    SetupMeasureAxis(ax, x, direction=1, title='Location from source (m)')
    FinalizePlot(fig, ax, filename=filename)
