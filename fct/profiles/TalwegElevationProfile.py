"""
Planform signal,
talweg shift with respect to given reference axis
"""

import logging
from multiprocessing import Pool
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import lstsq

import click
import fiona
import rasterio as rio
from shapely.geometry import LineString, asShape
import xarray as xr

from ..cli import starcall
from ..tileio import as_window
from ..config import (
    DatasetParameter,
    LiteralParameter
)
from ..corridor.ValleyBottomFeatures import MASK_VALLEY_BOTTOM
from ..metadata import set_metadata
# from ..plotting.PlotCorridor import (
#     SetupPlot,
#     SetupMeasureAxis,
#     FinalizePlot
# )

class Parameters:
    """
    Planform metrics parameters
    """

    dem = DatasetParameter(
        'elevation raster (DEM)',
        type='input')

    nearest = DatasetParameter(
        'nearest drainage axis (raster)',
        type='input')

    measure = DatasetParameter(
        'location along nearest reference axis (raster)',
        type='input')

    valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')

    talweg = DatasetParameter(
        'stream/talweg polyline',
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

        self.dem = 'dem'

        if axis is None:

            self.talweg = 'network-cartography-ready'
            self.nearest = 'nearest_drainage_axis'
            self.measure = 'axis_measure'
            self.valley_bottom = 'valley_bottom_final'
            self.output = 'metrics_planform'

        else:

            self.talweg = dict(key='ax_talweg', axis=axis)
            self.nearest = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
            self.output = dict(key='metrics_planform', axis=axis)

        self.sample_distance = 10.0

def extract_talweg_data(axis, talweg, params: Parameters) -> xr.Dataset:

    distance = params.sample_distance

    if distance > 0:

        talweg_geometry = LineString(talweg)
        talweg_measure = np.arange(0, talweg_geometry.length, distance)
        talweg_xy = np.array([
            talweg_geometry.interpolate(x).coords[0]
            for x in talweg_measure
        ])
        talweg_xy = np.float32(talweg_xy[:, :2])

    else:

        talweg_xy = np.float32(talweg[:, :2])
        talweg_measure = np.cumsum(np.linalg.norm(talweg_samples[1:] - talweg_samples[:-1], axis=1))
        talweg_measure = np.concatenate([np.zeros(1), talweg_measure])

    with rio.open(params.nearest.filename()) as ds:

        nearest = np.array(list(ds.sample(talweg_xy, 1)))
        nearest = nearest.squeeze()
        valid = (nearest == axis)

    with rio.open(params.measure.filename()) as ds:

        measure = np.array(list(ds.sample(talweg_xy, 1)), dtype='float32')
        # measure = measure[:, 0]
        measure = measure.squeeze()
        valid = valid & (measure != ds.nodata)

    with rio.open(params.dem.filename()) as ds:

        z = np.array(list(ds.sample(talweg_xy, 1)), dtype='float32')
        # z = z[:, 0]
        z = z.squeeze()
        valid = valid & (z != ds.nodata)

    return xr.Dataset(
        {
            'x': ('sample', talweg_xy[valid, 0]),
            'y': ('sample', talweg_xy[valid, 1]),
            'z': ('sample', z[valid]),
            'measure': ('sample', measure[valid])
        }, coords={
            'axis': ('sample', np.full_like(talweg_measure[valid], axis, dtype='uint32')),
            'talweg_measure': ('sample', talweg_measure[valid])
        }
    )

def talweg_elevation_mp(params: Parameters, processes: int = 6, **kwargs) -> xr.Dataset:

    talweg_shapefile = params.talweg.filename(tileset=None)

    def length():

        with fiona.open(talweg_shapefile) as fs:
            return len(fs)

    def arguments():

        with fiona.open(talweg_shapefile) as fs:
            for feature in fs:

                axis = feature['properties']['AXIS']
                talweg = np.asarray(feature['geometry']['coordinates'])

                yield (
                    extract_talweg_data,
                    axis,
                    talweg,
                    params,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            values = list(iterator)

    return xr.concat(values, 'sample', 'all')

def TalwegElevation(params: Parameters, processes: int = 6, **kwargs) -> xr.Dataset:
    """
    Extract regularly spaced elevation data point and coordinates
    along talweg polyline
    """

    if processes == 1:

        talweg_shapefile = params.talweg.filename(tileset=None)
        values = list()

        with fiona.open(talweg_shapefile) as fs:
            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    axis = feature['properties']['AXIS']
                    talweg = np.asarray(feature['geometry']['coordinates'])
                    axis_values = extract_talweg_data(axis, talweg, params)
                    values.append(axis_values)

        dataset = xr.concat(values, 'sample', 'all')

    else:

        dataset = talweg_elevation_mp(params, processes, **kwargs)

    # set_metadata(dataset, 'metrics_planform')
    return dataset

def fit_swath_elevations(
        swath: xr.Dataset,
        axis: int,
        measure: float,
        bounds,
        params: Parameters,
        **kwargs) -> xr.Dataset:

    # calculate swath median elevation

    swath = swath.isel(measure=(swath.measure > measure - 100) &
        (swath.measure <= measure + 100))

    if swath.z.size > 0:

        zmed = np.median(swath.z.values)

    else:

        zmed = np.nan

    # use a longer swath for slope regressions

    swath = swath.isel(measure=(swath.measure > measure - 300) &
        (swath.measure <= measure + 300))

    # Axy = np.column_stack([
    #     swath.x.values,
    #     swath.y.values,
    #     np.ones_like(swath.x.values)
    # ])

    # fit_xy = lstsq(Axy, swath.z.values)
    # slope_xy = np.arctan(np.sqrt(fit_xy[0][0]**2 +  fit_xy[0][1])**2)

    # 1. z = f(axis measure) => valley slope

    Am = np.column_stack([
        swath.measure.values,
        np.ones_like(swath.measure.values)
    ])

    fit_m = lstsq(Am, swath.z.values)
    slope_m, z0_m = fit_m[0]

    # representative swath elevation z0
    z0 = slope_m * measure + z0_m

    # 2. z = f(talweg measure) => talweg slope

    As = np.column_stack([
        swath.talweg_measure.values,
        np.ones_like(swath.talweg_measure.values)
    ])

    fit_s = lstsq(As, swath.z.values)
    slope_s, z0_s = fit_s[0]

    # et ensuite ...
    # extraire l'altitude du MNT sur l'emprise de la swath
    # calculer h = z - fit*m
    # calculer la médiane de h dans le fdv vrai (pente < seuil)
    # calculer la distribution cumulée surface = f(h)
    # calculer l'histogramme par tranche de 1 m de hauteur tous les 0.5 m par ex
    # chercher le mode => hauteur moyenne du plan de la vallée
    # profil talweg z(m), profil du fdv z(m) + h

    dem_raster = params.dem.filename(**kwargs)
    # nearest_raster = params.nearest.filename(**kwargs)
    measure_raster = params.measure.filename(**kwargs)
    valley_bottom_raster = params.valley_bottom.filename(**kwargs)

    with rio.open(dem_raster) as ds:

        window = as_window(bounds, ds.transform)
        dem = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)
        mask = (dem != ds.nodata)

    with rio.open(measure_raster) as ds:

        window = as_window(bounds, ds.transform)
        measures = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)
        mask = (
            mask & (measures != ds.nodata) &
            (measures > measure - 100) &
            (measures <= measure + 100)
        )

    with rio.open(valley_bottom_raster) as ds:

        window = as_window(bounds, ds.transform)
        valley_bottom = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

        mask = mask & (valley_bottom == MASK_VALLEY_BOTTOM)

    # with rio.open(nearest_raster) as ds:

    #     window = as_window(bounds, ds.transform)
    #     nearest_axes = ds.read(
    #         1,
    #         window=window,
    #         boundless=True,
    #         fill_value=ds.nodata)
    #     mask = mask & (nearest_axes == axis)

    if np.any(mask):

        height = dem - (slope_m * measures + z0_m)
        # height[~mask] = -99999.0
        height_median = np.median(height[mask])

    else:

        height_median = np.nan

    return xr.Dataset(
        {
            'elevation_talweg': ('swath', np.array([z0], dtype='float32')),
            'elevation_talweg_med': ('swath', np.array([zmed], dtype='float32')),
            'height_valley_bottom': ('swath', np.array([height_median], dtype='float32')),
            'slope_talweg': ('swath', np.array([-slope_s], dtype='float32')),
            'slope_valley_bottom': ('swath', np.array([slope_m], dtype='float32'))
        }, coords={
            'axis': ('swath', np.array([axis], dtype='uint32')),
            'measure': ('swath', np.array([measure], dtype='float32'))
        })

    # height density curve => height mode ?
    # height_min = np.floor(np.min(height[mask]) / 0.5) * 0.5
    # height_max = np.ceil(np.max(height[mask]) / 0.5) * 0.5
    
    # breaks = np.arange(height_min, height_max + 0.2, 0.2)
    # height_dig = np.digitize(height, breaks)
    # height_dig[~mask] = 0

    # # area = f(h)
    # area_height_curve = np.cumsum(
    #     np.array([
    #         np.sum(height_dig == k+1)
    #         for k in range(len(breaks)-1)
    #     ])
    # )

    # area_height_fun = interp1d(breaks[:-1], area_height_curve, bounds_error=False, fill_value=0.0)

    # hx = np.arange(height_min + 0.5, height_max, 0.5)
    # height_density = np.array([
    #     area_height_fun(h + 0.5) - area_height_fun(h - 0.5)
    #     for h in hx
    # ])

    # height_mod = hx[np.argmax(height_density)]
    # height_density = np.column_stack([hx, height_density])

    # print(fit_m)
    # print(fit_s)
    # print(height_median)
    # print(height_mod)

    # return height_density

def fit_axis(data, axis, swath_bounds, params: Parameters):

    values = list()

    for ax, measure in swath_bounds:

        if ax != axis:
            continue

        bounds = swath_bounds[ax, measure]
        value = fit_swath_elevations(
            data,
            axis,
            measure,
            bounds,
            params)

        values.append(value)

    return xr.concat(values, 'swath', 'all')

def fit_mp(data, swath_bounds, params: Parameters, processes: int = 6, **kwargs) -> xr.Dataset:

    def length():

        return len(np.unique(data.axis))

    def arguments():

        for axis in np.unique(data.axis):

            # bounds = swath_bounds[axis, measure]
            swath = (
                data.set_index(sample=('axis', 'measure'))
                .sel(axis=axis)
            )

            yield (
                fit_axis,
                swath,
                axis,
                swath_bounds,
                params,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            values = list(iterator)

    return xr.concat(values, 'swath', 'all')

def TalwegElevationProfile(data, swath_bounds, params: Parameters, processes: int = 6, **kwargs) -> xr.Dataset:

    length = len(np.unique(data.axis))

    if length == 1 or processes == 1:

        logger = logging.getLogger(__name__)

        def values():

            with click.progressbar(swath_bounds) as iterator:
                for (axis, measure) in iterator:

                    bounds = swath_bounds[axis, measure]
                    swath = (
                        data.set_index(sample=('axis', 'measure'))
                        .sel(axis=axis)
                        .load()
                    )

                    try:

                        yield fit_swath_elevations(
                            swath,
                            axis,
                            measure,
                            bounds,
                            params,
                            **kwargs)

                    except ValueError:

                        logger.error('Error on swath (%d, %.1f)', axis, measure)
                        continue

        dataset = xr.concat(values(), 'swath', 'all')

    else:

        dataset = fit_mp(data, swath_bounds, params, processes, **kwargs)

    dataset['sinuosity'] = dataset.slope_valley_bottom / dataset.slope_talweg
    dataset.sinuosity[dataset.sinuosity < 1.0] = 1.0
    dataset.sinuosity[dataset.slope_talweg < 0.00005] = 1.0

    return dataset

# height = profile.height_valley_median.rolling(measure=3, min_periods=1).median()
# sinuo = (profile.slope_valley / profile.slope_talweg).rolling(measure=3, min_periods=1).mean()
# sinuo[sinuo < 1.0] = 1.0

def AdjustedProfile(profile, fitted):

    talweg = profile.set_index(sample=('axis', 'talweg_measure'))
    swaths = fitted.set_index(swath=('axis', 'measure'))

    def interpolate(axis):

        points_ax = talweg.sel(axis=axis)
        elevations_ax = swaths.sel(axis=axis).sortby('measure')

        # elevation_fun = UnivariateSpline(
        #     elevations_ax.measure.values,
        #     elevations_ax.elevation_talweg.values,
        #     k=3)

        elevation_fun = interp1d(
            elevations_ax.measure.values,
            elevations_ax.elevation_talweg.values,
            bounds_error=False,
            fill_value=np.nan)

        elevations = elevation_fun(points_ax.measure.values)
        missing = np.isnan(elevations)

        if np.any(missing):
            elevations[missing] = points_ax.z[missing]

        return xr.Dataset(
            {
                'elevation_mod': ('sample', np.float32(elevations))
            }, coords={
                'axis': ('sample', np.full_like(points_ax.talweg_measure.values, axis, dtype='uint32')),
                'talweg_measure': ('sample', points_ax.talweg_measure.values)
            })

    # elevation_mod.rolling(talweg_measure=9, min_periods=1).mean()

    dataset = xr.concat([
        interpolate(axis)
        for axis in np.unique(talweg.axis)
    ], 'sample', 'all')

    return dataset
        