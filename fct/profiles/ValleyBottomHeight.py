"""
Valley bottom height above talweg/drainage
"""

from multiprocessing import Pool
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

import click
import fiona
import rasterio as rio
from shapely.geometry import LineString, asShape
import xarray as xr

from ..cli import starcall
from ..tileio import as_window
from ..config import DatasetParameter
from ..corridor.ValleyBottomFeatures import MASK_VALLEY_BOTTOM
from ..metadata import set_metadata
from .. import speedup

class Parameters:
    """
    Planform metrics parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')

    dem = DatasetParameter(
        'elevation raster (DEM)',
        type='input')

    nearest = DatasetParameter(
        'nearest drainage axis (raster)',
        type='input')

    measure = DatasetParameter(
        'location along nearest reference axis (raster)',
        type='input')

    talweg = DatasetParameter('', type='input')

    valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')

    samples = DatasetParameter(
        'spatial samples raster',
        type='input')

    output = DatasetParameter(
        'destination netcdf dataset',
        type='output')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.dem = 'dem'

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.nearest = 'nearest_drainage_axis'
            self.measure = 'axis_measure'
            self.talweg = dict(key='metrics_talweg_points', tiled=False, subdir='NETWORK/METRICS')
            self.valley_bottom = 'valley_bottom_final'
            self.samples = 'poisson_samples'
            self.output = 'metrics_planform'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.nearest = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            self.talweg = dict(key='metrics_talweg_points', tiled=False, axis=axis)
            self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
            self.samples = dict(key='ax_poisson_samples', axis=axis)
            self.output = dict(key='metrics_planform', axis=axis)

def SwathValleyBottomHeight(
        talweg: xr.Dataset,
        axis: int,
        measure: float,
        bounds,
        params: Parameters,
        **kwargs):

    with rio.open(params.dem.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        dem = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

        # profile = ds.profile.copy()
        # nodata = ds.nodata

    with rio.open(params.measure.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        measures = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

        mask = (
            (measures > measure - 100) &
            (measures <= measure + 100)
        )

    if np.sum(mask) > 1000:

        with rio.open(params.samples.filename(**kwargs)) as ds:

            window = as_window(bounds, ds.transform)
            samples = ds.read(
                1,
                window=window,
                boundless=True,
                fill_value=ds.nodata)

            mask = mask & (samples == 1)
            del samples

    with rio.open(params.valley_bottom.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        valley_bottom = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

    # with rio.open(params.continuity.filename(**kwargs)) as ds:

    #     window = as_window(bounds, ds.transform)
    #     continuity = ds.read(
    #         1,
    #         window=window,
    #         boundless=True,
    #         fill_value=ds.nodata)

    mask1 = mask & (valley_bottom == MASK_VALLEY_BOTTOM)
    # height = np.full_like(dem, nodata)

    if np.any(mask1):

        size = np.sum(mask1)
        z = dem[mask1]
        m = measures[mask1]
        # x = np.column_stack([m, z])
        # scaler = StandardScaler().fit(x)
        # xs = scaler.transform(x)

        try:

            regressor = HuberRegressor()
            # regressor.fit(xs[:, 0].reshape(-1, 1), xs[:, 1])
            regressor.fit(m.reshape(size, 1), z.reshape(size))

            # height[mask] = dem[mask] - regressor.predict(measures[mask])

            talweg = talweg.isel(
                measure=(
                    (talweg.measure > measure - 300) &
                    (talweg.measure <= measure + 300)
                ))

            # y = np.column_stack([
            #     talweg.measure.values,
            #     talweg.z.values
            # ])

            # ys = scaler.transform(y)

            predicted = regressor.predict(talweg.measure.values.reshape(-1, 1))
            height = np.median(talweg.z - predicted)
            zfp = regressor.predict(np.array([measure]).reshape(-1, 1))

        except ConvergenceWarning:

            height = np.nan
            zfp = np.array([np.nan])

        except ValueError:

            height = np.nan
            zfp = np.array([np.nan])

        # regressor2 = HuberRegressor()
        # regressor2.fit(talweg.measure.values.reshape(-1, 1), talweg.z.values)

    else:

        height = 0.0
        zfp = np.array([np.nan])

    return xr.Dataset(
        {
            'height_talweg': (('swath',), np.array([height], dtype='float32')),
            'elevation_valley_bottom': (('swath',), np.float32(zfp)),
        }, coords={
            'axis': (('swath',), np.array([axis], dtype='uint32')),
            'measure': (('swath',), np.array([measure], dtype='float32'))
        })

def axis_loop(talweg, axis, swath_bounds, params: Parameters):

    def values():

        for ax, measure in swath_bounds:

            if ax != axis:
                continue

            bounds = swath_bounds[ax, measure]

            yield SwathValleyBottomHeight(
                talweg,
                ax,
                measure,
                bounds,
                params)

    return xr.concat(values(), 'swath', 'all')

def ValleyBottomHeight(swath_bounds, params: Parameters, processes: int = 1, **kwargs):

    data = (
        xr.open_dataset(params.talweg.filename())
        .set_index(sample=('axis', 'measure'))
    )

    length = len(np.unique(data.axis))

    if length == 1 or processes == 1:

        def values():

            with click.progressbar(swath_bounds) as iterator:
                for (axis, measure) in iterator:

                    talweg = data.sel(axis=axis).load()
                    bounds = swath_bounds[axis, measure]

                    yield SwathValleyBottomHeight(
                        talweg,
                        axis,
                        measure,
                        bounds,
                        params)

        return xr.concat(values(), 'swath', 'all')

    def arguments():

        for axis in np.unique(data.axis):

            # bounds = swath_bounds[axis, measure]
            talweg = data.sel(axis=axis).load()

            yield (
                axis_loop,
                talweg,
                axis,
                swath_bounds,
                params,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length) as iterator:
            values = list(iterator)

    return xr.concat(values, 'swath', 'all')
