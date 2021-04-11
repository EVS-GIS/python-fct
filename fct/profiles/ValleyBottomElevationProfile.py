"""
Valley bottom elevation profile
"""

import numpy as np

import click
import fiona
import fiona.crs
from shapely.geometry import LineString
import xarray as xr

from ..config import DatasetParameter
from .TalwegElevationProfile import (
    Parameters as RefaxisParameters,
    TalwegElevation as RefaxisElevation
)

class Parameters:
    """
    Valley bottom elevation profile parameters
    """

    river_profile = DatasetParameter('river profile metrics (netcdf)', type='input')
    refaxis_points = DatasetParameter('reference axis sample points (netcdf)', type='input')
    output = DatasetParameter('elevation profile linestring (shapefile)', type='output')

    def __init__(self, axis=None):
        """
        Default parameters values
        """

        if axis is None:

            # self.river_profile = dict(key='metrics_river_profile', tiled=False, subdir='NETWORK/METRICS')
            self.river_profile = dict(key='metrics_floodplain_height', tiled=False, subdir='NETWORK/METRICS')
            self.refaxis_points = dict(key='metrics_refaxis_points', tiled=False, subdir='NETWORK/METRICS')
            self.output = dict(key='elevation_profile_valley_bottom', tiled=False, subdir='NETWORK/REF')

        else:

            # self.river_profile = dict(key='metrics_river_profile', tiled=False, axis=axis)
            self.river_profile = dict(key='metrics_floodplain_height', tiled=False, axis=axis)
            self.refaxis_points = dict(key='metrics_refaxis_points', tiled=False, axis=axis)
            self.output = dict(key='elevation_profile_valley_bottom', tiled=False, axis=axis)

def RefaxisSamplePoints(params: Parameters, axis: int = None):

    refaxis_params = RefaxisParameters(axis=axis)
    
    if axis is None:
        refaxis_params.talweg = 'refaxis'
    else:
        refaxis_params.talweg = dict(key='ax_refaxis', axis=axis)
    
    refaxis = RefaxisElevation(refaxis_params)
    refaxis.to_netcdf(params.refaxis_points.filename())

def ValleyBottomElevationProfile(params: Parameters):
    """
    Project reference axis on valley bottom z profile
    """

    data = (
        xr.open_dataset(params.river_profile.filename())
        .set_index(swath=('axis', 'measure'))
        .load()
    )

    refaxis = (
        xr.open_dataset(params.refaxis_points.filename())
        .set_index(sample=('axis', 'talweg_measure'))
        .load()
    )

    def open_output_shapefile(filename):
        """
        create shapefile output coroutine
        """

        driver = 'ESRI Shapefile'
        crs = fiona.crs.from_epsg(2154)
        schema = {
            'geometry': '3D LineString',
            'properties': [
                ('AXIS', 'int')
            ]
        }

        options = dict(driver=driver, crs=crs, schema=schema)

        with fiona.open(filename, 'w', **options) as fst:
            while True:

                axis, geometry = (yield)
                fst.write({
                    'geometry': geometry.__geo_interface__,
                    'properties': {
                        'AXIS': int(axis)
                    }
                })

    sink = open_output_shapefile(params.output.filename())
    next(sink)

    with click.progressbar(np.unique(data.axis)) as iterator:
        for axis in iterator:

            dx = data.sel(axis=axis).sortby('measure')
            dx = dx.isel(measure=(dx.measure > 200.0))

            zvb = dx.elevation_valley_bottom.values

            missing = np.isnan(zvb)
            # zvb[missing] = np.interp(
            #     dx.measure[missing],
            #     dx.measure[~missing],
            #     zvb[~missing])

            zvb = xr.DataArray(
                zvb[~missing],
                dims=('measure',),
                coords={'measure': dx.measure[~missing]})

            zvb_smooth = (
                zvb.sortby('measure')
                .rolling(measure=3, min_periods=1, center=True)
                .median()
                .rolling(measure=5, min_periods=1, center=True)
                .mean())

            refx = refaxis.sel(axis=axis)
            z = np.interp(refx.measure, zvb_smooth.measure, zvb_smooth.values)

            linestringz = LineString(
                np.column_stack([
                    refx.x,
                    refx.y,
                    z
                ]))

            sink.send((axis, linestringz))

    sink.close()
