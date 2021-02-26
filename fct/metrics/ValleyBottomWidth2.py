"""
Valley bottom width metrics
"""

import numpy as np
from scipy.stats import linregress

import click
import xarray as xr

# from ..config import config
from ..config import (
    # DatasetParameter,
    LiteralParameter
)

class Parameters:
    """
    Width calculation parameters
    """

    swath_length = LiteralParameter('')
    distance_delta = LiteralParameter('')
    resolution = LiteralParameter('')

    def __init__(self):
        """
        Default parameter values
        """

        self.swath_length = 200.0
        self.distance_delta = 10.0
        self.resolution = 5.0

def swath_width(data, axis, measure, params: Parameters):
    """
    Aggregate one swath data into width metrics
    """

    pixarea = params.resolution**2
    swath_length = params.swath_length
    distance_delta = params.distance_delta

    density = data.density.sum(axis=1)
    reg = linregress(data.distance[density > 0], density[density > 0])
    density_ref = reg.slope * data.distance + reg.intercept

    area = data.density.sum(axis=0) * pixarea
    width1 = area / swath_length
    width2 = np.sum(data.density / density_ref, axis=0) * distance_delta

    return xr.Dataset({
        'area': (('swath', 'side'), np.float32(area.values.reshape(1, 2))),
        'width1': (('swath', 'side'), np.float32(width1.values.reshape(1, 2))),
        'width2': (('swath', 'side'), np.float32(width2.values.reshape(1, 2))),
    }, coords={
        'axis': (('swath',), np.uint32([axis])),
        'measure': (('swath',), np.float32([measure])),
        'side': ['left', 'right']
    })

def ValleyBottomWidth(dataset: xr.Dataset, params: Parameters):
    """
    Aggregate elevation swath profiles into valley bottom width metrics
    """

    def coordinates():

        for ax in np.unique(dataset.axis):
            for m in np.unique(dataset.sel(axis=ax).measure):
                yield ax, m

    coords = list(coordinates())

    with click.progressbar(coords) as iterator:

        result = xr.concat([
            swath_width(dataset.sel(axis=ax, measure=m), ax, m, params)
            for ax, m in iterator
        ], 'swath', 'all')

    return result

def test():

    from workflows.SwathProfiles import init, config
    from pathlib import Path

    init()
    workdir = Path(config.workspace.workdir)

    data = (
        xr.open_dataset(workdir / 'V2/NETWORK/METRICS/SWATHS_ELEVATION.nc')
        .set_index(sample=('axis', 'measure', 'distance'))
        .load()
    )

    params = Parameters()
    widths = ValleyBottomWidth(data, params)
    widths.to_netcdf(workdir / 'V2/NETWORK/METRICS/WIDTH_VALLEY_BOTTOM.nc')
