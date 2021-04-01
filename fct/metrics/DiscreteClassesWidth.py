"""
Valley bottom width metrics
"""

from multiprocessing import Pool
import numpy as np
from scipy.stats import linregress

import click
import xarray as xr

# from ..config import config
from ..cli import starcall
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

    # data = (
    #     xr.open_dataset(source.filename())
    #     .set_index(sample=('axis', 'measure', 'distance'))
    #     .sel(axis=axis, measure=measure)
    #     .load()
    # )

    pixarea = params.resolution**2
    swath_length = params.swath_length
    distance_delta = params.distance_delta
    nlabels = len(data.label)

    density = data.profile.sum(axis=(1, 2))
    reg = linregress(data.distance[density > 0], density[density > 0])
    density_ref = reg.slope * data.distance + reg.intercept

    area = data.profile.sum(axis=0) * pixarea
    width1 = area / swath_length
    width2 = np.sum(data.profile / density_ref, axis=0) * distance_delta

    return xr.Dataset({
        'area': (('swath', 'label', 'side'), np.float32(area.values.reshape(1, nlabels, 2))),
        'width1': (('swath', 'label', 'side'), np.float32(width1.values.reshape(1, nlabels, 2))),
        'width2': (('swath', 'label', 'side'), np.float32(width2.values.reshape(1, nlabels, 2))),
    }, coords={
        'axis': (('swath',), np.uint32([axis])),
        'measure': (('swath',), np.float32([measure])),
        'label': data.label.values,
        'side': ['left', 'right']
    })

def axis_swath_width(data, axis, params: Parameters):

    values = list()

    for measure in np.unique(data.measure):

        swath_data = data.sel(measure=measure)
        value = swath_width(swath_data, axis, measure, params)
        values.append(value)

    return xr.concat(values, 'swath', 'all')

def DiscreteClassesWidth(dataset: xr.Dataset, params: Parameters, processes: int = 6, **kwargs):
    """
    Aggregate elevation swath profiles into valley bottom width metrics
    """

    if processes == 1:

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

    # dataset = (
    #     xr.open_dataset(source.filename())
    #     .set_index(sample=('axis', 'measure', 'distance'))
    # )

    def length():

        return sum(
            1 for ax in np.unique(dataset.axis)
        )

    def arguments():

        for ax in np.unique(dataset.axis):

            axis_data = dataset.sel(axis=ax)

            yield (
                axis_swath_width,
                axis_data,
                ax,
                params,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            values = list(iterator)

    result = xr.concat(values, 'swath', 'all')

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

    
