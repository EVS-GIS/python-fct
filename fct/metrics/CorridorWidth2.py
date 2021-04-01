from matplotlib import pyplot as plt
import xarray as xr

def CorridorWidth(width_continuity: xr.Dataset, length_talweg: xr.Dataset) -> xr.Dataset:
    """
    Calculate summary corridor widths from continuity widths:

    - active channel width (ACW)
    - natural corridor width (NCW)
    - connected corridor width (CCW)
    """

    # width = xr.open_dataset(params.width_continuity.filename(tileset=None))
    # length_talweg = xr.open_dataset(params.length_talweg.filename(tileset=None))

    width = width_continuity.set_index(swath=('axis', 'measure'))
    length = length_talweg.set_index(swath=('axis', 'measure'))

    width_sum = (
        width.sum(['label', 'side'])
        .rename(area='area_valley_bottom'))

    wcw = (
        width.sel(label='Water channel')
        .sum('side')
        .drop_vars('label')
        .rename(area='area_wc'))

    acw = (
        width.sel(label=['Water channel', 'Active channel'])
        .sum(('side', 'label'))
        # .drop_vars('label')
        .rename(area='area_ac'))

    ncw = (
        width.sel(label='Riparian')
        .drop_vars('label')
        .rename(width2='width_nc'))

    ccw = (
        width.sel(label=['Riparian', 'Meadows', 'Cultivated'])
        .sum('label')
        .rename(width2='width_cc'))

    data = xr.merge([
        wcw.area_wc,
        acw.area_ac,
        ncw.width_nc,
        ccw.width_cc,
        width_sum,
        length
    ]).load()

    data['width_water_channel'] = data.area_wc / data.length_talweg
    data['width_active_channel'] = data.area_ac / data.length_talweg
    data['width_natural_corridor'] = data.width_nc * data.width1 / data.width2
    data['width_connected_corridor'] = data.width_cc * data.width1 / data.width2

    data = data.drop_vars([
        'area_wc',
        'area_ac',
        'width_nc',
        'width_cc',
        'width1',
        'width2'])

    # data.reset_index('swath').to_netcdf(params.output.filename(tileset=None))

    return data

def plot_profile(data, axis):

    datax = data.sel(axis=axis).groupby('measure').sum('side')

    plt.plot(
        datax.measure[1:],
        datax.width_natural_corridor[1:])
    plt.plot(
        datax.measure[1:],
        datax.width_connected_corridor[1:])
    plt.show()

def plot_profile_left_right(data, axis):

    datax = data.sel(axis=axis).groupby('measure')

    plt.plot(
        datax.measure[1:],
        -datax.sel(side='left').width_natural_corridor[1:],
        datax.measure[1:],
        datax.sel(side='right').width_natural_corridor[1:])
    plt.plot(
        datax.measure[1:],
        -datax.sel(side='left').width_connected_corridor[1:],
        datax.measure[1:],
        datax.sel(side='right').width_connected_corridor[1:])
    plt.show()

def test():

    from workflows.SwathProfiles import init, config
    from fct.metrics.DiscreteClassesWidth import Parameters, DiscreteClassesWidth
    from pathlib import Path

    init()
    workdir = Path(config.workspace.workdir)

    data = (
        xr.open_dataset(workdir / 'V2/NETWORK/METRICS/SWATHS_LANDCOVER.nc')
        .set_index(sample=('axis', 'measure', 'distance'))
        .load()
    )

    params = Parameters()
    width_landcover = DiscreteClassesWidth(data, params)
    width_sum = width_landcover.sum(['label', 'side'])

    width_landcover['width_landcover'] = (
        width_landcover.width2 / width_sum.width2 * width_sum.width1
    )
    
    width_landcover.to_netcdf(workdir / 'V2/NETWORK/METRICS/WIDTH_LANDCOVER.nc')

    data = (
        xr.open_dataset(workdir / 'V2/NETWORK/METRICS/SWATHS_CONTINUITY.nc')
        .set_index(sample=('axis', 'measure', 'distance'))
        .load()
    )

    params = Parameters()
    width_continuity = DiscreteClassesWidth(data, params)
    width_continuity.to_netcdf(workdir / 'V2/NETWORK/METRICS/WIDTH_CONTINUITY.nc')

    # width_continuity = xr.open_dataset(workdir / 'V2/NETWORK/METRICS/WIDTH_CONTINUITY.nc')
    length_talweg = xr.open_dataset(workdir / 'V2/NETWORK/METRICS/LENGTH_TALWEG.nc')

    width_corridor = CorridorWidth(width_continuity, length_talweg)
    width_corridor.reset_index('swath').to_netcdf(workdir / 'V2/NETWORK/METRICS/WIDTH_CORRIDOR.nc')

    return width_corridor
