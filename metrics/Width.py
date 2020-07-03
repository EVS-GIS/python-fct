import os
import math
import itertools
from operator import itemgetter
from collections import defaultdict

import numpy as np

import xarray as xr
import click

import rasterio as rio
from rasterio.windows import Window
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

import terrain_analysis as ta

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import EngFormatter
from Plotting import MapFigureSizer

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset
    width = col_end - col_offset

    return Window(col_offset, row_offset, width, height)

def FluvialCorridorWidth(axis):
    """
    Defines
    -------

    fcw: fluvial corridor width (meter)

        fcw2: measured at +2.0 m above valley floor
        fcw8: measured at +8.0 m above nearest drainage
        fcw10: measured at +10.0 m above nearest drainage

    bankh: estimated bank height (meter) above water channel

        bankh1: opposite of minimum of swath elevation above valley floor
        bankh2: opposite of median of swath elevation above valley floor
                for swath pixels below -bankh1 + 1.0 m,
                or bankh1 if no such pixels
    """

    dgo_shapefile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'REF', 'DGO.shp')
    # dgo_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'DGO.vrt')
    # accumulation_raster = '/var/local/fct/RMC/ACC_RGE5M_TILES.vrt'
    # output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_DRAINAGE_AREA.csv')
    # metrics = dict()

    gids = list()
    measures = list()
    values = list()

    with fiona.open(dgo_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                swathfile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'ELEVATION', 'SWATH_%04d.npz' % gid)
                data = np.load(swathfile, allow_pickle=True)

                x = data['x']
                hand = data['hand']
                hvf = data['hvf']

                try:
                    density = data['density']
                    density_max = np.max(density)
                except ValueError:
                    density = np.zeros(0, dtype='uint32')
                    density_max = 0

                if density_max == 0 or x.shape[0] < 3:

                    gids.append(gid)
                    measures.append(measure)
                    values.append((np.nan, np.nan, np.nan, np.nan, np.nan))
                    continue

                # unit width of observations
                w = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                w[0] = x[1] - x[0]
                w[-1] = x[-1] - x[-2]

                if hvf.size > 0:

                    selection = (hvf[:, 2] <= 2.0)
                    if selection.size > 0:
                        fcw2 = np.sum(w[selection] * density[selection]) / density_max
                    else:
                        fcw2 = np.nan

                    mask = np.isnan(hvf[:, 2])
                    bankh1 = np.ma.min(np.ma.masked_array(hvf[:, 2], mask))
                    if bankh1 is np.ma.masked:

                        bankh1 = np.nan
                        bankh2 = np.nan

                    else:

                        bankh1 = -bankh1

                        mask = (hvf[:, 2] >= min(-0.5, -bankh1 + 1.0))
                        bankh2 = np.ma.median(np.ma.masked_array(hvf[:, 2], mask))

                        if bankh2 is np.ma.masked:
                            bankh2 = bankh1
                        else:
                            bankh2 = -bankh2

                else:

                    fcw2 = np.nan
                    bankh1 = np.nan
                    bankh2 = np.nan

                selection = (hand[:, 2] <= 8.0)
                if selection.size > 0:
                    fcw8 = np.sum(w[selection] * density[selection]) / density_max
                else:
                    fcw8 = np.nan

                selection = (hand[:, 2] <= 10.0)
                if selection.size > 0:
                    fcw10 = np.sum(w[selection] * density[selection]) / density_max
                else:
                    fcw10 = np.nan

                # values.append((gid, measure, fcw2, fcw8, fcw10, bankh1, bankh2))

                gids.append(gid)
                measures.append(measure)
                values.append((fcw2, fcw8, fcw10, bankh1, bankh2))

    # dtype = np.dtype([
    #     ('gid', 'int'),
    #     ('measure', 'float32'),
    #     ('fcw2', 'float32'),
    #     ('fcw8', 'float32'),
    #     ('fcw10', 'float32'),
    #     ('bankh1', 'float32'),
    #     ('bankh2', 'float32')
    # ])

    # return np.sort(np.array(values, dtype=dtype), order='measure')

    gids = np.array(gids, dtype='uint32')
    measures = np.array(measures, dtype='float32')
    data = np.array(values, dtype='float32')

    return xr.Dataset(
        {
            'measure': ('swath', measures),
            'fcw2': ('swath', data[:, 0]),
            'fcw8': ('swath', data[:, 1]),
            'fcw10': ('swath', data[:, 2]),
            'bankh1': ('swath', data[:, 3]),
            'bankh2': ('swath', data[:, 4])
        },
        coords={
            'axis': axis,
            'swath': gids,
        })

def WriteFluvialCorridorWidth(axis, data):

    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'FLUVIAL_CORRIDOR_WIDTH.nc')

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'fcw2': dict(zlib=True, complevel=9, least_significant_digit=1),
            'fcw8': dict(zlib=True, complevel=9, least_significant_digit=1),
            'fcw10': dict(zlib=True, complevel=9, least_significant_digit=1),
            'bankh1': dict(zlib=True, complevel=9, least_significant_digit=0),
            'bankh2': dict(zlib=True, complevel=9, least_significant_digit=0)
        })

def ContinuityWidth(axis):
    """
    Defines
    -------

    lcck: continuity width (meter) for land cover class k
    """

    dgo_shapefile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'REF', 'DGO.shp')

    gids = list()
    measures = list()
    values = list()
    with fiona.open(dgo_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                swathfile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'CONTINUITY', 'SWATH_CONTINUITY_%04d.npz' % gid)
                data = np.load(swathfile, allow_pickle=True)

                x = data['x']
                classes = data['classes']
                swath = data['swath']

                try:
                    density = data['density']
                    density_max = np.max(density)
                except ValueError:
                    density = np.zeros(0, dtype='uint32')
                    density_max = 0

                if density_max == 0 or x.shape[0] < 3:

                    gids.append(gid)
                    measures.append(measure)
                    width = np.zeros((9, 2), dtype='float32')
                    values.append(width)
                    continue

                # count = np.ma.sum(np.ma.masked_array(swath, np.isnan(swath)), axis=1)
                dominant = np.ma.argmax(np.ma.masked_array(swath, np.isnan(swath)), axis=1)

                # unit width of observations
                unit_width = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                unit_width[0] = x[1] - x[0]
                unit_width[-1] = x[-1] - x[-2]

                # width = np.zeros(9, dtype='float32')
                width = np.zeros((9, 2), dtype='float32')

                for k in range(len(classes)):

                    if classes[k] == 255:
                        continue

                    # selection = (swath[:, k] / count) > threshold
                    # if selection.size > 0:
                    #     width[classes[k]] = np.sum(unit_width[selection] * density[selection]) / density_max
                    # else:
                    #     width[classes[k]] = 0

                    selection = (dominant == k)
                    if selection.size > 0:
                        width[classes[k]] = np.sum(unit_width[selection] * density[selection]) / density_max
                    else:
                        width[classes[k]] = 0

                    selection = (dominant == k) & (x >= 0)
                    if selection.size > 0:
                        width[classes[k], 0] = np.sum(unit_width[selection] * density[selection]) / density_max
                    else:
                        width[classes[k], 0] = 0

                    selection = (dominant == k) & (x < 0)
                    if selection.size > 0:
                        width[classes[k], 1] = np.sum(unit_width[selection] * density[selection]) / density_max
                    else:
                        width[classes[k], 1] = 0

                # values.append(tuple([gid, measure] + width.tolist()))

                gids.append(gid)
                measures.append(measure)
                values.append(width)

    # dtype = [
    #     ('gid', 'int'),
    #     ('measure', 'float32')
    # ] + [('lcc%d' % k, 'float32') for k in range(9)]

    # return np.sort(np.array(values, dtype=np.dtype(dtype)), order='measure')
    gids = np.array(gids, dtype='uint32')
    measures = np.array(measures, dtype='float32')
    data = np.array(values, dtype='float32')

    return xr.Dataset(
        {
            'measure': (('swath'), measures),
            'lcc': (('swath', 'landcover', 'side'), data)
        },
        coords={
            'axis': axis,
            'swath': gids,
            'landcover': [
                'Water Channel',
                'Gravel Bars',
                'Natural Open',
                'Forest',
                'Grassland',
                'Crops',
                'Diffuse Urban',
                'Dense Urban',
                'Infrastructures'
            ],
            'side': [
                'left',
                'right'
            ]
        })

def WriteContinuityWidth(axis, data):

    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'LANDCOVER_WIDTH.nc')

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'lcc': dict(zlib=True, complevel=9, least_significant_digit=2)
        })

# def PlotMetric(data, fieldx, fieldy, window=1, title='', filename=None):

#     fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
#     gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
#     ax = fig.add_subplot(gs[25:100,10:95])

#     ax.spines['top'].set_linewidth(1)
#     ax.spines['left'].set_linewidth(1)
#     ax.spines['right'].set_linewidth(1)
#     ax.spines['bottom'].set_linewidth(1)
#     ax.set_ylabel(fieldy)
#     ax.set_xlabel(fieldx)
#     formatter = EngFormatter(unit='m')
#     ax.xaxis.set_major_formatter(formatter)
#     ax.tick_params(axis='both', width=1, pad = 2)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.set_pad(2)
#     ax.grid(which='both', axis='both', alpha=0.5)

#     if fieldx == 'measure':
#         ax.set_xlim([np.max(data[fieldx]), np.min(data[fieldx])])
    
#     x = data[fieldx]
#     y = data[fieldy]

#     if window > 1 and fieldx == 'measure':
#         y = y.rolling(swath=window, min_periods=1, center=True).mean()

#     ax.plot(x, y, "#48638a", linewidth=1)

#     fig_size_inches = 12.5
#     aspect_ratio = 4
#     cbar_L = "None"
#     [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

#     plt.title(title)
#     fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
#     ax.set_position(map_axes)

#     if filename is None:
#         fig.show()
#     elif filename.endswith('.pdf'):
#         plt.savefig(filename, format='pdf', dpi=600)
#         plt.clf()
#     else:
#         plt.savefig(filename, format='png', dpi=300)
#         plt.clf()

def PlotMetric(data, fieldx, *args, window=1, title='', filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.set_ylabel('Width (m)')
    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    ax.grid(which='both', axis='both', alpha=0.5)

    x = data[fieldx]

    if fieldx == 'measure':
        ax.set_xlabel('Location along reference axis (from network outlet)')
        ax.set_xlim([np.max(x), np.min(x)])
    else:
        ax.set_xlabel(fieldx)

    colors = [
        "#48638a",
        "darkgreen",
        "darkred"
    ]

    for k, fieldy in enumerate(args):

        y = data[fieldy]

        if window > 1 and fieldx == 'measure':
            y = y.rolling(swath=window, min_periods=1, center=True).mean()

        ax.plot(x, y, colors[k], linewidth=1, label=fieldy)

    if len(args) > 1:
        ax.legend()

    fig_size_inches = 12.5
    aspect_ratio = 4
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()

def PlotContinuityProfile(data, title='', window=1, proportion=False, direction='upright', filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    if proportion:
        ax.set_ylabel("Cover Class Proportion")
    else:
        ax.set_ylabel("Cover Class Width (m)")
    ax.set_xlabel("Location along reference axis (from network outlet)")
    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    ax.grid(which='both', axis='both', alpha=0.5)

    colors = [
        '#a5bfdd', # Water
        '#cccccc', # Gravels
        '#bee62e', # Natural
        '#6f9e00', # Forest
        '#ffe45a', # Grassland
        '#ffff99', # Crops
        '#fa7c85', # Diffuse Urban
        '#fa1524', # Urban
        '#fa1665'  # Disconnected
    ]

    x = data['measure']
    fcw = data['fcw8']
    lcc = data['lcc'][:, :, 0] + data['lcc'][:, :, 1]

    if window > 1:
        fcw = fcw.rolling(swath=window, min_periods=1, center=True).mean()
        lcc = lcc.rolling(swath=window, min_periods=1, center=True).mean()

    # reverse measure direction
    ax.set_xlim([np.max(x), np.min(x)])
    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)

    # Do not plot zeros

    parts = np.split(
        np.column_stack([
            x,
            fcw,
            lcc]),
        np.where(np.isnan(fcw))[0])

    for k, part in enumerate(parts):

        if k == 0:

            xk = part[:, 0]
            fcwk = part[:, 1]
            lcck = part[:, 2:]

        else:

            xk = part[1:, 0]
            fcwk = part[1:, 1]
            lcck = part[1:, 2:]

        if proportion:

            baseline = np.sum(lcck[:, :2], axis=1)
            baseline[baseline > fcwk] = fcwk[baseline > fcwk]
            fcwk = fcwk - baseline
            lcck = lcck / fcwk[:, np.newaxis]
            baseline = np.zeros_like(fcwk)
            fcwk = np.ones_like(fcwk)

        else:

            baseline = np.sum(lcck[:, :2], axis=1)
            baseline[baseline > fcwk] = fcwk[baseline > fcwk]

        cumulative = np.copy(baseline)
        lagged = np.copy(cumulative)

        if xk.size > 0:

            variables = range(2, lcck.shape[1])
            variables = reversed(variables) if direction == 'updown' else variables

            for variable in variables:

                cumulative += lcck[:, variable]
                cumulative[cumulative > fcwk] = fcwk[cumulative > fcwk]

                # runs = np.split(
                #     np.column_stack([xk, cumulative, lagged]),
                #     np.where(lcck[:, variable] < 1.0)[0])

                # for i, run in enumerate(runs):

                #     if i == 0:

                #         xki = run[:, 0]
                #         cumi = run[:, 1]
                #         lagi = run[:, 2]

                #     else:

                #         xki = run[1:, 0]
                #         cumi = run[1:, 1]
                #         lagi = run[1:, 2]

                #     ax.fill_between(xki, lagi, cumi, facecolor=colors[variable], alpha = 0.7, interpolate=True)
                #     ax.plot(xki, cumi, colors[variable], linewidth = 1.0)

                ax.fill_between(xk, lagged - baseline, cumulative - baseline, facecolor=colors[variable], alpha = 0.7, interpolate=True)
                if variable < lcck.shape[1]-2:
                    ax.plot(xk, cumulative - baseline, colors[variable], linewidth = 0.8)

                lagged += lcck[:, variable]
                lagged[lagged > fcwk] = fcwk[lagged > fcwk]

    if not proportion:
        ax.plot(x, fcwk - baseline, 'darkgray', linewidth = 1.0)

    fig_size_inches = 12.5
    aspect_ratio = 4
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()

def PlotLeftRightContinuityProfile(data, title='', window=1, proportion=False, direction='upright', filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    if proportion:
        ax.set_ylabel("Cover Class Proportion")
    else:
        ax.set_ylabel("Cover Class Width (m)")
    ax.set_xlabel("Location along reference axis (from network outlet)")
    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    ax.grid(which='both', axis='both', alpha=0.5)

    colors = [
        '#a5bfdd', # Water
        '#cccccc', # Gravels
        '#bee62e', # Natural
        '#6f9e00', # Forest
        '#ffe45a', # Grassland
        '#ffff99', # Crops
        '#fa7c85', # Diffuse Urban
        '#fa1524', # Urban
        '#fa1665'  # Disconnected
    ]

    x = data['measure']
    fcw = data['fcw8']
    # lcc = data['lcc']
    left = data['lcc'][:, :, 0]
    right = data['lcc'][:, :, 1]

    print(fcw.shape, left.shape, right.shape)

    if window > 1:
        fcw = fcw.rolling(swath=window, min_periods=1, center=True).mean()
        # lcc = lcc.rolling(swath=window, min_periods=1, center=True).mean()
        left = left.rolling(swath=window, min_periods=1, center=True).mean()
        right = right.rolling(swath=window, min_periods=1, center=True).mean()

    # reverse measure direction
    ax.set_xlim([np.max(x), np.min(x)])
    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)

    # Do not plot zeros

    parts = np.split(
        np.column_stack([
            x,
            fcw,
            left,
            right]),
        np.where(np.isnan(fcw))[0])

    for k, part in enumerate(parts):

        print(part.shape)

        if k == 0:

            xk = part[:, 0]
            fcwk = part[:, 1]
            leftk = part[:, 2:11]
            rightk = part[:, 11:]

        else:

            xk = part[1:, 0]
            fcwk = part[1:, 1]
            leftk = part[1:, 2:11]
            rightk = part[1:, 11:]

        print(leftk.shape, rightk.shape)

        lcck = np.zeros(leftk.shape + (2,))
        lcck[:, :, 0] = leftk
        lcck[:, :, 1] = rightk

        baseline = np.sum(lcck[:, :2, :], axis=1)
        print(lcck.shape, baseline.shape)
        baseline[baseline[:, 0] > fcwk, 0] = fcwk[baseline[:, 0] > fcwk]
        baseline[baseline[:, 1] > fcwk, 1] = fcwk[baseline[:, 1] > fcwk]

        if proportion:

            fcwk = fcwk - np.sum(baseline, axis=1)
            lcck = lcck / fcwk[:, np.newaxis, np.newaxis]
            baseline = np.zeros(fcwk.shape + (2,))
            fcwk = np.ones_like(fcwk)

        cumulative = np.copy(baseline)
        lagged = np.copy(cumulative)

        if xk.size > 0:

            variables = range(2, lcck.shape[1])
            variables = reversed(variables) if direction == 'updown' else variables

            for variable in variables:

                cumulative += lcck[:, variable, :]

                for side in (0, 1):

                    cumulative[cumulative[:, side] > fcwk, side] = fcwk[cumulative[:, side] > fcwk]

                    sign = 1 if side == 0 else -1
                    ax.fill_between(
                        xk,
                        sign * (lagged[:, side] - baseline[:, side]),
                        sign*(cumulative[:, side] - baseline[:, side]),
                        facecolor=colors[variable],
                        alpha=0.7,
                        interpolate=True)
                    if variable < lcck.shape[1]-2:
                        ax.plot(
                            xk,
                            sign*(cumulative[:, side] - baseline[:, side]),
                            colors[variable],
                            linewidth=0.8)

                    lagged[:, side] += lcck[:, variable, side]
                    lagged[lagged[:, side] > fcwk, side] = fcwk[lagged[:, side] > fcwk]

    fig_size_inches = 12.5
    aspect_ratio = 4
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()

def test(axis=1044):

    mpl.use('cairo')

    fcw = FluvialCorridorWidth(axis)
    lcc = ContinuityWidth(axis)

    WriteFluvialCorridorWidth(axis, fcw)
    WriteContinuityWidth(axis, lcc)

    data = fcw.merge(lcc).sortby(fcw['measure'])
    print(data)

    PlotMetric(
        data,
        'measure',
        'fcw2',
        window=5,
        title='Corridor Width (FCW2)',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW2.pdf'))

    PlotMetric(
        data,
        'measure',
        'fcw8',
        window=5,
        title='Corridor Width (FCW8)',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW8.pdf'))

    PlotMetric(
        data,
        'measure',
        'fcw10',
        window=5,
        title='Corridor Width (FCW10)',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW10.pdf'))

    PlotMetric(
        data,
        'measure',
        'fcw2',
        'fcw8',
        'fcw10',
        window=5,
        title='Fluvial Corridor Width Metrics',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW_METRICS.pdf'))

    PlotContinuityProfile(
        data,
        window=5,
        proportion=False,
        title='Corridor Width Profile',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'CORRIDOR_PROFILE.pdf'))

    PlotContinuityProfile(
        data,
        window=5,
        proportion=True,
        title='Corridor Width Profile',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'CORRIDOR_PROFILE_PROP.pdf'))

    PlotLeftRightContinuityProfile(
        data,
        window=5,
        proportion=False,
        title='Corridor Width Profile',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'CORRIDOR_PROFILE_LEFTRIGHT.pdf'))

    return data
