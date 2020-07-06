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
from Width import FluvialCorridorWidth, ContinuityWidth

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import EngFormatter
from Plotting import MapFigureSizer

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def ReadData():

    axis = 1044
    ago_ew_file = os.path.join(workdir, 'AUX', 'AGO_EW_LIN.shp')
    measure_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'AXIS_MEASURE.vrt')
    values = list()

    with rio.open(measure_raster) as ds:
        with fiona.open(ago_ew_file) as fs:

            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    properties = feature['properties']
                    geometry = feature['geometry']['coordinates']
                    ma, mb = list(ds.sample([geometry[0], geometry[-1]], 1))
                    style = properties['Types']
                    lfdv = properties['m_lrgm_fdv']
                    lba = properties['m_lrgm_ba']

                    values.append((ma, mb, style, lfdv, lba))

    return sorted(values, key=itemgetter(1, 0))

def MkCorridorProfile(data):

    profile = list()

    for ma, mb, style, lfdv, lba in data:

        if ma > 0 and mb > 0:

            profile.append((mb, lfdv))
            profile.append((ma, lfdv))

    return np.array(profile, dtype='float32')

def MkActiveChannelProfile(data):

    profile = list()

    for ma, mb, style, lfdv, lba in data:

        if ma > 0 and mb > 0:

            profile.append((mb, lba))
            profile.append((ma, lba))

    return np.array(profile, dtype='float32')

def PlotProfile(ax, profile):

    x = profile[:, 0]
    y = profile[:, 1]
    ax.set_xlim([np.max(x), np.min(x)])
    ax.plot(x, y, "black", linewidth=1.2, label='m_lrgm_fdv')

def PlotRiverStyle(ax, data):

    cmap = plt.cm.get_cmap('Pastel1', 6)
    coloridx = itertools.count(0)
    colors = defaultdict(lambda: cmap(next(coloridx)))

    def remap(style):

        if style in ('B2', 'B3', 'B4', 'B6', 'B11'):
            return str(style)

        return 'Other'

    for ma, mb, style, lfdv, lba in data:
        style = remap(style)
        ax.axvspan(ma, mb, linewidth=0.2, alpha=0.8, color=colors[style])


def PlotContinuityProfile(ax, data, window=1, basis=2, maxclass=None, proportion=False, direction='upright'):

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
    # ax.set_xlim([np.max(x), np.min(x)])
    # formatter = EngFormatter(unit='m')
    # ax.xaxis.set_major_formatter(formatter)

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

            baseline = np.sum(lcck[:, :basis], axis=1)
            baseline[baseline > fcwk] = fcwk[baseline > fcwk]
            fcwk = fcwk - baseline
            lcck = lcck / fcwk[:, np.newaxis]
            baseline = np.zeros_like(fcwk)
            fcwk = np.ones_like(fcwk)

        else:

            baseline = np.sum(lcck[:, :basis], axis=1)
            baseline[baseline > fcwk] = fcwk[baseline > fcwk]

        cumulative = np.copy(baseline)
        lagged = np.copy(cumulative)

        if xk.size > 0:

            if maxclass is None:
                maxvar = lcck.shape[1]
            else:
                maxvar = min(maxclass, lcck.shape[1])

            variables = range(basis, maxvar)
            variables = reversed(variables) if direction == 'updown' else variables

            for variable in variables:

                cumulative += lcck[:, variable]
                cumulative[cumulative > fcwk] = fcwk[cumulative > fcwk]

                ax.fill_between(xk, lagged - baseline, cumulative - baseline, facecolor=colors[variable], alpha = 0.7, interpolate=True)
                if variable < lcck.shape[1]-2:
                    ax.plot(xk, cumulative - baseline, colors[variable], linewidth = 0.8)

                lagged += lcck[:, variable]
                lagged[lagged > fcwk] = fcwk[lagged > fcwk]

def PlotCorridorLimit(ax, data, window=1, basis=2):

    x = data['measure']
    fcw = data['fcw8']
    lcc = data['lcc'][:, :, 0] + data['lcc'][:, :, 1]

    if window > 1:
        fcw = fcw.rolling(swath=window, min_periods=1, center=True).mean()
        lcc = lcc.rolling(swath=window, min_periods=1, center=True).mean()

    parts = np.split(
        np.column_stack([
            x,
            fcw,
            lcc
        ]),
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

        baseline = np.sum(lcck[:, :basis], axis=1)
        baseline[baseline > fcwk] = fcwk[baseline > fcwk]

        ax.plot(x, fcwk - baseline, 'darkgray', linewidth = 1.0)


def Plot(data1, data2, kind='corridor', title='', filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    ax.set_xlabel("Location along reference axis (from network outlet)")
    ax.set_ylabel('Width (m)')
    
    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)
    
    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    
    ax.grid(which='both', axis='both', alpha=0.5)

    if kind == 'corridor':

        PlotContinuityProfile(ax, data1, window=5, basis=0)
        PlotCorridorLimit(ax, data1, window=5, basis=0)
        profile = MkCorridorProfile(data2)
        PlotProfile(ax, profile)

    else:

        PlotContinuityProfile(ax, data1, window=5, basis=0, maxclass=2)
        profile = MkActiveChannelProfile(data2)
        PlotProfile(ax, profile)

    # if len(args) > 1:
    # ax.legend()

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

def test():

    mpl.use('cairo')

    fcw = FluvialCorridorWidth(1044)
    lcc = ContinuityWidth(1044)
    data1 = fcw.merge(lcc).sortby(fcw['measure'])
    data2 = ReadData()

    Plot(
        data1,
        data2,
        'corridor',
        title='Comparaison jeu de données EW',
        filename=os.path.join(workdir, 'AXES', 'AX1044', 'PDF', 'COMPARAISON_FDV_EW.pdf'))

    Plot(
        data1,
        data2,
        'active channel',
        title='Comparaison jeu de données EW',
        filename=os.path.join(workdir, 'AXES', 'AX1044', 'PDF', 'COMPARAISON_ACW_EW.pdf'))
