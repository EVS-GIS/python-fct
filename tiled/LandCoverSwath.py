import os
import math
from multiprocessing import Pool

import numpy as np
import click

import rasterio as rio
from rasterio.windows import Window
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

import matplotlib.pyplot as plt
import matplotlib as mpl
from Plotting import MapFigureSizer

import terrain_analysis as ta
from ransac import LinearModel, ransac
from Command import starcall

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset + 1
    width = col_end - col_offset + 1

    return Window(col_offset, row_offset, width, height)

def UnitLandCoverSwath(axis, gid, bounds):
    """
    Calculate Land Cover Swath Profile for Valley Unit (axis, gid)
    """

    dgo_raster = os.path.join(workdir, 'AX%03d_DGO.vrt' % axis)
    distance_raster = os.path.join(workdir, 'AX%03d_AXIS_DISTANCE.vrt' % axis)
    landcover_raster = os.path.join(workdir, 'AX%03d_CONTINUITY.vrt' % axis)
    relz_raster = os.path.join(workdir, 'AX%03d_NEAREST_RELZ.vrt' % axis)

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(relz_raster) as ds1:
            relz = ds1.read(1, window=window, boundless=True, fill_value=ds1.nodata)

        with rio.open(landcover_raster) as ds2:
            landcover = ds2.read(1, window=window, boundless=True, fill_value=ds2.nodata)

        with rio.open(dgo_raster) as ds3:
            mask = (ds3.read(1, window=window, boundless=True, fill_value=ds3.nodata) == gid)
            mask = mask & (relz < 20.0)

        assert(all([
            relz.shape == distance.shape,
            landcover.shape == distance.shape,
            mask.shape == distance.shape
        ]))

        xbins = np.arange(np.min(distance[mask]), np.max(distance[mask]), 10.0)
        binned = np.digitize(distance, xbins)
        x = 0.5*(xbins[1:] + xbins[:-1])

        density = np.zeros_like(x, dtype='int32')

        # Profile density

        for i in range(1, len(xbins)):
            density[i-1] = np.sum(mask & (binned == i))

        # Land cover classes count

        classes = np.unique(landcover)
        swath = np.zeros((len(x), len(classes)), dtype='uint16')

        for k, value in enumerate(classes):

            data = (landcover == value)

            for i in range(1, len(xbins)):
                swath[i-1, k] = np.sum(data[mask & (binned == i)])

        values = dict(
            x=x,
            density=density,
            classes=classes,
            swath=swath
        )

        return axis, gid, values

def LandCoverSwath(axis, processes=1):
    """
    DOCME
    """

    dgo_shapefile = os.path.join(workdir, 'AX%03d_DGO.shp' % axis)
    
    def output(axis, gid):
        return os.path.join(workdir, 'OCS', 'AX%03d_SWATH_%04d.npz' % (axis, gid))

    kwargs = dict()
    profiles = dict()
    arguments = list()

    with fiona.open(dgo_shapefile) as fs:
        for feature in fs:

            gid = feature['properties']['GID']
            measure = feature['properties']['M']
            geometry = asShape(feature['geometry'])

            profiles[axis, gid] = [axis, gid, measure]
            arguments.append([UnitLandCoverSwath, axis, gid, geometry.bounds, kwargs])

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:

            for axis, gid, values in iterator:

                profile = profiles[axis, gid]

                np.savez(
                    output(axis, gid),
                    profile=profile,
                    **values)

def plot_swath(x, classes, swath, title=None, filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    ax.set_ylabel("Cover Class Proportion")
    ax.set_xlabel("Distance from reference axis (m)")
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

    cumulative = np.zeros_like(x, dtype='uint16')
    lagged = np.copy(cumulative)

    count = np.sum(swath, axis=1)
    count[count == 0] = 1

    # TODO do not plot zeros

    for k in range(swath.shape[1]):

        if classes[k] == 255:
            continue

        cumulative += swath[:, k]

        ax.fill_between(x, lagged / count, cumulative / count, facecolor=colors[classes[k]], alpha = 0.6, interpolate=True)
        ax.plot(x, cumulative / count, colors[classes[k]], linewidth = 1.0)

        lagged += swath[:, k]

    fig_size_inches = 6.25
    aspect_ratio = 2
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

def PlotSwath(axis, gid, output=None):

    filename = os.path.join(workdir, 'OCS', 'AX%03d_SWATH_%04d.npz' % (axis, gid))

    if os.path.exists(filename):

        data = np.load(filename, allow_pickle=True)
        x = data['x']
        classes = data['classes']
        swath = data['swath']
        _, _,  measure = data['profile']

        bins = np.linspace(np.min(x), np.max(x), int((np.max(x) - np.min(x)) // 30.0) + 1)
        binned = np.digitize(x, bins)
        
        _x = -0.5 * (bins[1:] + bins[:-1])
        # _width = bins[1:] - bins[:-1]
        _swath = np.zeros((_x.size, classes.size), dtype='uint16')
        print(_swath.shape)

        for i in range(1, len(bins)):
            for k in range(len(classes)):
                _swath[i-1, k] = np.sum(swath[binned == i][:, k])

        if swath.shape[0] == x.shape[0]:
            title = 'Land Cover Swath Profile #%d, PK %.1f km' % (gid, measure / 1000.0)
            if output is True:
                output = os.path.join(workdir, 'OCS', 'AX%03d_SWATH_%04d.pdf' % (axis, gid))
            plot_swath(_x, classes, _swath, title, output)
        else:
            click.secho('Invalid swath data')
