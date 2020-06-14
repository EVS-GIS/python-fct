# coding: utf-8

import numpy as np
from collections import defaultdict
from multiprocessing import Pool
import click

from config import tileindex, filename
from PreProcessing import ReadDEMTile
from Command import starcall
import speedup

def TileMinMax(row, col):
    """
    Returns (minz, maxz) for tile (row, col)
    """

    nodata = -99999.0
    elevations = ReadDEMTile(row, col)
    elevations = elevations[elevations != nodata]
    
    if elevations.size == 0:
        return row, col, nodata, nodata

    return row, col, np.min(elevations), np.max(elevations)

def MinMax(processes=1, **kwargs):
    """
    >>> from operator import itemgetter
    >>> mmx = MinMax(6)
    >>> for (row, col), (zmin, zmax) in sorted([t for t in mmx.items()], key=itemgetter(1)): 
    >>>    print(row, col, zmin, zmax)
    """

    tile_index = tileindex()
    minmax = dict()

    arguments = ([TileMinMax, row, col, kwargs] for row, col in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(tile_index)) as bar:
            for row, col, zmin, zmax in bar:
                minmax[row, col] = (zmin, zmax)

    return minmax

def TileHypsometry(row, col, zbins):
    """
    DOCME
    """

    # TODO
    # use optional mask

    elevations = ReadDEMTile(row, col)
    nodata = -99999.0

    binned = np.digitize(elevations, zbins)
    binned[elevations == nodata] = 0
    # represented = set(np.unique(binned))

    # def area(k):

    #     if k in represented:
    #         return np.count_nonzero(binned == k)
        
    #     return 0

    # areas = {k: area(k) for k in range(1, zbins.size)}
    # areas[0] = 25.0*np.count_nonzero(elevations == nodata)

    return speedup.count_by_value(binned)

def Hypsometry(processes=1, **kwargs):
    """
    """

    tile_index = tileindex()
    areas = defaultdict(lambda: 0)

    minz = 0.0
    maxz = 4800.0
    dz = 10.0
    zbins = np.arange(minz, maxz + dz, dz)

    arguments = ([TileHypsometry, row, col, zbins, kwargs] for row, col in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(tile_index)) as bar:
            for t_areas in bar:
                areas.update({k: areas[k] + 25.0e-6*t_areas[k] for k in t_areas})

    return zbins, np.array([areas[k] for k in range(0, zbins.size)])

def plot_hypsometry(zbins, areas):

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from Plotting import MapFigureSizer

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,150,bottom=0.15,left=0.1,right=1.0,top=1.0)

    nodata_area = areas[0]
    areas = areas[1:]

    z = zbins[1:]
    ax = fig.add_subplot(gs[10:95, 40:140])
    cum_areas = np.flip(np.cumsum(np.flip(areas, axis=0)), axis=0)
    total_area = np.sum(areas)

    # if hypsometry:
    #     ax.fill_between(100 *cum_areas / total_area, 0, z, color='#f2f2f2')
    #     ax.plot(100 * cum_areas / total_area, z, color='k', linestyle='--', linewidth=0.6)

    ax.fill_between(100 * cum_areas / total_area, 0, z, color='lightgray')
    ax.plot(100 * cum_areas / total_area, z, color='k', linewidth=1.0)

    minz = 0.0
    maxz = 4810.0
    dz = 100.0

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.set_xlabel("Cumulative surface (%)")
    ax.set_ylim(minz, maxz)
    ax.tick_params(axis='both', width=1, pad = 2)

    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)

    z = np.arange(minz, maxz + dz, dz)
    groups = np.digitize(zbins[:-1], z)
    
    ax = fig.add_subplot(gs[10:95, 10:30])

    # if hypsometry:
    #     grouped_hyp = np.array([np.sum(areas[groups == k]) for k in range(1, z.size)])
    #     ax.barh(z[:-1], 100.0 * grouped_hyp / total_area, dz, align='edge', color='#f2f2f2', edgecolor='k')

    grouped = np.array([np.sum(areas[groups == k]) for k in range(1, z.size)])
    ax.barh(z[:-1], 100.0 * grouped / total_area, dz, align='edge', color='lightgray', edgecolor='k')

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.set_xlabel("Surface (%)")
    ax.set_ylim(minz, maxz)
    ax.set_ylabel("Altitude (m)")
    ax.tick_params(axis='both', width=1, pad = 2)

    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)

    fig_size_inches = 12.50
    aspect_ratio = 2.0
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc = cbar_L, title = "None")

    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])

    return fig

def test():

    zbins, areas = Hypsometry(6)
    fig = plot_hypsometry(zbins, areas)
    fig.show()

if __name__ == '__main__':
    test()
