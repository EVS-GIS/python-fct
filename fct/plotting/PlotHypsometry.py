# coding: utf-8

"""
Plot Hypsometry

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt
from .MapFigureSizer import MapFigureSizer

def PlotHypsometry(hypsometer):

    zbins = hypsometer['z'].values
    areas = hypsometer['area'].values

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,150,bottom=0.15,left=0.1,right=1.0,top=1.0)

    nodata_area = areas[0]
    z = zbins[1:]
    areas = areas[1:]

    ax = fig.add_subplot(gs[10:95, 40:140])
    cum_areas = np.flip(np.cumsum(np.flip(areas, axis=0)), axis=0)
    total_area = np.sum(areas)

    # if hypsometry:
    #     ax.fill_between(100 *cum_areas / total_area, 0, z, color='#f2f2f2')
    #     ax.plot(100 * cum_areas / total_area, z, color='k', linestyle='--', linewidth=0.6)

    ax.fill_between(100 * cum_areas / total_area, 0, z, color='lightgray')
    ax.plot(100 * cum_areas / total_area, z, color='k', linewidth=1.0)

    minz = np.min(z[areas > 0])
    maxz = np.max(z[areas > 0])
    dz = 100.0

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.set_xlabel("Cumulative surface (%)")
    ax.set_xlim([0, 100])
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