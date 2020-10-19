# coding: utf-8

"""
LandCover Swath Vizualization

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import click

from ..config import config
from .MapFigureSizer import MapFigureSizer

def plot_swath_landcover(x, classes, swath, direction='forward', title=None, filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    ax.set_ylabel("Cover Class Proportion", fontsize=14)
    ax.set_xlabel("Distance from reference axis (m)", fontsize=14)
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

    count = np.sum(swath, axis=1)

    # ====================================================================
    
    # count[count == 0] = 1
    # cumulative = np.zeros_like(x, dtype='uint16')
    # lagged = np.copy(cumulative)

    # for k in range(swath.shape[1]):

    #     if classes[k] == 255:
    #         continue

    #     cumulative += swath[:, k]

    #     ax.fill_between(x, lagged / count, cumulative / count, facecolor=colors[classes[k]], alpha = 0.6, interpolate=True)
    #     ax.plot(x, cumulative / count, colors[classes[k]], linewidth = 1.0)

    #     lagged += swath[:, k]

    # ====================================================================

    # cumulative = np.zeros_like(x, dtype='uint16')
    # lagged = np.copy(cumulative)

    # for variable in range(swath.shape[1]):

    #     if classes[variable] == 255:
    #         continue

    #     cumulative += swath[:, variable]

    #     parts = np.split(
    #         np.column_stack([x, count, lagged, cumulative]),
    #         np.where((count == 0) | (swath[:, variable] == 0))[0])

    #     for k, part in enumerate(parts):

    #         if k == 0:

    #             xk = part[:, 0]
    #             countk = part[:, 1]
    #             laggedk = part[:, 2]
    #             cumulativek = part[:, 3]

    #         else:

    #             xk = part[1:, 0]
    #             countk = part[1:, 1]
    #             laggedk = part[1:, 2]
    #             cumulativek = part[1:, 3]

    #         # print(k, xk.shape, countk.shape, swathk.shape)

    #         # cumulative = np.zeros_like(xk)
    #         # lagged = np.copy(cumulative)

    #         if xk.size > 0:

    #             countk[countk == 0] = 1
    #             laggedk = laggedk / countk
    #             cumulativek = cumulativek / countk

    #             # cumulative += swathk / countk

    #             ax.fill_between(xk, laggedk, cumulativek, facecolor=colors[classes[variable]], alpha = 0.6, interpolate=True)
    #             ax.plot(xk, cumulativek, colors[classes[variable]], linewidth = 1.0)

    #             # lagged += swathk / countk

    #     lagged += swath[:, variable]

    # ====================================================================

    # Do not plot zeros

    parts = np.split(
        np.column_stack([x, count, swath]),
        np.where(count == 0)[0])

    for k, part in enumerate(parts):

        if k == 0:

            xk = part[:, 0]
            countk = part[:, 1]
            swathk = part[:, 2:]
        
        else:

            xk = part[1:, 0]
            countk = part[1:, 1]
            swathk = part[1:, 2:]

        # print(k, xk.shape, countk.shape, swathk.shape)

        cumulative = np.zeros_like(xk)
        lagged = np.copy(cumulative)

        if xk.size > 0:

            variables = range(swath.shape[1])
            variables = reversed(variables) if direction == 'reversed' else variables

            for variable in variables:

                if classes[variable] == 255:
                    continue

                cumulative += swathk[:, variable] / countk

                ax.fill_between(xk, lagged, cumulative, facecolor=colors[classes[variable]], alpha = 0.7, interpolate=True)
                ax.plot(xk, cumulative, colors[classes[variable]], linewidth = 1.0)

                lagged += swathk[:, variable] / countk

    fig_size_inches = 12.5
    aspect_ratio = 6
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    # if axis == 1:
    #     if swath == 135:
    # ax.set_xlim((-584.1333333333334, 394.1333333333333))
        # elif swath == 196:
    ax.set_xlim((-256.073974609375, 457.92602539062494))

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()

def plot_swath_continuity(x, classes, swath, direction='forward', title=None, filename=None):

    klass_labels = [1, 10, 20, 30, 40, 50]

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
        '#0050c8', # Active channel
        'darkgreen', # Riparian corridor
        '#6f9e00', # Semi-natural
        'orange', # Reversible
        '#f2f2f2', # Disconnected
        '#4d4d4d' # Built environment
    ]

    count = np.sum(swath, axis=1)

    parts = np.split(
        np.column_stack([x, count, swath]),
        np.where(count == 0)[0])

    for k, part in enumerate(parts):

        if k == 0:

            xk = part[:, 0]
            countk = part[:, 1]
            swathk = part[:, 2:]
        
        else:

            xk = part[1:, 0]
            countk = part[1:, 1]
            swathk = part[1:, 2:]

        # print(k, xk.shape, countk.shape, swathk.shape)

        cumulative = np.zeros_like(xk)
        lagged = np.copy(cumulative)

        if xk.size > 0:

            variables = range(swath.shape[1])
            variables = reversed(variables) if direction == 'reversed' else variables

            for variable in variables:

                if classes[variable] == 255:
                    continue

                cumulative += swathk[:, variable] / countk

                klass_index = klass_labels.index(classes[variable])
                ax.fill_between(xk, lagged, cumulative, facecolor=colors[klass_index], alpha=0.35, interpolate=True)
                ax.plot(xk, cumulative, colors[klass_index], linewidth=0.9, zorder=len(variables) - variable)

                lagged += swathk[:, variable] / countk

    fig_size_inches = 12.5
    aspect_ratio = 6
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    print(ax.get_xlim())

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()

def PlotLandCoverSwath(axis, gid, kind='continuity', direction='forward', output=None):

    # if kind == 'continuity':
    #     filename = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'CONTINUITY', 'SWATH_CONTINUITY_%04d.npz' % gid)
    # elif kind == 'landcover':
    #     filename = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'LANDCOVER', 'SWATH_LANDCOVER_%04d.npz' % gid)
    # else:

    if kind not in ('std', 'continuity', 'interpreted'):
        click.secho('Unknown landcover swath kind %s' % kind, fg='yellow')
        return

    if kind == 'std':
        subset = 'TOTAL_BDT'
        plot_swath = plot_swath_landcover
    elif kind == 'continuity':
        subset = 'CONT_BDT'
        plot_swath = plot_swath_landcover
    elif kind == 'interpreted':
        subset = 'REMAPPED'
        plot_swath = plot_swath_continuity

    filename = config.filename('ax_swath_landcover_npz', axis=axis, gid=gid, kind=kind.upper(), subset=subset)

    if os.path.exists(filename):

        data = np.load(filename, allow_pickle=True)
        x = data['x']
        classes = data['landcover_classes']
        swath = data['landcover_swath']
        _, _,  measure = data['profile']

        bins = np.linspace(np.min(x), np.max(x), int((np.max(x) - np.min(x)) // 30.0) + 1)
        binned = np.digitize(x, bins)
        
        _x = -0.5 * (bins[1:] + bins[:-1])
        # _width = bins[1:] - bins[:-1]
        _swath = np.zeros((_x.size, classes.size), dtype='uint16')

        for i in range(1, len(bins)):
            for k in range(len(classes)):
                _swath[i-1, k] = np.sum(swath[binned == i][:, k])

        if swath.shape[0] == x.shape[0]:
            title = 'Land Cover Swath Profile #%d, PK %.1f km' % (gid, measure / 1000.0)
            if output is True:
                # if kind == 'continuity':
                #     output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SWATH_CONTINUITY_%04d.pdf' % gid)
                # elif kind == 'landcover':
                #     output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SWATH_LANDCOVER_%04d.pdf' % gid)
                output = config.filename('pdf_ax_swath_landcover', axis=axis, gid=gid, kind=kind.upper())
            plot_swath(_x, classes, _swath, direction, title, output)
        else:
            click.secho('Invalid swath data')

def test():

    for dgo in [45, 47, 61, 75, 87, 122, 125, 167, 190, 412, 630, 811, 819, 885, 897]: 
        PlotLandCoverSwath(1044, dgo, 'continuity', output=True) 