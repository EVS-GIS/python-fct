# coding: utf-8

"""
Fluvial Corridor Profile Vizualization

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
from matplotlib.ticker import EngFormatter
import click

from .MapFigureSizer import MapFigureSizer

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

def PlotLandCoverProfile(data, basis=0, title='', window=1, proportion=False, direction='upright', filename=None):

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
    fcw = data['fcw2']
    fcw2 = data['fcw8']
    lcc = data['lcw'][:, :, 0] + data['lcw'][:, :, 1]
    fcw[np.isnan(fcw)] = fcw2[np.isnan(fcw)]

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

            baseline = np.sum(lcck[:, :basis], axis=1)
            # baseline[baseline > fcwk] = fcwk[baseline > fcwk]
            fcwk = fcwk - baseline
            lcck = lcck / fcwk[:, np.newaxis]
            baseline = np.zeros_like(fcwk)
            fcwk = np.ones_like(fcwk)

        else:

            baseline = np.sum(lcck[:, :basis], axis=1)
            # baseline[baseline > fcwk] = fcwk[baseline > fcwk]

        cumulative = np.copy(baseline)
        lagged = np.copy(cumulative)

        if xk.size > 0:

            variables = range(basis, lcck.shape[1])
            variables = reversed(variables) if direction == 'updown' else variables

            for variable in variables:

                cumulative += lcck[:, variable]
                # cumulative[cumulative > fcwk] = fcwk[cumulative > fcwk]

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
                # lagged[lagged > fcwk] = fcwk[lagged > fcwk]

    # if not proportion:
    #     ax.plot(x, fcwk - baseline, 'darkgray', linewidth = 1.0)

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
    fcw = data['fcw2']
    fcw2 = data['fcw8']
    lcc = data['lcc'][:, :, 0] + data['lcc'][:, :, 1]
    fcw[np.isnan(fcw)] = fcw2[np.isnan(fcw)]

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

    # print(fcw.shape, left.shape, right.shape)

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
