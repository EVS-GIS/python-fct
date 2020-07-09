# coding: utf-8

"""
Elevation Swath Vizualization

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

def plot_swath(x, swath, ylabel=None, title=None, filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    if ylabel:
        # ax.set_ylabel("Relative Elevation (m)")
        ax.set_ylabel(ylabel)
    # else:
    #     ax.set_ylabel("Elevation (m NGF)")
    ax.set_xlabel("Distance from reference axis (m)")
    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    ax.grid(which='both', axis='both', alpha=0.5)

    parts = np.split(
        np.column_stack([x, swath]),
        np.where(np.isnan(swath[:, 0]))[0])

    for k, part in enumerate(parts):

        if k == 0:

            xk = part[:, 0]
            swathk = part[:, 1:]

        else:

            xk = part[1:, 0]
            swathk = part[1:, 1:]

        if xk.size > 0:

            ax.fill_between(xk, swathk[:, 0], swathk[:, 4], facecolor='#b9d8e6', alpha = 0.2, interpolate=True)
            ax.plot(xk, swathk[:, 0], "gray", xk, swathk[:, 4], "gray", linewidth = 0.5, linestyle='--')
            ax.fill_between(xk, swathk[:, 1], swathk[:, 3], facecolor='#48638a', alpha = 0.5, interpolate=True)
            ax.plot(xk, swathk[:, 2], "#48638a", linewidth = 1)

    # fig_size_inches = 6.25
    # aspect_ratio = 3
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

def PlotSwath(axis, gid, kind='absolute', output=None):

    # filename = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'ELEVATION', 'SWATH_%04d.npz' % gid)

    if kind not in ('absolute', 'hand', 'hvf'):
        click.secho('Unknown swath kind %s' % kind, fg='yellow')
        return

    filename = config.filename('ax_swath_elevation', axis=axis, gid=gid)

    if os.path.exists(filename):

        data = np.load(filename, allow_pickle=True)

        x = data['x']
        _, _,  measure = data['profile']

        if kind == 'absolute':

            swath = data['sz']
            ylabel = 'Elevation (m NGF)'

        elif kind == 'hand':

            swath = data['hand']
            ylabel = 'Height above nearest drainage (m)'

        elif kind == 'hvf':

            swath = data['hvf']
            ylabel = 'Height above valley floor (m)'

            if swath.size == 0:
                click.secho('No relative-to-valley-bottom swath profile for DGO (%d, %d)' % (axis, gid), fg='yellow')
                click.secho('Using relative-to-nearest-drainage profile', fg='yellow')
                swath = data['hand']

        if swath.shape[0] == x.shape[0]:
            title = 'Swath Profile #%d, PK %.1f km' % (gid, measure / 1000.0)
            if output is True:

                output = config.filename(
                    'pdf_ax_swath_elevation',
                    axis=axis,
                    gid=gid,
                    kind=kind.upper())

            plot_swath(-x, swath, ylabel, title, output)
        else:
            click.secho('Invalid swath data')
