# coding: utf-8

"""
Watershed Metrics Plot

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

from matplotlib.ticker import EngFormatter
import click

from .PlotCorridor import (
    SetupPlot,
    FinalizePlot
)

def PlotRefElevation(ax, segments):

    mmin = float('inf')
    mmax = float('-inf')

    for segment in segments:

        if segment.size > 0:

            ax.plot(segment[:, 3], segment[:, 2])
            mmin = min(mmin, np.min(segment[:, 3]))
            mmax = max(mmax, np.max(segment[:, 3]))

    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim([mmax, mmin])

def ShowRefElevation(segments):

    fig, ax = SetupPlot()
    PlotRefElevation(ax, segments)
    FinalizePlot(fig, ax, title='Reference axis elevation profile')

