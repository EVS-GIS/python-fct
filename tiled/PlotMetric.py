import os
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import EngFormatter

from Plotting import MapFigureSizer

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def PlotMetric(axis, y, x='measure'):

    filename = os.path.join(workdir, 'METRICS', 'AX%03d_METRICS.csv' % axis)
    fields = None

    known = defaultdict(lambda: float)

    known.update({
        'axis': int,
        'gid': int,
        'measure': float,
        'elevation': float,
        'slope': float,
        'drainage': float,
    })

    def recast(values, fields):

        return [known[field](value) for field, value in zip(fields, values)]

    if os.path.exists(filename):

        data = list()

        with open(filename) as fp:
            for line in fp:

                values = line.strip().split(',')

                if fields is None:
                    fields = [value.lower() for value in values]
                    xidx = fields.index(x)
                    yidx = fields.index(y)
                else:
                    data.append(recast(values, fields))

        data = np.array(data)
        print(data)

        fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
        gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
        ax = fig.add_subplot(gs[25:100,10:95])

        ax.spines['top'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.set_ylabel(y)
        ax.set_xlabel(x)
        ax.set_xlim([np.max(data[:, xidx]), np.min(data[:, xidx])])
        formatter = EngFormatter(unit='m')
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', width=1, pad = 2)
        for tick in ax.xaxis.get_major_ticks():
            tick.set_pad(2)
        ax.grid(which='both', axis='both', alpha=0.5)

        ax.plot(data[:, xidx], data[:, yidx], "#48638a", linewidth = 1)

        fig.show()
