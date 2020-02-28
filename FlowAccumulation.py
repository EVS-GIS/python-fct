#!/usr/bin/env python
# coding: utf-8

"""
Dérive un réseau sous-maille
à partir d'un carroyage et d'un plan de drainage
de résolution différente

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import click
import numpy as np
import terrain_analysis as ta
import rasterio as rio

def CalculateFlowAccumulation(basin, zone, root, overwrite):
    """
    Calculate flow accumulation from D8 flow direction raster.
    """

    output = os.path.join(root, basin, zone, 'FLOW_ACCUMULATION.tif')
    flow_raster = os.path.join(root, basin, zone, 'FLOW.tif')

    if os.path.exists(output) and not overwrite:
        # click.secho('Output already exists : %s' % output, fg='yellow')
        return

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        acc = ta.flow_accumulation(flow)

        profile = ds.profile.copy()
        profile.update(dtype=np.uint32, nodata=0, compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(acc, 1)

    return True

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def zone(basin, zone, workdir, overwrite):
    """
    DOCME
    """

    CalculateFlowAccumulation(basin, zone, workdir, overwrite)

def Starred(args):
    """
    Starred version of `function` for use with pool.imap_unordered()
    """

    return CalculateFlowAccumulation(*args)

@cli.command()
@click.argument('zonelist')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def batch(zonelist, workdir, overwrite, processes):
    """
    DOCME
    """

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    if processes > 1:

        from multiprocessing import Pool

        click.secho('Running %d processes ...' % processes, fg='yellow')
        arguments = (tuple(z) + (workdir, overwrite) for z in zones)

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(Starred, arguments)
            with click.progressbar(pooled, length=len(zones)) as progress:
                for _ in progress:
                    pass

    else:

        def display_item(item):
            if item:
                return item[1]
            return '...'

        with click.progressbar(zones, item_show_func=display_item) as progress:
            for basin, zone in progress:

                CalculateFlowAccumulation(basin, zone, workdir, overwrite)
    

if __name__ == '__main__':
    cli()