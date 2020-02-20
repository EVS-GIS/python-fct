#!/usr/bin/env python
# coding: utf-8

"""
Assemble les polygones des zones hydro redélimitées sur le MNT

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import click
import os
import fiona

@click.command()
@click.argument('zonelist')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--output', '-o', default='ZONEHYDRO.shp')
def AggregateZoneHydro5m(zonelist, overwrite, output):
    """
    Assemble les polygones des zones hydro redélimitées sur le MNT
    """

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists : %s' % output, fg='yellow')
        return

    with open(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    dst = None
    count = 0

    def write(feature, driver, schema, crs):
        """
        Write feature to output
        """

        nonlocal dst

        if dst is None:
            options = dict(driver=driver, schema=schema, crs=crs)
            dst = fiona.open(output, 'w', **options)

        dst.write(feature)

    def progress_status(item):
        if item is not None:
            return item[1]
        return ''

    with click.progressbar(zones, label='Aggregate features', item_show_func=progress_status) as progress:
        for bassin, zone in progress:
            
            shapefile = os.path.join(bassin, zone, 'ZONEHYDRO5M.shp')
            
            with fiona.open(shapefile) as fs:
                for feature in fs:
                    write(feature, fs.driver, fs.schema, fs.crs)
                    count += 1

    if dst:
        click.secho('Wrote %d features to %s' % (count, output), fg='green')
        dst.close()

if __name__ == '__main__':
    AggregateZoneHydro5m()
