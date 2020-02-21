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
from collections import defaultdict

@click.command()
@click.argument('zonelist')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def FixZoneHydroOutlet(zonelist, workdir, overwrite):
    """
    DOCME
    """

    import numpy as np
    import rasterio as rio
    import fiona

    output = os.path.join(workdir, 'ZONEHYDRO_FIXED_OUTLETS.shp')
    outlet_shapefile = os.path.join(workdir, 'ZONEHYDRO_OUTLETS.shp')

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists : %s' % output, fg='yellow')
        return

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    def display_item(item):
        if item:
            return item[1]
        return '...'

    def extract_outlet(feature):

        cdzonehydr = feature['properties']['CDZONEHYDR']
        drainage = feature['properties']['DRAINAGE']
        x, y = feature['geometry']['coordinates']

        return x, y, cdzonehydr, drainage

    with fiona.open(outlet_shapefile) as fs:
        crs = fs.crs
        driver = fs.driver
        outlets = [extract_outlet(f) for f in fs]

    resx = 5.0
    resy = -5.0
    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).T
    neighborhood = np.array([resx*dx, resy*dy]).reshape(2,9).T

    min_value = 40000
    fixes = defaultdict(list)

    with click.progressbar(zones, item_show_func=display_item) as progress:
        for basin, zone in progress:

            # flow_raster = os.path.join(workdir, basin, zone, 'FLOW.tif')
            acc_raster = os.path.join(workdir, basin, zone, 'FLOW_ACCUMULATION.tif')

            with rio.open(acc_raster) as ds:

                def isdata(i, j):
                    return i >= 0 and j >=0 and i < ds.height and j < ds.width

                for current, (x, y, cdzonehydr, drainage) in enumerate(outlets):

                    # if cdzonehydr == zone:
                    #     continue

                    i, j = ds.index(x, y)

                    if isdata(i, j):

                        value = next(ds.sample([(x, y)], 1))
                        if value >= min_value:
                            continue

                        try:

                            values = [
                                (i, v)
                                for i, v in enumerate(ds.sample(neighborhood + (x, y), 1))
                                if v >= min_value
                            ]

                            fixes[current].extend(values)

                        except IndexError:
                            continue

    schema = {'geometry': 'Point', 'properties': [('CDZONEHYDRO', 'str:4'), ('DRAINAGE', 'int'), ('FIXED', 'int:1')]}
    options = dict(driver=driver, crs=crs, schema=schema)

    with fiona.open(output, 'w', **options) as dst:

        fixed = 0

        for current, (x, y, cdzonehydr, drainage) in enumerate(outlets):

            pixels = defaultdict(lambda: 0)

            for i, value in fixes[current]:
                pixels[i] += value

            if pixels:

                xy = neighborhood + (x, y)
                pixels = sorted([(v, i) for i, v in pixels.items()])
                v0, i0 = pixels[-1]
                x0, y0 = xy[i0]

                geom = {'type': 'Point', 'coordinates': [x0, y0]}
                props = {'CDZONEHYDRO': cdzonehydr, 'DRAINAGE': drainage, 'FIXED': 1}
                dst.write({'geometry': geom, 'properties': props})

                fixed += 1

            else:

                geom = {'type': 'Point', 'coordinates': [x, y]}
                props = {'CDZONEHYDRO': cdzonehydr, 'DRAINAGE': drainage, 'FIXED': 0}
                dst.write({'geometry': geom, 'properties': props})

    click.secho('Fixed %d outlets' % fixed, fg='green')

if __name__ == '__main__':
    FixZoneHydroOutlet()
