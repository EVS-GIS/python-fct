#!/usr/bin/env python
# coding: utf-8

"""
DOCME

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
from collections import defaultdict
import numpy as np
import rasterio as rio
import fiona

@click.command()
@click.argument('zonelist')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def FixZoneHydroOutlet(zonelist, workdir, overwrite):
    """
    Recherche un exutoire cohérent entre les plans de drainage
    qui se superposent d'une zone à l'autre,
    c'est-à-dire un point qui soit sur le réseau théorique
    de toutes les zones superposées,
    en recherchant dans le voisinage du point déterminé à partir d'une seule zone.
    """

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
    
    def make_neighborhood(n):
        """
        Create a search neighborhood of size n x n.
        Elements in neighborhood are on offset
        in x and y in world coordinates.
        """
        k = n // 2
        a = np.arange(-k, k+1)
        dx = a.repeat(n).reshape(n, n).T
        dy = a.repeat(n).reshape(n, n)
        return np.array([resx*dx, resy*dy]).reshape(2, n**2).T
    
    neighborhood = make_neighborhood(5)

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

                    if cdzonehydr == zone:
                        continue

                    i, j = ds.index(x, y)

                    if isdata(i, j):

                        value = next(ds.sample([(x, y)], 1))
                        if value >= min_value:
                            continue

                        try:

                            values = [
                                (k, v)
                                for k, v in enumerate(ds.sample(neighborhood + (x, y), 1))
                                if v >= min_value
                            ]

                            fixes[current].extend(values)

                        except IndexError:

                            values = list()

                            for k, (xk, yk) in enumerate(neighborhood + (x, y)):

                                ik, jk = ds.index(xk, yk)
                                if isdata(ik, jk):
                                    v = next(ds.sample([(xk, yk)], 1))
                                    values.append((k, v))

                            fixes[current].extend(values)

    schema = {
        'geometry': 'Point',
        'properties': [('CDZONEHYDRO', 'str:4'), ('DRAINAGE', 'int'), ('FIXED', 'int:1')]}
    options = dict(driver=driver, crs=crs, schema=schema)

    with fiona.open(output, 'w', **options) as dst:

        num_fixed = 0

        for current, (x, y, cdzonehydr, drainage) in enumerate(outlets):

            pixels = defaultdict(lambda: 0)

            for i, value in fixes[current]:
                pixels[i] += value

            if pixels:

                xy = neighborhood + (x, y)
                pixels = reversed(sorted([(v, i) for i, v in pixels.items()]))

                acc_raster = os.path.join(workdir, cdzonehydr[0], cdzonehydr, 'FLOW_ACCUMULATION.tif')

                with rio.open(acc_raster) as ds:

                    for vk, ik in pixels:

                        xk, yk = xy[ik]
                        value = next(ds.sample([(xk, yk)], 1))

                        if value >= min_value:

                            x, y = xk, yk
                            fixed = 1
                            num_fixed += 1
                            break

                    else:

                        fixed = 0

            else:

                fixed = 0

            geom = {'type': 'Point', 'coordinates': [x, y]}
            props = {'CDZONEHYDRO': cdzonehydr, 'DRAINAGE': drainage, 'FIXED': fixed}
            dst.write({'geometry': geom, 'properties': props})

    click.secho('Fixed %d outlets' % num_fixed, fg='green')

if __name__ == '__main__':
    FixZoneHydroOutlet()
