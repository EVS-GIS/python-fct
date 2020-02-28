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
from heapq import (
    heapify,
    heappop
)

import numpy as np
import click

import fiona
import rasterio as rio
from rasterio.features import rasterize
import terrain_analysis as ta

def riotogdal(transform):
    """
    RasterIO AffineTransform to GDAL GeoTransform tuple
    """

    return (
        transform.c,
        transform.a,
        transform.d,
        transform.f,
        transform.b,
        transform.e
    )

def FindPolygonOutlet(basin, zone, root, overwrite):
    """
    DOCME
    """

    output = os.path.join(basin, zone, 'ZONEHYDRO_OUTLETS.shp')
    # output2 = os.path.join(basin, zone, 'ZONEHYDRO_OUTLET.shp')
    flow_raster = os.path.join(basin, zone, 'FLOW.tif')
    zone_shapefile = os.path.join(basin, zone, 'ZONEHYDRO_BDC.shp')
    min_lca = 1e6 / 25

    if os.path.exists(output) and not overwrite:
        # if os.path.exists(output1):
        #     click.secho('Output already exists : %s' % output1, fg='yellow')
        # if os.path.exists(output2):
        #     click.secho('Output already exists : %s' % output2, fg='yellow')
        return 0
    
    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        mask = np.zeros_like(flow, dtype=np.uint8)

        def shapes():
            with fiona.open(zone_shapefile) as fs:
                for feature in fs:
                    yield feature['geometry'], 1

        rasterize(shapes(), fill=0, transform=ds.transform, out=mask)

        outlets = list()

        with fiona.open(zone_shapefile) as fs:

            for f in fs:

                driver = fs.driver
                crs = fs.crs
                geometry = np.float32(f['geometry']['coordinates'][0])
                outlets.extend(ta.polygon_outlets(geometry, flow, mask, riotogdal(ds.transform)))

        def isdata(i, j):
            return i >= 0 and i < ds.height and j >=0 and j < ds.width

        def path(i, j, flow):
            ix, jx = i, j
            while isdata(ix, jx) and flow[ix, jx] != 0 and flow[ix, jx] != -1:
                yield ix, jx
                direction = int(np.log2(flow[ix, jx]))
                ix = ix + ci[direction]
                jx = jx + cj[direction]

        ci = [-1, -1,  0,  1,  1,  1,  0, -1]
        cj = [ 0,  1,  1,  1,  0, -1, -1, -1]

        def common_outlet(a, b, flow):
            ai, aj = a
            path_a = set(path(ai, aj, flow))
            bi, bj = b
            for i, j in path(bi, bj, flow):
                if (i, j) in path_a:
                    return i, j
            return None

        outlet = None
        score = 0
        zone_area = np.sum(mask)
        count = 0

        schema = {'geometry': 'Point', 'properties': [('CDZONEHYDRO', 'str:4'), ('DRAINAGE', 'int'), ('SCORE', 'float')]}
        options=dict(driver='ESRI Shapefile', schema=schema, crs=crs)

        queue = [(-lca, (i, j)) for (i, j), lca in outlets]
        outlets = list()
        cum_area = 0
        heapify(queue)

        while queue:

            lca, (i, j) = heappop(queue)
            lca = -lca

            if outlets and (lca < min_lca or cum_area > 0.95*zone_area):
                break

            # if lca / zone_area >= 0.005:

            #     if outlet is None:
            #         outlet = (i, j)
            #         score = lca
            #     else:
            #         o = common_outlet(outlet, (i, j), flow)
            #         if o is not None:
            #             outlet = o
            #             score += lca

            outlets.append(((i, j), lca))
            cum_area += lca

            # count += 1
            # if count > 100:
            #     break

        with fiona.open(output, 'w', **options) as dst:
            for (i, j), lca in outlets:

                geom = {
                'type': 'Point',
                'coordinates': ds.xy(i, j)
                }
                props = {
                    'CDZONEHYDRO': zone,
                    'DRAINAGE': lca,
                    'SCORE': lca/zone_area*100
                }
                
                dst.write({'geometry': geom, 'properties': props})

        # if outlet is not None:

        #     with fiona.open(output2, 'w', **options) as dst:
                
        #         geom = {
        #             'type': 'Point',
        #             'coordinates': ds.xy(*outlet)
        #         }
        #         props = {
        #             'CDZONEHYDRO': zone,
        #             'DRAINAGE': score,
        #             'SCORE': score/zone_area*100
        #         }
                
        #         dst.write({'geometry': geom, 'properties': props})

        # else:

        #     click.secho('No outlet found for zone %s' % zone, fg='red')

    return len(outlets)

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def zone(basin, zone, workdir, overwrite):

    output = os.path.join(basin, zone, 'ZONEHYDRO_OUTLETS.shp')
    # output2 = os.path.join(basin, zone, 'ZONEHYDRO_OUTLET.shp')

    # if (os.path.exists(output1) or os.path.exists(output2)) and not overwrite:
    #     if os.path.exists(output1):
    #         click.secho('Output already exists : %s' % output1, fg='yellow')
    #     if os.path.exists(output2):
    #         click.secho('Output already exists : %s' % output2, fg='yellow')
    #     return


    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists : %s' % output, fg='yellow')
        return 

    count = FindPolygonOutlet(basin, zone, workdir, overwrite)

    click.secho('Found %d outlets' % count, fg='green')

@cli.command()
@click.argument('zonelist')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def batch(zonelist, workdir, overwrite):

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    def display_item(item):
        if item:
            return item[1]
        return '...'

    with click.progressbar(zones, item_show_func=display_item) as progress:
        for basin, zone in progress:
            
            FindPolygonOutlet(basin, zone, workdir, overwrite)

if __name__ == '__main__':
    cli()