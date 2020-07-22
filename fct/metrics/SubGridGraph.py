# coding: utf-8

"""
Calculate SubTopography Graph from Flow Raster

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import itertools
from collections import defaultdict
from operator import itemgetter
from multiprocessing import Pool

import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona
import fiona.crs

from .. import terrain_analysis as ta
from .. import speedup
from ..cli import starcall
from ..config import config
from ..tileio import as_window

def SubGridOutlet(window, i, j, flow_raster, acc_raster):

    with rio.open(flow_raster) as ds1:
        with rio.open(acc_raster) as ds2:

            flow = ds1.read(1, window=window, boundless=True, fill_value=ds1.nodata)
            acc = ds2.read(1, window=window, boundless=True, fill_value=ds2.nodata)
            (io, jo), area = speedup.region_outlet(flow, acc)

            if io == -1:
                return (i, j), None, 0.0, 0.0

            transform = ds1.transform * ds1.transform.translation(window.col_off, window.row_off)
            x, y = ta.xy(io, jo, transform)

            return (i, j), (x, y), area, acc[io, jo]

def SubGridOutlets(processes=1, **kwargs):

    mask_raster = config.filename('subgrid_mask')
    # flow_raster = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'
    # acc_raster = '/var/local/fct/RMC/ACC_RGE5M_TILES.vrt'
    flow_raster = config.datasource('flow').filename
    acc_raster = config.datasource('acc').filename
    output = config.filename('subgrid_outlets')

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('i', 'int'),
            ('j', 'int'),
            ('area', 'float:10.0'),
            ('drainage', 'float:10.3')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, schema=schema, crs=crs)

    with rio.open(mask_raster) as ds:

        mask = ds.read(1)
        height, width = mask.shape

        with rio.open(acc_raster) as accds:

            resolution = int(ds.transform.a // accds.transform.a)
            # pixel_area = 1 / resolution**2
            half_resolution = int(resolution // 2)
            arguments = list()

            for i in range(height):
                for j in range(width):

                    if mask[i, j] == 0:

                        x, y = ds.xy(i, j)
                        di, dj = accds.index(x, y)

                        window = Window(
                            dj - half_resolution,
                            di - half_resolution,
                            resolution,
                            resolution)

                        arguments.append((SubGridOutlet, window, i, j, flow_raster, acc_raster, kwargs))

            with Pool(processes=processes) as pool:

                pooled = pool.imap_unordered(starcall, arguments)

                with fiona.open(output, 'w', **options) as dst:
                    with click.progressbar(pooled, length=len(arguments)) as iterator:
                        for (i, j), outlet, area, drainage in iterator:

                            if outlet is not None:

                                # drainage = float(next(accds.sample([outlet], 1)))

                                geometry = {'type': 'Point', 'coordinates': outlet}
                                properties = {
                                    'i': i,
                                    'j': j,
                                    'area': float(area * 25.0),
                                    'drainage': float(drainage)
                                }

                                dst.write({'geometry': geometry, 'properties': properties})

def TileSubGraph(row, col, bounds, items=None):

    flow_raster = config.datasource('flow').filename
    outlet_shapefile = config.filename('subgrid_outlets')

    ci = [-1, -1,  0,  1,  1,  1,  0, -1]
    cj = [ 0,  1,  1,  1,  0, -1, -1, -1]
    nodata = -1
    noflow = 0

    # graph: feature A --(outlet xb, yb)--> feature B
    graph = dict()
    spillovers = dict()

    with rio.open(flow_raster) as ds:

        window = as_window(bounds, ds.transform)
        flow = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        height, width = flow.shape
        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)

        def intile(i, j):

            return all([
                i >= 0,
                i < height,
                j >= 0,
                j < width
            ])

        def translate(i, j):

            if i < 0:
                trow = row - 1
                ti = i + height
            elif i >= height:
                trow = row + 1
                ti = i - height
            else:
                trow = row
                ti = i

            if j < 0:
                tcol = col - 1
                tj = j + width
            elif j >= width:
                tcol = col + 1
                tj = j - width
            else:
                tcol = col
                tj = j

            return trow, tcol, ti, tj

        outlets = defaultdict(list)

        with fiona.open(outlet_shapefile) as fs:
            for feature in fs.filter(bbox=bounds):

                x, y = feature['geometry']['coordinates']
                i, j = ta.index(x, y, transform)
                outlets[i, j].append(int(feature['id']))

        if items:

            resolved = outlets
            outlets = defaultdict(list)

            for fid, row, col, i, j in items:
                outlets[i, j].append(fid)

        else:

            resolved = dict()

        for (i, j) in outlets:
            for fid in outlets[i, j]:

                while intile(i, j):

                    direction = flow[i, j]

                    if direction in (nodata, noflow):
                        break

                    x = int(np.log2(direction))

                    i = i + ci[x]
                    j = j + cj[x]

                    if (i, j) in resolved:

                        if resolved[i, j] != fid:
                            graph[fid] = resolved[i, j][0]
                            break

                    if (i, j) in outlets:

                        if outlets[i, j] != fid:
                            graph[fid] = outlets[i, j][0]
                            break

                else:

                    spillovers[fid] = translate(i, j)

    return graph, spillovers

def LinkOutlet(fid):
    """
    For debug purpose
    """

    flow_raster = config.datasource('flow').filename
    outlet_shapefile = config.filename('subgrid_outlets')
    tileindex = config.tileset('landcover').tileindex
    minx, _, _, maxy = config.tileset('landcover').bounds

    spillovers = dict()
    tiles = dict()

    with fiona.open(outlet_shapefile) as fs:

        feature = fs.get(fid)
        x, y = feature['geometry']['coordinates']

        row = int((maxy - y) // 10000)
        col = int((x - minx) // 10000)

        print(row, col)

        tile = tileindex[row, col]
        bounds = tile.bounds

        with rio.open(flow_raster) as ds:

            window = as_window(bounds, ds.transform)
            transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)
            i, j = ta.index(x, y, transform)

    print(fid, row, col, i, j)
    items = [(fid, row, col, i, j)]

    subgraph, subspillovers = TileSubGraph(row, col, bounds, items)

    # graph.update(subgraph)
    spillovers.update(subspillovers)

    step = 0
    index = itemgetter(1, 2)
    processed = set()

    while spillovers:

        step += 1
        click.echo('Step %d, %d spillovers' % (step, len(spillovers)))

        unresolved = [(fid, row, col, i, j) for fid, (row, col, i, j) in spillovers.items()]
        unresolved = sorted([item for item in unresolved if item not in processed], key=index)
        processed.update(unresolved)

        groups = itertools.groupby(unresolved, key=index)
        spillovers = dict()

        with click.progressbar(groups, length=len(tiles)) as iterator:
            for (row, col), items in iterator:
                if (row, col) in tiles:

                    bounds = tiles[row, col]
                    subgraph, subspillovers = TileSubGraph(row, col, bounds, items)

                    # graph.update(subgraph)
                    spillovers.update(subspillovers)

    return subgraph

def SubGraph():

    tileindex = config.tileset('landcover').tileindex
    graph = dict()
    spillovers = dict()

    with click.progressbar(tileindex.values()) as iterator:
        for tile in iterator:

            subgraph, subspillovers = TileSubGraph(tile.row, tile.col, tile.bounds)

            graph.update(subgraph)
            spillovers.update(subspillovers)

    step = 0
    index = itemgetter(1, 2)
    processed = set()

    while spillovers:

        step += 1
        click.echo('Step %d, %d spillovers' % (step, len(spillovers)))

        unresolved = [(fid, row, col, i, j) for fid, (row, col, i, j) in spillovers.items()]
        unresolved = sorted([item for item in unresolved if item not in processed], key=index)
        processed.update(unresolved)

        groups = itertools.groupby(unresolved, key=index)
        spillovers = dict()

        with click.progressbar(groups, length=len(tileindex)) as iterator:
            for (row, col), items in iterator:
                if (row, col) in tileindex:

                    tile = tileindex[row, col]
                    subgraph, subspillovers = TileSubGraph(row, col, tile.bounds, items)

                    graph.update(subgraph)
                    spillovers.update(subspillovers)

    return graph

def ExportSubGraph(graph):

    outlet_shapefile = config.filename('subgrid_outlets')
    output = config.filename('subgrid_graph')

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'LineString',
        'properties': [
            ('i', 'int'),
            ('j', 'int'),
            ('area', 'float:10.0'),
            ('drainage', 'float:10.3'),
            ('nodea', 'int'),
            ('nodeb', 'int'),
            ('outleti', 'int'),
            ('outletj', 'int')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, schema=schema, crs=crs)

    with fiona.open(outlet_shapefile) as fs:
        with fiona.open(output, 'w', **options) as dst:

            with click.progressbar(graph) as iterator:
                for fid in iterator:

                    feature = fs.get(fid)
                    target_fid = graph[fid]
                    target = fs.get(target_fid)

                    x, y = feature['geometry']['coordinates']
                    xe, ye = target['geometry']['coordinates']
                    feature['geometry'] = {
                        'type': 'LineString',
                        'coordinates': [[x, y], [xe, ye]]
                    }

                    feature['properties'].update(
                        outleti=target['properties']['i'],
                        outletj=target['properties']['j'],
                        nodea=fid,
                        nodeb=target_fid
                    )

                    dst.write(feature)

def LoadSubGraph():

    graph_shapefile = config.filename('subgrid_graph')
    graph = dict()

    with fiona.open(graph_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                nodea = properties['nodea']
                nodeb = properties['nodeb']
                graph[nodea] = nodeb

    return graph

def AsPixelGraph(feature_graph):

    outlet_shapefile = config.filename('subgrid_outlets')
    graph = dict()
    outlets = dict()

    with fiona.open(outlet_shapefile) as fs:
        with click.progressbar(feature_graph) as iterator:
            for fid in iterator:

                feature = fs.get(fid)
                target_fid = feature_graph[fid]
                target = fs.get(target_fid)

                i = feature['properties']['i']
                j = feature['properties']['j']

                ti = target['properties']['i']
                tj = target['properties']['j']

                graph[i, j] = (ti, tj)
                x, y = feature['geometry']['coordinates']
                outlets[i, j] = (fid, x, y)
                x, y = target['geometry']['coordinates']
                outlets[ti, tj] = (target_fid, x, y)

    return graph, outlets

def SubGridProfile(axis, sourcex, sourcey, pixgraph, outlets):

    acc_raster = config.datasource('acc').filename
    measure_raster = config.filename('ax_axis_measure', axis=axis)
    subgrid_raster = config.filename('subgrid_mask')
    output = config.filename('ax_subgrid_profile', axis=axis)
    tileset = config.tileset('landcover')
    tileset_resolution = 10000.0

    def tileset_index(x, y):

        minx, _, _, maxy = tileset.bounds
        row = int((maxy - y) // tileset_resolution)
        col = int((x - minx) // tileset_resolution)
        return row, col

    with rio.open(subgrid_raster) as ds:
        i, j = ds.index(sourcex, sourcey)

    profile = [(i, j)]

    while (i, j) in pixgraph:
        i, j = pixgraph[i, j]
        profile.append((i, j))

    driver = 'ESRI Shapefile'
    crs = fiona.crs.from_epsg(2154)
    schema = {
        'geometry': 'Point',
        'properties': [
            ('GID', 'int'),
            ('AXIS', 'int:5'),
            ('ROW', 'int:5'),
            ('COL', 'int:5'),
            ('M', 'float:10.0'),
            ('DRAINAGE', 'float:10.3'),
            ('TOX', 'float:10.2'),
            ('TOY', 'float:10.2')
        ]
    }
    options = dict(driver=driver, schema=schema, crs=crs)

    with fiona.open(output, 'w', **options) as fst:
        with rio.open(acc_raster) as ds1:
            with rio.open(measure_raster) as ds2:

                for k, (i, j) in enumerate(profile):

                    _, x, y = outlets[i, j]
                    row, col = tileset_index(x, y)

                    if (i, j) in pixgraph:

                        ie, je = pixgraph[i, j]
                        _, xe, ye = outlets[ie, je]

                    else:

                        xe, ye = None, None

                    drainage = float(next(ds1.sample([(x, y)], 1)))
                    measure = float(next(ds2.sample([(x, y)], 1)))
                    geom = {
                        'type': 'Point',
                        'coordinates': [x, y]
                    }
                    properties = {
                        'GID': k+1,
                        'AXIS': axis,
                        'ROW': row,
                        'COL': col,
                        'M': measure,
                        'DRAINAGE': drainage,
                        'TOX': xe,
                        'TOY': ye
                    }

                    fst.write(dict(geometry=geom, properties=properties))

def workflow():

    click.echo('Find Grid Outlets')
    SubGridOutlets(7)
    click.echo('Build Grid Graph')
    graph = SubGraph()
    click.echo('Write Graph Shapefile')
    ExportSubGraph(graph)
    pixgraph = AsPixelGraph(graph)
    return pixgraph
