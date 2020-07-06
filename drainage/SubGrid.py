#!/usr/bin/env python
# coding: utf-8

"""
SubGrid Procedures

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
import itertools
from collections import defaultdict
from operator import itemgetter
from multiprocessing import Pool

import numpy as np

import click
from affine import Affine
import rasterio as rio
from rasterio.windows import Window
import fiona
from shapely.geometry import asShape

import terrain_analysis as ta
import speedup
from Command import starcall
from config import tileindex, parameter

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset
    width = col_end - col_offset

    return Window(col_offset, row_offset, width, height)

def DefineSubGrid():
    """
    Using np.ma conventions:
    True means masked/nodata
    """

    subgrid_shapefile = os.path.join(workdir, 'SUBGRID', 'SubGrid200m.shp')
    output = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')
    resolution = 200.0

    with fiona.open(subgrid_shapefile) as fs:

        x0, y1, x1, y0 = fs.bounds
        height = int((y0 - y1) // resolution)
        width = int((x1- x0) // resolution)
        transform = Affine.from_gdal(x0, resolution, 0.0, y0, 0.0, -resolution)

        mask = np.full((height, width), True)

        def intile(i, j):
            return all([
                i >= 0,
                i < height,
                j >= 0,
                j < width
            ])

        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                
                cx = 0.5 * (properties['right'] + properties['left'])
                cy = 0.5 * (properties['top'] + properties['bottom'])
                i, j = ta.index(cx, cy, transform)
                
                assert(intile(i, j))
                
                mask[i, j] = False

        profile = dict(
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='uint8',
            compress='deflate',
            transform=transform,
            crs=rio.crs.CRS.from_epsg(2154)
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(np.uint8(mask), 1)

        click.echo('Dimensions:   %d x %d' % (height, width))
        click.echo('GeoTransform: %s' % list(transform.to_gdal()))
        click.echo('CRS:          %s' % 'EPSG:2154')
        click.echo('Data:         %d' % np.sum(~mask))
        click.echo('Nodata:       %d' % np.sum(mask))

def AggregateCell(i, j, datafile, window, fun, nodata):

    with rio.open(datafile) as datas:

        data = datas.read(1, window=window, boundless=True, fill_value=datas.nodata)
        return i, j, fun(data, datas, nodata)

def AggregateMetric(datafile, output, fun, dtype, nodata, processes, **kwargs):

    mask_file = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')

    with rio.open(mask_file) as ds:

        mask = ds.read(1)
        height, width = mask.shape

        out = np.zeros_like(mask, dtype=dtype)

        with rio.open(datafile) as datas:

            resolution = int(ds.transform.a // datas.transform.a)
            half_resolution = int(resolution // 2)
            arguments = list()

            for i in range(height):
                for j in range(width):

                    if mask[i, j] == 0:

                        x, y = ds.xy(i, j)
                        di, dj = datas.index(x, y)

                        window = Window(
                            dj - half_resolution,
                            di - half_resolution,
                            resolution,
                            resolution)

                        arguments.append((AggregateCell, i, j, datafile, window, fun, nodata, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)

            with click.progressbar(pooled, length=len(arguments)) as iterator:
                for i, j, value in iterator:
                    out[i, j] = value

        out[mask == 1] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype=dtype,
            nodata=nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def aggregate_sum(data, ds, nodata):

    masked = np.ma.masked_array(data, data == ds.nodata)
    value = np.ma.sum(masked)

    if value is np.ma.masked:
        return nodata

    return value

def AggregatePopulation(processes=1):

    datafile = os.path.join(workdir, 'GLOBAL', 'POP_2015.vrt')
    output = os.path.join(workdir, 'SUBGRID', 'POP_2015.tif')

    AggregateMetric(
        datafile,
        output,
        aggregate_sum,
        'uint32',
        np.iinfo('uint32').max,
        processes)

def AggregateLandCoverCell(i, j, datafile, window, n):

    with rio.open(datafile) as datas:

        data = datas.read(1, window=window, boundless=True, fill_value=datas.nodata)
        return i, j, speedup.count_by_uint8(data, datas.nodata, n)

def AggregateLandCover(processes=1, **kwargs):

    mask_file = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')
    datafile = os.path.join(workdir, 'GLOBAL', 'LANDCOVER_2018.vrt')
    output = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018.tif')

    dtype = 'float32'
    nodata = -99999.0
    classes = 9

    with rio.open(mask_file) as ds:

        mask = ds.read(1)
        height, width = mask.shape

        out = np.zeros((classes, height, width), dtype=dtype)

        with rio.open(datafile) as datas:

            resolution = int(ds.transform.a // datas.transform.a)
            pixel_area = 1 / resolution**2
            half_resolution = int(resolution // 2)
            arguments = list()

            for i in range(height):
                for j in range(width):

                    if mask[i, j] == 0:

                        x, y = ds.xy(i, j)
                        di, dj = datas.index(x, y)

                        window = Window(
                            dj - half_resolution,
                            di - half_resolution,
                            resolution,
                            resolution)

                        arguments.append((AggregateLandCoverCell, i, j, datafile, window, classes, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)

            with click.progressbar(pooled, length=len(arguments)) as iterator:
                for i, j, counts in iterator:
                    out[:, i, j] = counts[:-1] * pixel_area

        out[:, mask == 1] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype=dtype,
            nodata=nodata,
            compress='deflate',
            count=classes
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out)

def DominantLandCover():

    landcover_file = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018.tif')
    mask_file = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')
    output = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_DOMINANT.tif')

    with rio.open(landcover_file) as ds:

        landcover = ds.read()
        # landcover[landcover == ds.nodata] = 0

        dominant = np.argmax(landcover, axis=0)

        with rio.open(mask_file) as mskds:

            mask = mskds.read(1)
            dominant[mask == 1] = 255

            profile = ds.profile.copy()
            profile.update(
                count=1,
                dtype='uint8',
                nodata=255,
                compress='deflate')

            with rio.open(output, 'w', **profile) as dst:
                dst.write(np.uint8(dominant), 1)

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

    mask_raster = os.path.join(workdir, 'SUBGRID', 'SUBGRID_MASK.tif')
    flow_raster = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'
    acc_raster = '/var/local/fct/RMC/ACC_RGE5M_TILES.vrt'
    output = os.path.join(workdir, 'SUBGRID', 'SUBGRID_OUTLETS.shp')

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

    flow_raster = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'
    outlet_shapefile = os.path.join(workdir, 'SUBGRID', 'SUBGRID_OUTLETS.shp')

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

                if fid in (78751, 77683):
                    if intile(i, j):
                        print(fid, 'in tile')
                    else:
                        print(fid, 'not in tile')

                while intile(i, j):

                    direction = flow[i, j]
                    if direction in (nodata, noflow):
                        if fid in (78751, 77683):
                            print('fid', 'flow stop')
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

    tile_shapefile = '/media/crousson/Backup/PRODUCTION/OCSOL/GRILLE_10K_AIN.shp'
    flow_raster = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'
    outlet_shapefile = os.path.join(workdir, 'SUBGRID', 'SUBGRID_OUTLETS.shp')

    spillovers = dict()
    tiles = dict()

    with fiona.open(tile_shapefile) as fs:

        minx, miny, maxx, maxy = fs.bounds

        for feature in fs:

            geometry = asShape(feature['geometry'])
            x0, y0, x1, y1 = geometry.bounds
            x = 0.5 * (x0 + x1)
            y = 0.5 * (y0 + y1)

            row = int((maxy - y) // 10000)
            col = int((x - minx) // 10000)
            tiles[row, col] = geometry.bounds

    with fiona.open(outlet_shapefile) as fs:

        feature = fs.get(fid)
        x, y = feature['geometry']['coordinates']

        row = int((maxy - y) // 10000)
        col = int((x - minx) // 10000)

        print(row, col)

        bounds = tiles[row, col]

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

    tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')
    graph = dict()
    spillovers = dict()
    tiles = dict()

    with fiona.open(tile_shapefile) as fs:

        minx, miny, maxx, maxy = fs.bounds

        with click.progressbar(fs) as iterator:
            for feature in iterator:

                geometry = asShape(feature['geometry'])
                x0, y0, x1, y1 = geometry.bounds
                x = 0.5 * (x0 + x1)
                y = 0.5 * (y0 + y1)

                row = int((maxy - y) // 10000)
                col = int((x - minx) // 10000)
                tiles[row, col] = geometry.bounds

                subgraph, subspillovers = TileSubGraph(row, col, geometry.bounds)

                graph.update(subgraph)
                spillovers.update(subspillovers)

    step = 0
    index = itemgetter(1, 2)
    processed = set()

    while spillovers:

        step += 1
        click.echo('Step %d, %d spillovers' % (step, len(spillovers)))

        for fid in (78751, 77683):
            if fid in spillovers:
                print(fid, spillovers[fid])

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

                    graph.update(subgraph)
                    spillovers.update(subspillovers)

    return graph

def ExportSubGraph(graph):

    outlet_shapefile = os.path.join(workdir, 'SUBGRID', 'SUBGRID_OUTLETS.shp')
    output = os.path.join(workdir, 'SUBGRID', 'SUBGRID_GRAPH.shp')

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

    graph_shapefile = os.path.join(workdir, 'SUBGRID', 'SUBGRID_GRAPH.shp')
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

    outlet_shapefile = os.path.join(workdir, 'SUBGRID', 'SUBGRID_OUTLETS.shp')
    graph = dict()

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

    return graph

def CreateBufferMask(axis, distance):

    subgrid_raster = os.path.join(
        workdir,
        'SUBGRID',
        'SUBGRID_MASK.tif')

    outlet_shapefile = os.path.join(
        workdir,
        'SUBGRID',
        'SUBGRID_OUTLETS.shp')

    buffer_raster = os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'BUFFER%.0f.vrt' % distance)

    output = os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'SUBGRID_MASK_BUFFER%.0f.tif' % distance)

    with rio.open(subgrid_raster) as ds:

        mask = np.zeros((ds.height, ds.width), dtype='uint8')
        profile = ds.profile.copy()

        with rio.open(buffer_raster) as src:
            with fiona.open(outlet_shapefile) as fs:
                with click.progressbar(fs.filter(bbox=src.bounds)) as iterator:
                    for feature in iterator:

                        i = feature['properties']['i']
                        j = feature['properties']['j']
                        x, y = feature['geometry']['coordinates']
                        buf_value = next(src.sample([(x, y)], 1))
                        
                        if buf_value > 1:
                            mask[i, j] = 1

        with rio.open(output, 'w', **profile) as dst:
            dst.write(mask, 1)

    return mask

def AccumulatePopulation(graph, axis=None, mask=None, dataset=''):
    """
    DOCME
    """
    
    population_raster = os.path.join(workdir, 'SUBGRID', 'POP_2015.tif')
    
    if axis is not None:
        output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, dataset, 'POP_2015_ACC.tif')
    else:
        output = os.path.join(workdir, 'SUBGRID', 'POP_2015_ACC.tif')
    
    nodata = -99999.0

    with rio.open(population_raster) as ds:

        population = ds.read(1)
        nodata_mask = (population == ds.nodata)
        population = np.float32(population)
        
        if mask is None:
            population[nodata_mask] = 0.0
        else:
            population[nodata_mask | mask] = 0.0
        
        out = np.zeros_like(population)
        
        speedup.raster_acc(population, graph, out=out, coeff=0.001)
        out[nodata_mask] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype = 'float32',
            nodata = nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def AccumulateLandcover(graph, axis=None, mask=None, dataset=''):
    """
    DOCME
    """
    
    landcover_raster = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018.tif')

    if axis is not None:
        output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, dataset, 'LANDCOVER_2018_ACC.tif')
    else:
        output = os.path.join(workdir, 'SUBGRID', 'LANDCOVER_2018_ACC.tif')

    nodata = -99999.0

    with rio.open(landcover_raster) as ds:

        landcover = ds.read()
        nodata_mask = (landcover == ds.nodata)

        click.echo('Apply mask')
        print(landcover.shape, nodata_mask.shape, mask.shape)
        
        if mask is None:
            landcover[nodata_mask] = 0.0
        else:
            # for k in range(ds.count):
            #     landcover[nodata_mask[k, :, :] | mask] = 0.0
            landcover[nodata_mask | mask[np.newaxis, :, :]] = 0.0

        click.echo('Accumulate')

        out = np.zeros_like(landcover)
        
        speedup.multiband_raster_acc(landcover, graph, out=out, coeff=0.04)
        out[nodata_mask] = nodata

        profile = ds.profile.copy()
        profile.update(
            dtype = 'float32',
            nodata = nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out)

def workflow():

    click.secho('Create SubGrid Data', fg='cyan')

    click.echo('Define SubGrid')
    DefineSubGrid()

    click.echo('Aggregate Population')
    AggregatePopulation(7)
    click.echo('Aggregate Land Cover')
    AggregateLandCover(7)
    click.echo('Calculate Dominant Land Cover')
    DominantLandCover()

    click.secho('Accumulate SubGrid', fg='cyan')

    click.echo('Find Grid Outlets')
    SubGridOutlets(7)
    click.echo('Build Grid Graph')
    graph = SubGraph()
    click.echo('Write Graph Shapefile')
    ExportSubGraph(graph)
    pixgraph = AsPixelGraph(graph)
    click.echo('Accumulate Population')
    AccumulatePopulation(pixgraph)
    click.echo('Accumulate Land Cover')
    AccumulateLandcover(pixgraph)


# if __name__ == '__main__':
#     workflow()
