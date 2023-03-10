"""
Sequence :

4. Accumulate/Resolve Acc Graph/InletAreas
5. FlowAccumulation (*)
6. StreamToFeature (*)

(*) Possibly Parallel Steps

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
import glob
from collections import defaultdict, Counter
import itertools
from operator import itemgetter

import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from .. import speedup
from .. import terrain_analysis as ta
from ..config import (
    config,
    DatasetParameter,
    DatasourceParameter
)

from multiprocessing import Pool
from ..cli import starcall_nokwargs

def tileindex(tileset='default'):
    """
    Return default tileindex
    """
    return config.tileset(tileset).tileindex

class Parameters():
    """
    Drainage area and network extraction parameters
    """

    # exterior = DatasourceParameter('exterior flow')
    exterior_flow = DatasourceParameter('exterior flow')

    elevations = DatasetParameter('filled-resolved elevation raster (DEM)', type='input')
    flow = DatasetParameter('flow direction raster', type='input')

    outlets = DatasetParameter('tile outlets (point) shapefile', type='output')
    outlets_pattern = DatasetParameter('tile outlets shapefile glob pattern', type=None)
    inlets = DatasetParameter('tile inlets (point) shapefile', type='output')
    inlet_areas = DatasetParameter('', type='output')
    acc = DatasetParameter('accumulation raster (drainage area)', type='output')

    def __init__(self):
        """
        Default parameter values
        """

        self.exterior_flow = 'off' # 'exterior-inlets'
        self.elevations = 'dem'
        self.flow = 'flow'
        self.outlets = 'outlets'
        self.outlets_pattern = 'outlets-glob'
        self.inlets = 'inlets'
        self.inlet_areas = 'inlet-areas'
        self.acc = 'acc'

def TileOutlets(row, col, params, verbose=False, tileset='default'):
    """
    Find tile outlets,
    ie. pixels connecting to anoter tile according to flow direction.
    """

    tile_index = tileindex(tileset)

    crs = fiona.crs.from_epsg(config.srid)
    driver = 'GeoJSON'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('LCA', 'int'),
            ('TO', 'int'),
            ('TOX', 'float'),
            ('TOY', 'float'),
            ('Z', 'float'),
            ('TOZ', 'float')
        ]
    }
    options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

    # read_tile_index()

    flow_raster = params.flow.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('flow', row=row, col=col)

    gid = tile_index[(row, col)].gid
    tiles = defaultdict(list)

    # def readz(trow, tcol, x, y):

    #     with rio.open(config.tileset().tilename('filled', row=trow, col=tcol)) as src:
    #         return float(next(src.sample([(x, y)], 1)))

    with rio.open(flow_raster) as ds:

        height, width = ds.shape
        flow = ds.read(1)
        mask = np.ones_like(flow, dtype=np.uint8)
        outlets, targets = ta.tile_outlets(flow, mask)

        for current, (ti, tj) in enumerate(targets):

            top = (ti < 0)
            bottom = (ti >= height)
            left = (tj < 0)
            right = (tj >= width)

            if top:
                di = -1
            elif bottom:
                di = +1
            else:
                di = 0

            if left:
                dj = -1
            elif right:
                dj = +1
            else:
                dj = 0

            tiles[(row+di, col+dj)].append(current)

                # target = tile_index[(row+di, col+dj)].gid
                # (i, j), area = outlets[current]
                # x, y = ds.xy(i, j)
                # tx, ty = ds.xy(ti, tj)
                # outlet_z = readz(row, col, x, y)
                # target_z = readz(row+di, col+dj, tx, ty)

                # geom = {'type': 'Point', 'coordinates': [x, y]}
                # props = {
                #     'TILE': gid,
                #     'LCA': area,
                #     'TO': target,
                #     'TOX': tx,
                #     'TOY': ty,
                #     'Z': outlet_z,
                #     'TOZ': target_z
                # }

                # dst.write({'geometry': geom, 'properties': props})

        cum_area = 0
        skipped = 0

        schema = {
            'geometry': 'Point',
            'properties': [
                ('TILE', 'int'),
                ('LCA', 'int'),
                ('FROM', 'int'),
                ('FROMX', 'float'),
                ('FROMY', 'float')
            ]
        }
        options = dict(driver=driver, crs=crs, schema=schema)

        for trow, tcol in tiles:

            if (trow, tcol) not in tile_index:
                skipped += len(tiles[(trow, tcol)])
                continue

            target = tile_index[(trow, tcol)].gid
            output = params.outlets.tilename(row=trow, col=tcol, gid=gid, tileset=tileset)
            # config.tileset().tilename('outlets', row=trow, col=tcol, gid=gid)

            # if os.path.exists(output):
            #     mode = 'a'
            # else:
            #     mode = 'w'

            with fiona.open(output, 'w', **options) as dst:
                for idx in tiles[(trow, tcol)]:

                    (i, j), area = outlets[idx]
                    ti, tj = targets[idx]
                    x, y = ds.xy(i, j)
                    tx, ty = ds.xy(ti, tj)

                    cum_area += area
                
                    geom = {'type': 'Point', 'coordinates': [tx, ty]}
                    props = {
                        'TILE': target,
                        'LCA': area,
                        'FROM': gid,
                        'FROMX': x,
                        'FROMY': y
                    }

                    dst.write({'geometry': geom, 'properties': props})

    if verbose:
        click.secho('\nSkipped %d outlets' % skipped, fg='yellow')
        click.secho('Tile (%02d, %02d) Coverage = %.1f %%' % (row, col, (cum_area / (height*width) * 100)), fg='green')

    return cum_area
    
    
def AggregateOutlets(params, tileset='default'):
    """
    Aggregate ROW_COL_INLETS_ORIGIN.geojson files
    into one ROW_COL_INLETS.shp shapefile
    """

    tile_index = tileindex(tileset)

    crs = fiona.crs.from_epsg(config.srid)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('LCA', 'int'),
            ('FROM', 'int'),
            ('FROMX', 'float'),
            ('FROMY', 'float')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            output = params.inlets.tilename(row=row, col=col, tileset=tileset)
            # config.tileset().tilename('inlets', row=row, col=col)

            with fiona.open(output, 'w', **options) as dst:

                pattern = str(params.outlets_pattern.tilename(row=row, col=col, tileset=tileset))
                # config.tileset().tilename(
                #     'outlets-glob',
                #     row=row,
                #     col=col)

                for name in glob.glob(pattern):
                    with fiona.open(name) as fs:

                        for feature in fs:
                            dst.write(feature)

def CreateOutletsGraph(params, exterior='exterior-inlets'):
    """
    DOCME
    """

    tile_index = tileindex()
    elevation_raster = params.elevations.filename()
    # config.tileset().filename('dem')

    click.secho('Build outlets graph', fg='cyan')

    graph = dict()
    indegree = Counter()

    # provide world/pixel geotransform
    dem = rio.open(elevation_raster)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[(row, col)].gid
            inlet_shapefile = params.inlets.tilename(row=row, col=col)
            # config.tileset().tilename('inlets', row=row, col=col)
            flow_raster = params.flow.tilename(row=row, col=col)
            # config.tileset().tilename('flow', row=row, col=col)

            with rio.open(flow_raster) as ds:

                flow = ds.read(1)
                height, width = flow.shape

                with fiona.open(inlet_shapefile) as fs:
                    for feature in fs:

                        # connect outlet->inlet

                        from_tile = feature['properties']['FROM']
                        area = feature['properties']['LCA']
                        from_i, from_j = dem.index(feature['properties']['FROMX'], feature['properties']['FROMY'])
                        i, j = dem.index(*feature['geometry']['coordinates'])
                        graph[(from_tile, from_i, from_j)] = (tile, i, j, area)
                        indegree[(tile, i, j)] += 1

                        # connect inlet->tile outlet

                        loci, locj = ds.index(*feature['geometry']['coordinates'])
                        locti, loctj = ta.outlet(flow, loci, locj)
                        ti, tj = dem.index(*ds.xy(locti, loctj))

                        if (locti, loctj) == (loci, locj):
                            continue
                        
                        if ti >= 0 and tj >= 0:
                            graph[(tile, i, j)] = (tile, ti, tj, 0)
                            indegree[(tile, ti, tj)] += 1

                # exterior_flow = params.exterior_flow.filename()

                # if exterior_flow and os.path.exists(exterior_flow):
                if not params.exterior_flow.none:

                    exterior_flow = params.exterior_flow.filename()

                    with fiona.open(exterior_flow) as fs:
                        for feature in fs:

                            loci, locj = ds.index(*feature['geometry']['coordinates'])

                            if not all([loci >= 0, loci < height, locj >= 0, locj < width]):
                                continue

                            # connect exterior->inlet

                            i, j = dem.index(*feature['geometry']['coordinates'])
                            
                            pixelSizeX = ds.profile['transform'][0]
                            pixelSizeY =-ds.profile['transform'][4]
                            coeff = (pixelSizeX*pixelSizeY)*1e-6
                            
                            area = feature['properties']['AREAKM2'] / coeff
                            graph[(-2, i-1, j-1)] = (tile, i, j, area)
                            indegree[(tile, i, j)] += 1

                            # connect inlet->tile outlet

                            locti, loctj = ta.outlet(flow, loci, locj)
                            ti, tj = dem.index(*ds.xy(locti, loctj))

                            if (locti, loctj) == (loci, locj):
                                continue
                            
                            if ti >= 0 and tj >= 0:
                                graph[(tile, i, j)] = (tile, ti, tj, 0)
                                indegree[(tile, ti, tj)] += 1

    dem.close()

    click.secho('Created graph with %d nodes' % len(graph), fg='green')

    return graph, indegree

    # queue = [pixel for pixel in graph if indegree[pixel] == 0]
    # areas = defaultdict(lambda: 0)
    # seen = set()

    # with click.progressbar(length=len(indegree)) as progress:
    
    #     while queue:

    #         tile, i, j = queue.pop(0)

    #         if (tile, i, j) in seen:
    #             continue

    #         progress.update(1)
    #         seen.add((tile, i, j))

    #         if (tile, i, j) in graph:

    #             tile, i, j, area = graph[(tile, i, j)]
    #             areas[(tile, i, j)] += area*25e-6 # convert to km^2
    #             indegree[(tile, i, j)] -= 1

    #             if indegree[(tile, i, j)] == 0:
    #                 queue.append((tile, i, j))

    # return areas


def InletAreasTile(row, col, gid, params, keys, areas, tileset='default'):
    """
    Output inlet points,
    attributed with the total upstream drained area.
    """

    crs = fiona.crs.from_epsg(config.srid)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('AREAKM2', 'float')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    # provide world/pixel geotransform
    dem_file = params.elevations.filename(tileset=tileset)
    # config.tileset().filename('dem')
    dem = rio.open(dem_file)

    cum_areas = defaultdict(lambda: 0.0)

    for key in keys:
        cum_areas[key[1:]] += areas.get(key[1:], 0)

    inlet_shapefile = params.inlets.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('inlets', row=row, col=col)
    emitted = set()

    output = params.inlet_areas.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('inlet-areas', row=row, col=col)

    with fiona.open(output, 'w', **options) as dst:
        with fiona.open(inlet_shapefile) as fs:
            for feature in fs:

                x, y = feature['geometry']['coordinates']
                i, j = dem.index(x, y)

                if (i, j) in emitted:
                    continue

                assert (i, j) in cum_areas
                area = cum_areas[i, j]

                geom = {'type': 'Point', 'coordinates': [x, y]}
                props = {'TILE': gid, 'AREAKM2': area}
                feature = {'geometry': geom, 'properties': props}
                dst.write(feature)

                emitted.add((i, j))

    dem.close()

def InletAreas(params, tileset='default'):
    """
    Accumulate areas across tiles
    and output per tile inlet shapefiles
    with contributing area flowing into tile.
    """

    tile_index = tileindex(tileset)
    tiles = {tile.gid: tile for tile in tile_index.values()}

    graph, indegree = CreateOutletsGraph(params)

    # Check a random tile just to get pixels x and y size
    flow_raster = params.flow.tilename(row=tiles.get(1).row, col=tiles.get(1).col, tileset=tileset)
    with rio.open(flow_raster) as ds:
        pixelSizeX = ds.profile['transform'][0]
        pixelSizeY =-ds.profile['transform'][4]
        
    coeff = (pixelSizeX*pixelSizeY)*1e-6
    
    click.secho('Accumulate areas', fg='cyan')
    areas, res = speedup.graph_acc(graph, coeff)

    keys = sorted(graph.keys() | indegree.keys(), key=itemgetter(0))
    groups = itertools.groupby(keys, key=itemgetter(0))

    click.secho('Write inlet shapefiles', fg='cyan')
    with click.progressbar(groups, length=len(tile_index)) as progress:
        for tile_gid, keys in progress:

            if tile_gid in tiles:
                tile = tiles[tile_gid]
                InletAreasTile(tile.row, tile.col, tile.gid, params, keys, areas, tileset)

def FlowAccumulationTile(row, col, params, overwrite, tileset):
    """
    Calculate D8 flow direction tile
    """

    flow_raster = params.flow.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('flow', row=row, col=col)
    inlet_shapefile = params.inlet_areas.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('inlet-areas', row=row, col=col)
    output = params.acc.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('acc', row=row, col=col)

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        
        pixelSizeX = ds.profile['transform'][0]
        pixelSizeY =-ds.profile['transform'][4]
        
        coeff = (pixelSizeX*pixelSizeY)*1e-6

        out = np.full_like(flow, coeff, dtype='float32')
        height, width = flow.shape

        if os.path.exists(inlet_shapefile):
            with fiona.open(inlet_shapefile) as fs:
                for feature in fs:

                    i, j = ds.index(*feature['geometry']['coordinates'])
                    out[i, j] += feature['properties']['AREAKM2']

        # with fiona.open(filename('exterior-inlets')) as fs:
        #     for feature in fs:
        #         i, j = ds.index(*feature['geometry']['coordinates'])
        #         if all([i >= 0, i < height, j >= 0, j < width]):
        #             out[i, j] += feature['properties']['AREAKM2']

        speedup.flow_accumulation(flow, out)

        # click.secho('Save to %s' % output, fg='green')

        if os.path.exists(inlet_shapefile):
            with fiona.open(inlet_shapefile) as fs:
                for feature in fs:

                    x, y = feature['geometry']['coordinates']
                    i, j = ds.index(x, y)
                    out[i, j] = feature['properties']['AREAKM2']

        profile = ds.profile.copy()
        profile.update(compress='deflate', nodata=0, dtype=np.float32)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)


def Outlets(params, verbose=False, tileset='default', processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                TileOutlets,
                row,
                col,
                params,
                verbose,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
            

def FlowAccumulation(params, overwrite, tileset='default', processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                FlowAccumulationTile,
                row,
                col,
                params,
                overwrite,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass