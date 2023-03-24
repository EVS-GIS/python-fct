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
from multiprocessing import Pool
import click

import rasterio as rio
import numpy as np
import fiona
import fiona.crs

from shapely.geometry import Polygon
from ..config import config
from ..tileio import as_window
from ..cli import starcall

def ExtractTile(datasource, dataset, tile, tileset, overwrite=False):

    raster = datasource
    output = tileset.tilename(dataset, row=tile.row, col=tile.col)

    if os.path.exists(output) and not overwrite:
        return

    with rio.open(raster) as ds:

        window = as_window(tile.bounds, ds.transform)
        data = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)

        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            height=window.height,
            width=window.width,
            transform=transform,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(data, 1)

def DatasourceToTiles(datasource, tileset, dataset, processes=1, **kwargs):

    arguments = list()

    for tile in config.tileset(tileset).tileindex.values():
        arguments.append((ExtractTile, config.datasource(datasource).filename, dataset, tile, config.tileset(tileset), kwargs))
    
    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)
        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def MakeDataTile(tileset, dataset, tile):
    """
    Extract tile within arbitrary tileset
    from global dataset
    """

    flow_raster = config.datasource(dataset).filename
    output = config.tileset(tileset).tilename(
        dataset,
        row=tile.row,
        col=tile.col)

    with rio.open(flow_raster) as ds:
    
        window = as_window(tile.bounds, ds.transform)
        flow = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        height, width = flow.shape
        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)
        
        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            height=height,
            width=width,
            transform=transform,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(flow, 1)

def RetileDatasource(datasource, tileset, processes=1, **kwargs):

    arguments = [
        (MakeDataTile, tileset, datasource, tile, kwargs)
        for tile in config.tileset(tileset).tileindex.values()
    ]

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)
        
        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass


def CreateTileset(datasource: str = 'bdalti', 
                  resolution: float = 10000.0, 
                  tileset1: str = '../outputs/10k_tileset.gpkg',
                  tileset2: str = '../outputs/10kbis_tileset.gpkg'):
    """
    Create two shifted tilesets with a specified resolution, using a datasource defined in config.ini as a reference layer
    """

    schema = { 
        'geometry': 'Polygon', 
        'properties': {'GID': 'int',
                       'ROW': 'int',
                       'COL': 'int',
                       'X0': 'float',
                       'Y0': 'float'} }
    
    options = dict(
        driver='GPKG',
        schema=schema,
        crs=fiona.crs.from_epsg(config.srid))
    
    with rio.open(config.datasource(datasource).filename) as src:
        minx, miny, maxx, maxy = src.bounds
    
    # Tileset 1
    
    minx -= resolution/2
    miny -= resolution/2
    
    maxx += resolution/2
    maxy += resolution/2
    
    gx, gy = np.arange(minx, maxx, resolution), np.arange(miny, maxy, resolution)

    gid = 1
    with fiona.open(tileset1, 'w', **options) as dst:   
        for i in range(len(gx)-1):
            for j in range(len(gy)-1):
                
                coordinates = [(gx[i],gy[j]),(gx[i],gy[j+1]),(gx[i+1],gy[j+1]),(gx[i+1],gy[j])]
                
                feature = {'geometry': {
                            'type':'Polygon',
                            'coordinates': [coordinates] 
                            },
                           'properties': {
                               'GID': gid,
                               'ROW': len(gy)-j-1,
                               'COL': i+1,
                               'Y0': gy[j+1],
                               'X0': gx[i]
                           }
                    }
                
                dst.write(feature)
                gid+=1
    
    # Tileset 2 (shifted)
    
    minx -= (resolution/2)
    miny -= (resolution/2)
    maxx += (resolution/2)
    maxy += (resolution/2)
    
    # nx = int((maxx - minx) // resolution) + 1 
    # ny = int((maxy - miny) // resolution) + 1
    
    gx, gy = np.arange(minx, maxx, resolution), np.arange(miny, maxy, resolution)

    gid = 1
    with fiona.open(tileset2, 'w', **options) as dst:   
        for i in range(len(gx)-1):
            for j in range(len(gy)-1):
                
                coordinates = [(gx[i],gy[j]),(gx[i],gy[j+1]),(gx[i+1],gy[j+1]),(gx[i+1],gy[j])]
                
                feature = {'geometry': {
                            'type':'Polygon',
                            'coordinates': [coordinates] 
                            },
                           'properties': {
                               'GID': gid,
                               'ROW': len(gy)-j-1,
                               'COL': i+1,
                               'Y0': gy[j+1],
                               'X0': gx[i]
                           }
                    }
                
                dst.write(feature)
                gid+=1