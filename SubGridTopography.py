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
from collections import defaultdict
import numpy as np

import click
import fiona
import rasterio as rio
import terrain_analysis as ta

SUCCESS = 'green'
INFO = 'cyan'
WARNING = 'yellow'
ERROR = 'red'

def BuildNetworkGraph(outlets, flow, feedback):
    """
    Link each outlet to the next one
    following the flow direction raster.
    """

    # graph: feature A --(outlet xb, yb)--> feature B
    graph = dict()
    height, width = flow.shape
    ci = [-1, -1,  0,  1,  1,  1,  0, -1]
    cj = [ 0,  1,  1,  1,  0, -1, -1, -1]
    nodata = -1
    noflow = 0

    pixels = {(i, j): (fid, area) for fid, ((i, j), area) in outlets.items()}

    def isdata(i, j):
        """
        True if (py, px) is a valid pixel coordinate
        """

        return j >= 0 and i >= 0 and j < width and i < height

    total = 100.0 / len(outlets) if outlets else 0.0

    # for current, feature in enumerate(layer.getFeatures()):
    for current, fid in enumerate(outlets):

        (i, j), area = outlets[fid]

        while isdata(i, j):

            direction = flow[i, j]
            if direction == nodata or direction == noflow:
                break

            x = int(np.log2(direction))

            i = i + ci[x]
            j = j + cj[x]

            if (i, j) in pixels:

                ide = pixels[(i, j)][0]
                # xe, ye = pixeltoworld(np.array([j, i]), transform)
                # graph[fid] = Outlet(ide, float(xe), float(ye))
                graph[fid] = ide
                break

        feedback.setProgress(int(current * total))

    feedback.setProgress(100)

    return graph

def OutputNetworkNodes(outlets, graph, acc, hashes, ds, grid_shp, output, feedback):
    """
    Write Outlets Shapefile
    """

    reverse_graph = defaultdict(list)
    for a, b in graph.items():
        reverse_graph[b].append(a)

    with fiona.open(grid_shp) as fs:

        schema_props = fs.schema['properties'].copy()
        schema_props.update({
            'LINKHASH': 'str:7',
            'LINKX': 'float',
            'LINKY': 'float',
            'DRAINAGE': 'int',
            'LOCALCA': 'int',
            'CONTRIB': 'int'
        })
        schema = {
            'geometry': 'Point',
            'properties': schema_props
        }
        options = dict(
            driver=fs.driver,
            crs=fs.crs,
            schema=schema
        )

        count = 0
        total = 100.0 / len(outlets) if outlets else 0.0

        with fiona.open(output, 'w', **options) as dst:
            for fid, feature in enumerate(fs):

                if feature['properties']['CDZONEGEN'] != feature['properties']['CDZONEHYDR']:
                    continue

                (i, j), lca = outlets[fid]
                x, y = ds.xy(i, j)
                outlet_area = float(acc[i, j])

                target_id = graph.get(fid, None)

                if target_id is not None:
                    (it, jt), lcat = outlets[target_id]
                    xt, yt = ds.xy(it, jt)
                    target_geohash = hashes[target_id]
                else:
                    xt = yt = None
                    target_geohash = None

                upstream_area = 0.0
                for a in reverse_graph[fid]:
                    if a in outlets:
                        (ia, ja), lcaa = outlets[a]
                        a_area = float(acc[ia, ja])
                        upstream_area += a_area

                contributive_area = outlet_area - upstream_area

                geom = {'type': 'Point', 'coordinates': [x, y]}
                props = feature['properties'].copy()
                props.update({
                    'LINKHASH': target_geohash,
                    'LINKX': xt,
                    'LINKY': yt,
                    'DRAINAGE': outlet_area,
                    'LOCALCA': lca,
                    'CONTRIB': contributive_area
                })
                node = {
                    'geometry': geom,
                    'properties': props
                }

                dst.write(node)
                count += 1
                feedback.setProgress(int(fid * total))

    feedback.setProgress(100)
    click.secho('Output %d nodes to %s' % (count, os.path.basename(output)), fg=SUCCESS)

def OutputLinks(input_shp, output):
    """
    Transforme les noeuds sous-maille en ligne
    pour les représenter sous forme de flèche.
    """

    with fiona.open(input_shp) as fs:

        schema = fs.schema.copy()
        schema['geometry'] = 'LineString'
        options = dict(
            driver=fs.driver,
            crs=fs.crs,
            schema=schema
        )

        with fiona.open(output, 'w', **options) as dst:
            for feature in fs:
                x, y = feature['geometry']['coordinates']
                xt = feature['properties']['LINKX']
                yt = feature['properties']['LINKY']
                geom = {'type': 'LineString', 'coordinates': [(x, y), (xt, yt)]}
                link = {
                    'geometry': geom,
                    'properties': feature['properties']
                }
                dst.write(link)

@click.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--root', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--filename', '-f', default='GRIDEEA_OUTLETS.shp', help='Output basename')
@click.option('--flowdir', default='FLOW.tif', help='Flow Direction filename')
@click.option('--links/--no-links', default=True, help='Disable Links Output')
@click.option('--linkfilename', default='GRIDEEA_LINKS.shp', help='Links basename')
def SubGridTopography(basin, zone, root, overwrite, filename, flowdir, links, linkfilename):
    """
    Dérive un réseau sous-maille
    à partir d'un carroyage et d'un plan de drainage
    de résolution différente
    """

    click.secho('Processing zone %s' % zone, fg=SUCCESS)
    click.secho('Working Directory = %s' % root, fg=INFO)

    grid_shp = os.path.join(root, basin, zone, 'GRIDEEA.shp')
    flow_raster = os.path.join(root, basin, zone, flowdir)
    output = os.path.join(root, basin, zone, filename)

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists : %s' % output, fg=WARNING)
        return

    click.secho('Read input data', fg=INFO)
    click.secho('Flow Direction from %s' % flowdir, fg=INFO)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        acc = ta.flow_accumulation(flow)
        geometries = dict()
        hashes = dict()
        height, width = flow.shape

        with fiona.open(grid_shp) as fs:
            for fid, feature in enumerate(fs):
                geom = np.float32(feature['geometry']['coordinates'][0])
                geometries[fid] = geom
                hashes[fid] = feature['properties']['GEOHASH']

        click.secho('Find outlets', fg=INFO)

        feedback = ta.ConsoleFeedback()
        outlets = ta.subgrid_outlets(geometries, flow, acc, ds.transform.to_gdal(), feedback)
        feedback.setProgress(100)

        click.secho('Build network graph', fg=INFO)
        graph = BuildNetworkGraph(outlets, flow, feedback)

        click.secho('Output network nodes', fg=INFO)
        OutputNetworkNodes(outlets, graph, acc, hashes, ds, grid_shp, output, feedback)

        if links and linkfilename:

            link_output = os.path.join(root, basin, zone, linkfilename)

            if os.path.exists(link_output) and not overwrite:
                click.secho('Links file already exists : %s' % link_output, fg=WARNING)
            else:
                OutputLinks(output, link_output)
                click.secho('Wrote links to %s' % linkfilename, fg=SUCCESS)

if __name__ == '__main__':
    SubGridTopography()
