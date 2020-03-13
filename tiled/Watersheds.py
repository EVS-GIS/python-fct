# coding: utf-8

from collections import defaultdict, Counter
import click
import rasterio as rio
import fiona
import speedup
from config import tileindex, filename, parameter

# Input:
# - RHT hiérarchisé (Hack)
# - Plan de drainage

# 0. Hiérarchisation réseau hydro
#    - Identifiy Network Nodes
#    - Measure Network From Outlet
#    - Hack Order
#    - Strahler Order
#    - Drainage Area
# 1. Graph pixel A -> pixel B, rang, longueur
# 2. Éliminer les noeuds
#    qui sont à une distance < d0 du noeud aval de _même rang_
# 3. Numéroter les noeuds -> Numéro de BVU
#    Identifier la tuile d'appartenance
# 4. Watershed Analysis par tuile
#    Identifier BVU des inlets -> outlets
# 5. Refaire Watershed analysis par tuile
#    en ajoutant les outlets
# 6. Vectoriser chaque tuile
# 7. Aggréger les polygones jointifs qui appartiennent au même BVU

tile_index = tileindex()

network = dict()
reverse = defaultdict(list)
nodes = dict()

with fiona.open(filename('streams-ordered')) as fs:
    for feature in fs:

        props = feature['properties']
        length = props['LENGTH']
        order = props['HACK']
        a = props['NODEA']
        b = props['NODEB']

        network[a] = (b, order, length)
        reverse[b].append(a)

        nodes[(b, order)] = tuple(feature['geometry']['coordinates'][-2])

def distance_to_downstream(node, network, selection):
    """
    Return distance to first activated downstream node of same order
    """

    if node not in network:
        return float('inf'), 1

    a = node
    b, order, distance = network[node]

    while b in network and (b, order) not in selection:

        b, downstream_order, downstream_length = network[b]
        
        if downstream_order != order:
            break
        
        distance += downstream_length

    return distance, order


queue = [network[a][0] for a in network if network[a][0] not in network]
selection = set()
d0 = 2500.0

for node in queue:
    selection.add((node, 1))

while queue:

    node = queue.pop(0)
    distance, order = distance_to_downstream(node, network, selection)

    # Skip trivial tile junction
    if len(reverse[node]) == 1:
        queue.extend(reverse[node])
        continue

    if  distance > d0:
        selection.add((node, order))

    for upstream in reverse[node]:
        up_order = network[upstream][1]
        if up_order > order:
            selection.add((node, up_order))
        queue.append(upstream)

import fiona.crs

schema = {
    'geometry': 'Point',
    'properties': [
        ('NODEID', 'int'),
        ('ORDER', 'int:2'),
        ('TILE', 'int:4'),
        ('ROW', 'int:3'),
        ('COL', 'int:3'),
    ]
}
driver = 'ESRI Shapefile'
crs = fiona.crs.from_epsg(int(parameter('input.srs')))
options = dict(driver=driver, crs=crs, schema=schema)

dem_raster = filename('dem', 'input')
height = int(parameter('input.height'))
width = int(parameter('input.width'))
output = '/media/crousson/Backup/PRODUCTION/RGEALTI/TESTILES/RGE5M_TILE_WATERSHED_OUTLET.shp'

with rio.open(dem_raster) as dem:
    with fiona.open(output, 'w', **options) as dst:
        for node, order in selection:
            if (node, order) in nodes:
                coords = nodes[(node, order)]
                i, j = dem.index(*coords[:2])
                row = i // height
                col = j // width
                tile = tile_index[row, col]
                geom = {'type': 'Point', 'coordinates': coords}
                props = {
                    'NODEID': node,
                    'ORDER': order,
                    'TILE': tile.gid,
                    'ROW': row,
                    'COL': col
                }
                dst.write({'geometry': geom, 'properties': props})
