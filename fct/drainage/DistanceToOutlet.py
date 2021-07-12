"""
Compute distance to drainage outlet
for every point in space
"""

from collections import defaultdict
import itertools
from multiprocessing import Pool

import numpy as np

import click
import rasterio as rio

from .. import speedup
from ..cli import starcall
from ..tileio import border
from ..config import (
    config,
    DatasetParameter,
    LiteralParameter
)

class Parameters():
    """
    Distance to drainage outlet parameters
    """

    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    flow = DatasetParameter('flow direction raster', type='input')
    output = DatasetParameter('distance to network outlet', type='output')
    conversion = LiteralParameter('pixels to output unit conversion factor')

    def __init__(self):
        """
        Default parameter values
        """

        self.tiles = 'shortest_tiles'
        self.flow = 'flow'
        self.output = 'network-outlet-distance'
        # 5.0 m pixels to kilometers
        self.conversion = 5.0 / 1000.0

def ConnectTile(row: int, col: int, params: Parameters):
    """
    Connect every border pixel to local tile outlet (inbound pixel)
    or to other tile (outbound pixel)
    (t, i, j) -> (to, io, jo, do)
    """

    ci = [ -1, -1,  0,  1,  1,  1,  0, -1 ]
    cj = [  0,  1,  1,  1,  0, -1, -1, -1 ]

    flow_raster = params.flow.tilename(row=row, col=col)
    graph = dict()
    t = row, col

    with rio.open(flow_raster) as ds:

        height, width = ds.shape
        flow = ds.read(1)

    def pixel_identifier(i, j):

        tik, tjk = t
        ik, jk = i, j

        if i < 0:
            tik = tik - 1
            ik = ik + height
        elif i >= height:
            tik = tik + 1
            ik = ik - height
        
        if j < 0:
            tjk = tjk - 1
            jk = jk + width
        elif j >= width:
            tjk = tjk + 1
            jk = jk - width

        assert((0 <= ik < height) and (0 <= jk < width))

        return (tik, tjk), ik, jk

    # def distance_to_outlet(i, j):

    #     distance = 0
    #     direction = flow[i, j]

    #     while direction != -1 and direction != 0:

    #         n = int(np.log2(direction))
    #         ik = i + ci[n]
    #         jk = j + cj[n]

    #         if (0 <= ik < height) and (0 <= jk < width):

    #             # distance in pixels
    #             distance += np.sqrt((ik-i)**2 + (jk-j)**2)
    #             i, j = ik, jk
    #             direction = flow[i, j]

    #         else:

    #             break

    #     return distance

    for i, j in border(height, width):
    # with click.progressbar(border(height, width), length=2*(height+width)) as iterator:
    #     for i, j in iterator:

        direction = flow[i, j]
        
        if direction != -1 and direction != 0:

            n = int(np.log2(direction))
            ik = i + ci[n]
            jk = j + cj[n]

            if (0 <= ik < height) and (0 <= jk < width):
                
                # inbound pixel (i, j)
                tk = t
                # dk = distance_to_outlet(i, j)
                ik, jk, dk = speedup.pix_distance_to_outlet(flow, i, j)
                graph[(t, i, j)] = (tk, ik, jk, dk)

            else:
                
                # outbound pixel (i, j)
                dk = np.sqrt((ik-i)**2 + (jk-j)**2)
                tk, ik, jk = pixel_identifier(ik, jk)
                # distance in pixels
                graph[(t, i, j)] = (tk, ik, jk, dk)

    return graph

def TileFlow(params: Parameters, processes: int = 1, **kwargs):
    """
    Build flow graph between tiles by connecting tiles together
    """

    if params.tiles.none:

        tileset = config.tileset()

        def length():

            return len(tileset)

        def arguments():

            for row, col in tileset.tileindex:

                yield (
                    ConnectTile,
                    row,
                    col,
                    params,
                    kwargs
                )

    else:

        tilefile = params.tiles.filename(**kwargs)

        def length():

            with open(tilefile) as fp:
                return sum(1 for line in fp)

        def arguments():

            with open(tilefile) as fp:
                for line in fp:

                    row, col = tuple(int(x) for x in line.split(','))

                    yield (
                        ConnectTile,
                        row,
                        col,
                        params,
                        kwargs
                    )

    graph = dict()

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        with click.progressbar(pooled, length=length()) as iterator:
            for gt in iterator:
                
                graph.update(gt)

    return graph

def ResolveTileFlow(graph: dict, params: Parameters):
    """
    Dispatch distance from network outlets
    """

    distance = defaultdict(lambda: 0.0)
    reverse_graph = defaultdict(list)
    conversion = params.conversion

    for (t, i, j), (tk, ik, jk, dk) in graph.items():

        reverse_graph[(tk, ik, jk)].append((t, i, j, dk))

        if (tk, ik, jk) not in graph:

            distance[(tk, ik, jk)] = 0.0

    outlets = set(distance.keys())

    for outlet in outlets:

        stack = [outlet]

        while stack:

            node = stack.pop(0)

            for t, i, j, d in reverse_graph[node]:

                upstream = (t, i, j)
                distance[upstream] = distance[node] + d*conversion
                stack.append(upstream)

    return distance, outlets

def DistanceTile(row: int, col: int, distance_outlets: dict, params: Parameters):
    """
    Dispatch distance from tile outlets
    """

    ci = [ -1, -1,  0,  1,  1,  1,  0, -1 ]
    cj = [  0,  1,  1,  1,  0, -1, -1, -1 ]

    flow_raster = params.flow.tilename(row=row, col=col)
    output = params.output.tilename(row=row, col=col)
    conversion = params.conversion

    if not flow_raster.exists():
        return

    with rio.open(flow_raster) as ds:

        height, width = ds.shape
        nodata_flow = ds.nodata
        profile = ds.profile.copy()
        flow = ds.read(1)

    distance = np.zeros_like(flow, dtype='float32')
    seen = np.zeros_like(flow, dtype='bool')
    stack = list()

    for (i, j), d in distance_outlets.items():
        
        distance[i, j] = d
        stack.append((i, j))
        seen[i, j] = True

    speedup.distance_to_outlet(flow, distance, conversion)

    # while stack:

    #     i, j = stack.pop()

    #     for n in range(8):
            
    #         ik = i - ci[n]
    #         jk = j - cj[n]

    #         if (0 <= ik < height) and (0 <= jk < width):

    #             direction = flow[ik, jk]

    #             if direction != -1 and direction != 0:

    #                 nk = int(np.log2(direction))
    #                 if nk == n:

    #                     # distance in pixels
    #                     d = np.sqrt((ik-i)**2 + (j-jk)**2)
    #                     distance[ik, jk] = distance[i, j] + d*conversion

    #                     if not seen[ik, jk]:

    #                         stack.append((ik, jk))
    #                         seen[ik, jk] = True

    nodata = -99999.0
    distance[flow == nodata_flow] = nodata
    profile.update(dtype='float32', compress='deflate', nodata=nodata)

    with rio.open(output, 'w', **profile) as dst:
        dst.write(distance, 1)

def DistanceToOutlet(distance: dict, params: Parameters, processes: int = 1, **kwargs):
    """
    Compute distance to drainage outlet
    for every point in space
    """

    def tile(x):
        return x[0][0]

    def arguments():

        groups = itertools.groupby(sorted(distance.items(), key=tile), key=tile)

        for t, values in groups:

            row, col = t
            distance_outlets = {(i, j): d for (t, i, j), d in values}

            yield (
                DistanceTile,
                row,
                col,
                distance_outlets,
                params,
                kwargs
            )

    def length():

        if params.tiles.none:

            tileset = config.tileset()
            return len(tileset)

        tilefile = params.tiles.filename(**kwargs)

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

# with open('Ain/GLOBAL/DEM/distance.csv', 'w') as fp:
#     for ((row, col), i, j), d in distance.items():
#         fp.write('%d,%d,%d,%d,%f\n' % (row, col, i, j, d))

# distance = dict()
# with open('Ain/GLOBAL/DEM/distance.csv') as fp:
#     for line in fp:
#         values = line.split(',')
#         row, col, i, j = [int(v) for v in values[:4]]
#         d = float(values[4])
#         distance[(row, col), i, j] = d

# import rasterio as rio
# import numpy as np
# from rasterio.warp import Resampling

# ds = rio.open(params.output.filename())
# height, width = ds.shape
# tt, ht, wt = rio.warp.aligned_target(ds.transform, height, width, 500.0)
# out = np.zeros((ht, wt), dtype='float32')
# rio.warp.reproject(rio.band(ds, 1), destination=out, dst_crs=ds.crs, dst_transform=tt, resampling=Resampling.max)

# profile = ds.profile.copy()
# profile.update(height=ht, width=wt, transform=tt, compress='deflate', driver='GTiff')
# with rio.open('Ain/GLOBAL/DEM/OUT_DISTANCE_500.tif', 'w', **profile) as dst:
#     dst.write(out, 1)
