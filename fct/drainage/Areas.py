# coding: utf-8

from collections import defaultdict, Counter
import click
import rasterio as rio
from .. import speedup
from ..config import config

def workdir():
    """
    Return default working directory
    """
    return config.workdir

def tileindex(tileset='default'):
    """
    Return default tileindex
    """
    return config.tileset(tileset).tileindex

def WatershedUnitAreas(dataset='labels', coeff=25e-6, tileset='default'):
    """
    Calculate the size in km2 for each unit watershed (tile, label)

    dataset: str
        per-tile labels raster dataset name

    coeff: float
        pixel count to km2 conversion coefficient
    """

    tile_index = tileindex(tileset)
    areas = defaultdict(lambda: 0)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[row, col].gid
            label_raster = config.tileset(tileset).tilename(dataset, row=row, col=col, tileset=tileset)

            with rio.open(label_raster) as ds:
        
                labels = ds.read(1)
                this_areas = speedup.label_areas(labels)
                areas.update({(tile, w): area*coeff for w, area in this_areas.items()})

    return areas

def WatershedCumulativeAreas(directed, unitareas):
    """
    Cumulate areas according to upstream-downstream graph `directed`
    """

    areas = unitareas.copy()
    indegree = Counter()

    for watershed in directed:

        downstream, minz = directed[watershed]
        indegree[downstream] += 1

    queue = [w for w in directed if indegree[w] == 0]

    while queue:

        watershed = queue.pop(0)

        if watershed not in directed:
            continue

        downstream, minz = directed[watershed]

        areas[downstream] = areas[downstream] + areas[watershed]

        indegree[downstream] -= 1

        if indegree[downstream] == 0:
            queue.append(downstream)

    return areas
