from collections import Counter, defaultdict
import numpy as np
import click
import fiona
import rasterio as rio
from ..config import DatasetParameter

class Parameters:

    network = DatasetParameter('hydrographic network, not aggregated', type='input')
    measure = DatasetParameter('measure raster', type='input')

    def __init__(self, axis):

        self.network = dict(key='river-network', tiled=False)
        self.measure = dict(key='ax_axis_measure', axis=axis)

        if not self.measure.filename().exists():
            self.measure = 'axis_measure'

def Confluences(axis: int, params: Parameters = None):

    if params is None:
        params = Parameters(axis)

    shapefile = params.network.filename()

    points = dict()
    degree = Counter()
    priorities = defaultdict(lambda: 0.0)
    graph = dict()

    with fiona.open(shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                coordinates = feature['geometry']['coordinates']

                a = properties['NODEA']
                b = properties['NODEB']
                ax = properties['AXIS']
                lenax = properties['LENAXIS']

                degree[b] += 1

                if ax == axis:

                    graph[a] = b
                    points[a] = coordinates[0][:2]
                    points[b] = coordinates[-1][:2]

                else:

                    priorities[b] = max(priorities[b], lenax)

    sources = [node for node in graph if degree[node] == 0]
    assert(len(sources) == 1)

    def lookup_confluences():

        for source in sources:

            node = source
            while node in graph:

                node = graph[node]

                if degree[node] >= 2:

                    x, y = points[node]
                    priority = priorities[node]

                    yield (x, y, priority)

    confluences = np.array(list(lookup_confluences()), dtype='float32')

    with rio.open(params.measure.filename()) as ds:

        measures = np.array(list(ds.sample(confluences[:, :2], 1)), dtype='float32')
        measures = measures.squeeze()

    return np.column_stack([confluences, measures])

# output = 'TestAin/AXES/AX0007/REF/CONFLUENCES.shp'
# schema = {
#     'geometry': 'Point',
#     'properties': [
#         ('AXIS', 'int'),
#         ('MEASURE', 'float:8.1'),
#         ('PRIORITY', 'float:6.0')
#     ]
# }

# import fiona.crs
# crs = fiona.crs.from_epsg(2154)
# options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

# with fiona.open(output, 'w', **options) as fst:
#     for x, y, priority, measure in confluences:
#         fst.write({
#             'geometry': {'type': 'Point', 'coordinates': [x, y]},
#             'properties': {
#                 'AXIS': 7,
#                 'MEASURE': float(measure),
#                 'PRIORITY': float(priority)
#             }
#         })
