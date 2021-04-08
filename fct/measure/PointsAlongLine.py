import numpy as np
import click
import fiona
from shapely.geometry import asShape
from fct.config import (
    DatasetParameter,
    LiteralParameter
)

class Parameters:
    """
    Points generation parameters
    """

    lines = DatasetParameter('linestrings', type='input')
    output = DatasetParameter('points along line', type='output')
    distance = LiteralParameter('distance between points')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.distance = 10e3 # 10 km

        if axis is None:

            self.lines = dict(key='network-cartography-ready', tiled=False)
            self.output = dict(key='pk_talweg', tiled=False)

        else:

            self.lines = dict(key='ax_talweg', tiled=False, axis=axis)
            self.output = dict(key='ax_pk_talweg', tiled=False, axis=axis)

def PointsAlongLine(params: Parameters, **kwargs):
    """
    Generates regularly spaced points along input linestrings
    """

    shapefile = params.lines.filename(**kwargs)
    output = params.output.filename(**kwargs)

    with fiona.open(shapefile) as fs:

        options = dict(
            driver=fs.driver,
            crs=fs.crs)

        def open_output_shapefile(filename):
            """
            output coroutine
            """

            schema = {
                'geometry': 'Point',
                'properties': [
                    ('AXIS', 'int'),
                    ('CDENTITEHY', 'str:8'),
                    ('TOPONYME', 'str:254'),
                    ('distance', 'float')]
            }

            with fiona.open(filename, 'w', schema=schema, **options) as fst:
                while True:

                    feature = (yield)
                    fst.write(feature)

        sink = open_output_shapefile(output)
        next(sink)

        with click.progressbar(fs) as iterator:
            for feature in iterator:

                axis = feature['properties']['AXIS']
                toponyme = feature['properties']['TOPONYME']
                cdenthy = feature['properties']['CDENTITEHY']
                geometry = asShape(feature['geometry'])

                for dist in np.arange(0, geometry.length, params.distance):

                    point = geometry.interpolate(dist)
                    outfeature = {
                        'geometry': point.__geo_interface__,
                        'properties': {
                            'AXIS': axis,
                            'TOPONYME': toponyme,
                            'CDENTITEHY': cdenthy,
                            'distance': dist
                        }
                    }

                    sink.send(outfeature)

        sink.close()
