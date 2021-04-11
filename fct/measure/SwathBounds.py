"""
Swath bounds
"""

import click
import fiona
from shapely.geometry import asShape
# from ..network.SwathPolygons import measure_to_swath_identifier
from ..config import DatasetParameter

class Parameters:
    """
    Elevation swath profile extraction parameters
    """

    polygons = DatasetParameter(
        'swaths polygons',
        type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.polygons = 'swaths_refaxis_polygons'

        else:

            self.polygons = dict(key='ax_swaths_refaxis_polygons', axis=axis)

def SwathBounds(source=None, axis=None, **kwargs):

    if source is None:
        source = Parameters(axis).polygons

    shapefile = source.filename(tileset=None, **kwargs)
    geometries = dict()

    with fiona.open(shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                if feature['properties']['VALUE'] == 2:

                    axis = feature['properties']['AXIS']
                    measure = feature['properties']['M']
                    geometry = asShape(feature['geometry'])

                    if (axis, measure) in geometries:
                        geometries[axis, measure] = geometries[axis, measure].union(geometry)
                    else:
                        geometries[axis, measure] = geometry

    return {
        (axis, measure): geometries[axis, measure].bounds
        for axis, measure in geometries
    }
