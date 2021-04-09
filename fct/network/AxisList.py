import fiona
from ..config import (
    DatasetParameter,
    LiteralParameter
)

class Parameters:
    """
    Axis selection parameters
    """

    axes = DatasetParameter('reference network shapefile', type='input')
    length_min = LiteralParameter('minimum axis length')

    def __init__(self):
        """
        Default parameters values
        """

        self.axes = dict(
            key='network-cartography-ready',
            tiled=False)
        self.length_min = 50e3 # 50 km

def axis_list(params: Parameters = None):
    """
    Filter axis list by minimum axis length
    """

    if params is None:
        params = Parameters()

    length_min = params.length_min

    if length_min > 0:

        def filtr(feature):

            return feature['properties']['LENAXIS'] >= length_min

    else:

        filtr = lambda f: True

    with fiona.open(params.axes.filename()) as fs:

        return [
            (
                f['properties']['AXIS'],
                f['properties']['HACK'],
                f['properties']['LENAXIS']
            )
            for f in fs
            if filtr(f)
        ]
