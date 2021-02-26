# coding: utf-8

"""
Medial Axis Simplification

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""


import numpy as np
import fiona

from ..simplify import simplify, mask_simplify
from ..config import (
    # config,
    LiteralParameter,
    DatasetParameter
)

class Parameters:
    """
    Medial axis simplification parameters
    """

    medialaxis = DatasetParameter(
        'valley medial axis',
        type='input')
    simplified = DatasetParameter(
        'simplified valley medial axis',
        type='output')
    mask = DatasetParameter(
        'valley bottom (raster)',
        type='input')
    simplify_distance = LiteralParameter(
        'simplification distance threshold')

    def __init__(self):
        """
        Default parameter values
        """

        self.medialaxis = 'ax_medialaxis'
        self.simplified = 'ax_medialaxis_simplified'
        self.mask = 'ax_valley_bottom_final'
        self.simplify_distance = 50.0

def SimplifyMedialAxis(axis: int, params: Parameters):
    """
    Simplify medial axis using Visvalingam & Whyatt algorithm.

    Distance is converted to the area of a triangle
    of side length equals to distance,
    and this area is used as the simplification threshold.

    If mask_file is provided,
    ensure the resulting linestrings remain
    within the mask defined region.
    """

    shapefile = params.medialaxis.filename(axis=axis)
    output = params.simplified.filename(axis=axis)

    # sqrt(3)/4 = 0.4330
    threshold = 0.4330 * params.simplify_distance**2

    if not params.mask.none:

        def do_simplify(linestring):
            return mask_simplify(linestring, params.mask.filename(axis=axis))

    else:

        do_simplify = simplify

    with fiona.open(shapefile) as fs:

        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

        with fiona.open(output, 'w', **options) as fst:

            for feature in fs:

                linestring = np.array(feature['geometry']['coordinates'])

                coordinates = [
                    p for p, weight in do_simplify(linestring)
                    if weight >= threshold
                ]

                feature['geometry']['coordinates'] = coordinates

                fst.write(feature)
