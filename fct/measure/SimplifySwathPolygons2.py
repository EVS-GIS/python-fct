# coding: utf-8

"""
Simplify swath polygons

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import fiona
from rastachimp import simplify_dp, smooth_chaikin
from shapely.geometry import shape
from ..config import config, DatasetParameter, LiteralParameter

# def _simplify_dp_smooth(faces, edges, distance, iterations, keep_border=False):

#     edges_simpl = _simplify_dp_edges(faces, edges, distance, keep_border)
#     return _smooth_chaikin_edges(faces, edges_simpl, iterations, keep_border)

# def simplify_dp_smooth(features, distance, iterations, keep_border=False):
#     """
#     Simplify polygon features using the Douglas-Peucker method.
#     This op simplifies edges shared by multiple polygons in the same way. It will
#     also prevent edges from crossing each other.
#     """

#     simpl_features = apply_topological_op(
#         features, _simplify_dp_smooth, distance=distance, iterations=iterations, keep_border=keep_border
#     )

#     # remove degenerate polygons (no area)
#     f_simpl_features = []
#     for f in simpl_features:
#         # if polgygon is degenerate: This will make it empty
#         # if mulitp: It will remove degenerate member polygons
#         geom = f[0].buffer(0)
#         if geom.is_empty:
#             # degenerate
#             continue
#         f_simpl_features.append(set_geom(f, geom))

#     return f_simpl_features


class Parameters:
    """
    Simplify Swath Polygons parameters
    """
    
    swaths_polygons = DatasetParameter(
        'input swaths',
        type='input')
    dist_tolerance = LiteralParameter('simplification distance')
    smooth_iterations = LiteralParameter('number of smooth iterations')
    simplified = DatasetParameter(
        'output simplified polygons',
        type='output')
    

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.swaths_polygons = 'swaths_medialaxis_polygons'
            self.dist_tolerance = 20.0
            self.smooth_iterations = 1
            self.simplified = 'swaths_medialaxis_polygons_simplified'

        else:

            self.swaths_polygons = dict(key='ax_swaths_medialaxis_polygons', axis=axis)
            self.dist_tolerance = 20.0
            self.smooth_iterations = 1
            self.simplified = dict(key='ax_swaths_medialaxis_polygons_simplified', axis=axis)
            
            
def SimplifySwathPolygons(params):
    """
    Simplify (Douglas-Peucker) and smooth (Chaikin) swath polygons
    preserving shared boundaries

    @api    fct-swath:simplify

    @input  swaths_polygons:   ax_valley_swaths_polygons
    @param  dist_tolerance:    20.0
    @param  smooth_iterations: 3

    @output simplified: ax_swath_polygons_vb_simplified
    """

    with fiona.open(params.swaths_polygons.filename(tileset=None)) as fs:

        features = [
            (shape(f['geometry']), f['id'])
            for f in fs
            if f['properties']['VALUE'] == 2
        ]

    simplified = simplify_dp(
        features,
        params.dist_tolerance,
        keep_border=False)

    if params.smooth_iterations > 0:

        simplified = smooth_chaikin(
            simplified,
            params.smooth_iterations,
            keep_border=False)

    with fiona.open(params.swaths_polygons.filename(tileset=None)) as fs:

        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

        with fiona.open(params.simplified.filename(), 'w', **options) as dst:

            for geometry, fid in simplified:

                feature = fs.get(fid)
                feature.update(geometry=fiona.Geometry.from_dict(geometry.__geo_interface__))
                dst.write(feature)

# def SmoothSwathPolygons(
#         axis,
#         iterations,
#         polygon_shapefile='ax_swaths_polygons_simplified',
#         output_shapefile='ax_swaths_polygons_simplified'):
#     """
#     Smooth (Chaikin) swath polygons
#     preserving shared boundaries

#     @api    fct-swath:smooth

#     @input  swaths_polygons:   ax_swaths_polygons_simplified
#     @param  smooth_iterations: 3

#     @output simplified: ax_swaths_polygons_simplified
#     """

#     # polygon_shapefile = config.filename(polygons, axis=axis)
#     # output_shapefile = config.filename(output, axis=axis)

#     features = list()
#     properties = dict()

#     with fiona.open(polygon_shapefile) as fs:

#         options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

#         for feature in fs:
#             # if feature['properties']['VALUE'] == 2:

#             fid = feature['id']
#             properties[fid] = feature['properties']
#             features.append((shape(feature['geometry']), fid))

#     smoothed = smooth_chaikin(
#         features,
#         iterations,
#         keep_border=False)

#     with fiona.open(output_shapefile, 'w', **options) as dst:

#         for geometry, fid in smoothed:

#             feature = dict(
#                 geometry=geometry.__geo_interface__,
#                 properties=properties[fid])

#             dst.write(feature)
