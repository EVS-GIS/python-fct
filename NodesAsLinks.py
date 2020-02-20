#!/usr/bin/env python
# coding: utf-8

"""
Transforme les noeuds sous-maille en ligne
pour les représenter sous forme de flèche.

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import click
import os

@click.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def NodesAsLinks(basin, zone, overwrite):
    """
    Transforme les noeuds sous-maille en ligne
    pour les représenter sous forme de flèche.
    """

    input_shp = os.path.join(basin, zone, 'GRIDEEA_OUTLETS.shp')
    output = os.path.join(basin, zone, 'GRIDEEA_LINKS.shp')

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists : %s' % output, fg='yellow')
        return

    # from qgis_helper import (
    #     app,
    #     providers,
    #     execute
    # )

    # expression = """make_line( $geometry,  make_point(  "LINKX"  ,  "LINKY" ))"""

    # result = execute(
    #     'qgis:geometrybyexpression',
    #     input=input_shp,
    #     output=output,
    #     output_geometry=1,
    #     expression=expression)

    # if 'OUTPUT' in result:
    #     click.secho('Saved to %s' % result['OUTPUT'], fg='green')

    import fiona

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

if __name__ == '__main__':
    NodesAsLinks()
