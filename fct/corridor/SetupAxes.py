# coding: utf-8

"""
LandCover Lateral Continuity Analysis

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import click
import fiona
from ..config import config

def SetupAxes():
    """
    DOCME
    """

    refaxis_shapefile = config.filename('refaxis') # filename ok
    talweg_shapefile = config.filename('stream-network-draped') # filename ok
    drainage_shapefile = config.tileset().filename('streams')

    with fiona.open(refaxis_shapefile) as fs:

        with click.progressbar(fs) as iterator:
            for feature in iterator:

                axis = feature['properties']['AXIS']

                # 1. Copy reference axis

                output_refaxis = config.filename('ax_refaxis', axis=axis) # filename ok
                options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

                with fiona.open(output_refaxis, 'w', **options) as fst:
                    fst.write(feature)

                # 2. copy talweg from cartograhy

                output_talweg = config.filename('ax_talweg', axis=axis) # filename ok

                with fiona.open(talweg_shapefile) as talweg_fs:

                    options = dict(
                        driver=talweg_fs.driver,
                        crs=talweg_fs.crs,
                        schema=talweg_fs.schema)

                    with fiona.open(output_talweg, 'w', **options) as fst:
                        for talweg in talweg_fs:
                            if talweg['properties']['AXIS'] == axis:
                                fst.write(talweg)

                # 3. copy drainage network from DEM

                if not os.path.exists(drainage_shapefile):
                    click.secho('Not copying drainage network', fg='yellow')
                    click.echo('File %s does not exist' % drainage_shapefile)
                    return

                output_drainage = config.tileset().filename('ax_drainage_network', axis=axis)

                with fiona.open(drainage_shapefile) as drainage_fs:

                    options = dict(
                        driver=drainage_fs.driver,
                        crs=drainage_fs.crs,
                        schema=drainage_fs.schema)

                    with fiona.open(output_drainage, 'w', **options) as fst:
                        for drainage in drainage_fs:
                            if drainage['properties']['AXIS'] == axis:
                                fst.write(drainage)
