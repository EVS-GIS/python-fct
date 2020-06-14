# coding: utf-8

"""
Spatial Referencing :
Measure along Reference Axis, Space Discretization

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import math
import click
import numpy as np
import rasterio as rio
import fiona

import terrain_analysis as ta
import speedup

from Burn import rasterize_linestringz

def test():

    valley_bottom_rasterfile = '/media/crousson/Backup/WORK/TestAin/AIN_RELZ_FLOW_05_07.tif'
    refaxis_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_AXREF_05_07.shp'
    output_distance = '/media/crousson/Backup/WORK/TestAin/AIN_AXIS_DISTANCE_05_07.tif'
    output_measure = '/media/crousson/Backup/WORK/TestAin/AIN_AXIS_MEASURE_05_07.tif'
    output_dgo = '/media/crousson/Backup/WORK/TestAin/AIN_DGO_05_07.tif'

    with rio.open(valley_bottom_rasterfile) as ds:

        valley_bottom = ds.read(1)
        height, width = valley_bottom.shape

        distance = np.full_like(valley_bottom, ds.nodata)
        measure = np.copy(distance)
        domain = np.copy(distance)
        domain[valley_bottom != ds.nodata] = 1

        def intile(i, j):
            return all([i >= 0, i < height, j >= 0, j < width])

        with fiona.open(refaxis_shapefile) as fs:
            for feature in fs:

                m0 = feature['properties']['M0']

                coordinates = np.array([
                    (x, y, m0) for x, y in reversed(feature['geometry']['coordinates'])
                ], dtype='float32')

                coordinates[1:, 2] = m0 + np.cumsum(np.linalg.norm(
                    coordinates[1:, :] - coordinates[:-1, :],
                    axis=1))

                coordinates[:, :2] = ta.worldtopixel(coordinates[:, :2], ds.transform, gdal=False)

                for a, b in zip(coordinates[:-1], coordinates[1:]):
                    for i, j, m in rasterize_linestringz(a, b):
                        if intile(i, j):
                            distance[i, j] = 0
                            measure[i, j] = m

                # valid_pixels = np.array([intile(i, j) for i, j in pixels])
                # distance[pixels[valid_pixels, 0], pixels[valid_pixels, 1]] = 0.0
                # measure[pixels[valid_pixels, 0], pixels[valid_pixels, 1]] = coordinates[valid_pixels, 2]
                # axis[pixels[valid_pixels, 0], pixels[valid_pixels, 1]] = 1

        # ta.shortest_distance(axr, ds.nodata, startval=1, distance=distance, feedback=ta.ConsoleFeedback())
        # ta.shortest_ref(axr, ds.nodata, startval=1, fillval=0, out=measure, feedback=ta.ConsoleFeedback())

        speedup.shortest_value(domain, measure, ds.nodata, distance, 1000.0)
        dgo = np.int32(np.digitize(measure, np.linspace(444e3, 499e3, 200)))
        dgo[valley_bottom == ds.nodata] = 0

        profile = ds.profile.copy()

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        with rio.open(output_measure, 'w', **profile) as dst:
            dst.write(measure, 1)

        profile.update(nodata=0, dtype='int32')
        with rio.open(output_dgo, 'w', **profile) as dst:
            dst.write(dgo, 1)

        click.echo('DGO Range =  %d - %d' % (np.min(dgo), np.max(dgo)))

if __name__ == '__main__':
    test()
