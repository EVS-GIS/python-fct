# -*- coding: utf-8 -*-

"""
Extract region boundary points

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

def boundary(
    unsigned char[:, :] raster,
    unsigned char interior,
    unsigned char exterior):
    """
    Extract region boundary points

    Parameters
    ----------

    raster: 2D array, dtype uint8

        raster map with interior/exterior values

    interior: uint8

        value for interior region

    exterior: uint8

        value for exterior region
        (typical value is nodata)
    """

    cdef:

        Py_ssize_t width, height
        Py_ssize_t i, j, ik, jk
        short k

    height = raster.shape[0]
    width = raster.shape[1]

    # find boundary cells

    for i in range(height):
        for j in range(width):

            if raster[i, j] == interior:
                for k in range(8):

                    ik = i + ci[k]
                    jk = j + cj[k]

                    if not ingrid(height, width, ik, jk):
                        continue

                    if raster[ik, jk] == exterior:

                        yield i, j
