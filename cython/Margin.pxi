# coding: utf-8

"""
Valley bottom margin vs. holes detection/separation

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

def reclass_margin(
    unsigned char[:, :] raster,
    unsigned char margin,
    unsigned char exterior,
    unsigned char new_margin_value):
    """
    Valley bottom margin vs. holes detection/separation

    Parameters
    ----------

    raster: 2D array, dtype uint8

        raster map with at least 3 different regions : exterior, margin/holes, interior,
        will be modified in place

    margin: uint8

        value for margin/holes region

    exterior: uint8

        value for exterior region
        (typical value is nodata)

    new_margin_value: uint8

        assign this new value to confirmed margin cells ;
        remaining cells with initial margin value after processing
        are holes within interior region
    """

    cdef:

        Py_ssize_t width, height
        Py_ssize_t i, j, ik, jk

        Cell ij, ijk
        ShortestEntry entry
        ShortestQueue queue

        short k
        float[:, :] distance
        float dist

    height = raster.shape[0]
    width = raster.shape[1]

    distance = np.zeros((height, width), dtype='float32')

    with nogil:

        # find boundary cells

        for i in range(height):
            for j in range(width):

                if raster[i, j] == margin:
                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if not ingrid(height, width, ik, jk):
                            continue

                        if raster[ik, jk] == exterior:

                            # distance[i, j] = 0.1
                            # entry = ShortestEntry(-distance[i, j], Cell(i, j))
                            entry = ShortestEntry(0, Cell(i, j))
                            queue.push(entry)

                            break

        # expand margin region

        while not queue.empty():

            entry = queue.top()
            queue.pop()

            dist = -entry.first
            ij = entry.second
            i = ij.first
            j = ij.second

            if distance[i, j] < dist:
                continue

            distance[i, j] = dist
            raster[i, j] = new_margin_value

            for k in range(8):

                # D4 connectivity
                # if not (ci[x] == 0 or cj[x] == 0):
                #     continue

                ik = i + ci[k]
                jk = j + cj[k]

                if not ingrid(height, width, ik, jk):
                    continue

                if raster[ik, jk] == margin:

                    dist = distance[i, j] + sqrt((i - ik)**2 + (j - jk)**2)

                    if distance[ik, jk] == 0 or dist < distance[ik, jk]:

                        ijk = Cell(ik, jk)
                        entry = ShortestEntry(-dist, ijk)
                        queue.push(entry)
                        distance[ik, jk] = dist
