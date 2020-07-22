# -*- coding: utf-8 -*-

"""
Raster Buffer

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def raster_buffer(
        float[:, :] data,
        float nodata,
        float buffer_width,
        float fill=0.0):
    """
    Expand data area on no-data by a buffer of given width (expressed in pixels).
    Perform shortest path expansion from data pixels.

    Parameters
    ----------

    data: array-like, 2d, dtype=float32
        
        Input raster,
        that will be modified in place

    nodata: float
        
        No data value in `data`

    buffer_width: float

        Buffer width in pixels

    fill: float
        
        Fill value for expanded pixels,
        obviously different from `nodata`.

    Returns
    -------

    distance: array-like, 2d, dtype=float32

        Distance raster, from existing data pixels.
        Pixels with distance <= 0 are not in the calculated buffer,
        either outside, or within the generating data area.
    """

    cdef:

        long width, height
        long i, j, k, ik, jk
        float d

        Cell ij, ijk
        ShortestEntry entry
        ShortestQueue queue
        unsigned char[:, :] seen
        map[Cell, Cell] ancestors
        float[:, :] distance
        float[:, :] jitteri, jitterj

    height = data.shape[0]
    width = data.shape[1]
    distance = np.zeros((height, width), dtype=np.float32)
    seen = np.zeros((height, width), dtype=np.uint8)

    jitteri = np.float32(np.random.normal(0, 0.4, (height, width)))
    jitterj = np.float32(np.random.normal(0, 0.4, (height, width)))

    with nogil:

        for i in range(height):
            for j in range(width):

                if jitteri[i, j] > 0.4:
                    jitteri[i, j] = 0.4
                elif jitteri[i, j] < -0.4:
                    jitteri[i, j] = -0.4

                if jitterj[i, j] > 0.4:
                    jitterj[i, j] = 0.4
                elif jitterj[i, j] < -0.4:
                    jitterj[i, j] = -0.4

        # Sequential scan
        # Search for seed cells having data value
        # surrounded by at least one no-data cell

        for i in range(height):
            for j in range(width):

                if data[i, j] != nodata:

                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if ingrid(height, width, ik, jk) and data[ik, jk] == nodata:

                            entry = ShortestEntry(0, Cell(i, j))
                            queue.push(entry)
                            seen[i, j] = 1 # discovered

                            break

        # Djiskstra iteration

        while not queue.empty():

            entry = queue.top()
            queue.pop()

            d = -entry.first
            ij = entry.second
            i = ij.first
            j = ij.second

            if seen[i, j] == 2:
                # already settled
                continue

            if distance[i, j] < d:
                continue

            if ancestors.count(ij) > 0:

                ijk = ancestors[ij]
                ik = ijk.first
                jk = ijk.second
                
                if data[i, j] == nodata:
                    data[i, j] = fill
                
                if i == ik or j == jk:
                    distance[i, j] = distance[ik, jk] + 1
                else:
                    distance[i, j] = distance[ik, jk] + 1.4142135 # sqrt(2) float32
                
                ancestors.erase(ij)
            
            seen[i, j] = 2 # settled

            # Stop criteria

            if distance[i, j] > buffer_width:
                continue

            # Iterate over direct neighbor cells

            for k in range(8):

                # D4 connectivity

                # if not (ci[x] == 0 or cj[x] == 0):
                #     continue

                ik = i + ci[k]
                jk = j + cj[k]

                if not ingrid(height, width, ik, jk):
                    continue

                if data[ik, jk] != nodata:
                    continue

                # if k % 2 == 0:
                #     d = distance[i, j] + 1
                # else:
                #     d = distance[i, j] + 1.4142135 # sqrt(2) float32

                d = distance[i, j] + sqrt(
                    (i + jitteri[i, j] - ik - jitteri[ik, jk])**2 +
                    (j + jitterj[i, j] - jk - jitterj[ik, jk])**2)

                if seen[ik, jk] == 0:

                    ijk = Cell(ik, jk)
                    entry = ShortestEntry(-d, ijk)
                    queue.push(entry)
                    seen[ik, jk] = 1 # discovered
                    distance[ik, jk] = d
                    ancestors[ijk] = ij

                elif seen[ik, jk] == 1:

                    if d < distance[ik, jk]:

                        ijk = Cell(ik, jk)
                        entry = ShortestEntry(-d, ijk)
                        queue.push(entry)
                        distance[ik, jk] = d
                        ancestors[ijk] = ij

    return np.asarray(distance)
