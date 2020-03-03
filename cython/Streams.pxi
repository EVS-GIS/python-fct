# -*- coding: utf-8 -*-

"""
Vectorize Stream Features

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
def stream_to_feature(
    short[:, :] streams,
    short[:, :] flow):
    """
    Extract Stream Segments

    Parameters
    ----------

    streams: array-like
        Rasterized stream network, same shape as `elevations`,
        with stream cells >= 1

    flow: array-like
        D8 Flow direction raster (ndim=2)

    Returns
    -------

    List of stream segments in pixel coordinates.
    """

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        long i, j, ncells, nsegments, current
        int nodata = -1, x, di, dj
        short direction
        bint head
        float total

        char[:, :] inflow
        char inflowij

        CellStack stack
        Cell c
        list segment

    inflow = np.zeros((height, width), dtype=np.int8)

    total = 100.0 / (height*width)
    ncells = 0
    nsegments = 0

    for i in range(height):
        for j in range(width):

            direction = flow[i, j]

            if direction != nodata and streams[i, j] > 0:

                inflowij = 0

                for x in range(8):

                    di = ci[x]
                    dj = cj[x]

                    if ingrid(height, width, i+di, j+dj) \
                        and streams[i+di, j+dj] > 0 \
                        and (flow[i+di, j+dj] == upward[x]):

                        inflowij += 1

                if inflowij != 1:
                    stack.push(Cell(i, j))
                    nsegments += 1

                inflow[i, j] = inflowij
                ncells += 1

    while not stack.empty():

        c = stack.top()
        stack.pop()
        i = c.first
        j = c.second

        segment = [(j, i)]
        head = inflow[i, j] == 0
        direction = flow[i, j]

        while not (direction == -1 or direction == 0):

            x = ilog2(direction)
            di, dj = ci[x], cj[x]
            i, j = i+di, j+dj

            # segment.push_back(Cell(i, j))
            segment.append((j, i))

            if ingrid(height, width, i, j) and inflow[i, j] == 1:

                direction = flow[i, j]

            else:

                break

        yield np.array(segment), head
