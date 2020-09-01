# -*- coding: utf-8 -*-

"""
Vectorize Stream Features

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def stream_to_feature(
    short[:, :] streams,
    D8Flow[:, :] flow):
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
        long i, j, current
        int x, di, dj
        D8Flow direction
        bint head

        char[:, :] inflow
        char inflowij

        CellStack stack
        Cell c
        list segment

        D8Flow FLOW_NODATA = -1
        D8Flow NO_FLOW = 0

    inflow = np.zeros((height, width), dtype=np.int8)

    for i in range(height):
        for j in range(width):

            direction = flow[i, j]

            if direction != FLOW_NODATA and streams[i, j] > 0:

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

                inflow[i, j] = inflowij

    while not stack.empty():

        c = stack.top()
        stack.pop()
        i = c.first
        j = c.second

        segment = [(j, i)]
        head = inflow[i, j] == 0
        direction = flow[i, j]

        while not (direction == FLOW_NODATA or direction == NO_FLOW):

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
