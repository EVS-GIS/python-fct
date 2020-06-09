# -*- coding: utf-8 -*-

"""
Find problematic noflow cells on stream network.

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
def noflow(
    short[:, :] streams,
    short[:, :] flow):
    """
    Find problematic noflow cells on stream network.
    Returns a sequence of (row, col) pixel tuples
    """

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        long i, j, i0, j0

        short direction
        unsigned char[:, :] inflow
        unsigned char inflowij
        CellStack stack
        Cell c

        CellSequence notflowing

        short FLOW_NODATA = -1
        short NO_FLOW = 0

    # Find No-Flow pixels

    for i in range(height):
        for j in range(width):

            if flow[i, j] == NO_FLOW and streams[i, j] > 0:

                for x in range(8):

                    ix = i + ci[x]
                    jx = j + cj[x]

                    if ingrid(height, width, ix, jx) and flow[ix, jx] == FLOW_NODATA:
                        break

                else:
                
                    notflowing.push_back(Cell(i, j))

    # Find Loops (Re-entrant flow)

    inflow = np.zeros((height, width), dtype=np.uint8)

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
        i0 = c.first
        j0 = c.second

        i = i0
        j = j0
        direction = flow[i, j]

        while not (direction == FLOW_NODATA or direction == NO_FLOW):

            x = ilog2(direction)
            di, dj = ci[x], cj[x]
            i, j = i+di, j+dj

            if (i == i0) and (j == j0):
                notflowing.push_back(Cell(i0, j0))
                break

            if ingrid(height, width, i, j) and inflow[i, j] == 1:

                direction = flow[i, j]

            else:

                break

    return notflowing
