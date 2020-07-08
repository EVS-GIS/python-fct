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

        Py_ssize_t height = flow.shape[0], width = flow.shape[1]
        Py_ssize_t i, j, ik, jk

        short direction
        unsigned char[:, :] inflow
        unsigned char inflowij
        CellStack stack
        Cell c
        short k

        CellSequence notflowing

        short FLOW_NODATA = -1
        short NO_FLOW = 0

    # Find No-Flow pixels

    for i in range(height):
        for j in range(width):

            if flow[i, j] == NO_FLOW and streams[i, j] > 0:

                for k in range(8):

                    ik = i + ci[k]
                    jk = j + cj[k]

                    if ingrid(height, width, ik, jk) and flow[ik, jk] == FLOW_NODATA:
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

                for k in range(8):

                    ik = i + ci[k]
                    jk = j + cj[k]

                    if ingrid(height, width, ik, jk) \
                        and streams[ik, jk] > 0 \
                        and (flow[ik, jk] == upward[k]):

                        inflowij += 1

                if inflowij != 1:
                    stack.push(Cell(i, j))

                inflow[i, j] = inflowij

    while not stack.empty():

        c = stack.top()
        stack.pop()
        i = c.first
        j = c.second

        ik = i
        jk = j
        direction = flow[ik, jk]

        while not (direction == FLOW_NODATA or direction == NO_FLOW):

            k = ilog2(direction)
            ik = ik + ci[k]
            jk = jk + cj[k]

            if (ik == i) and (jk == j):
                notflowing.push_back(Cell(ik, jk))
                break

            if ingrid(height, width, ik, jk) and inflow[ik, jk] == 1:

                direction = flow[ik, jk]

            else:

                break

    return notflowing
