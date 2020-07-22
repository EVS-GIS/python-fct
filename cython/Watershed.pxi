# -*- coding: utf-8 -*-

"""
Watershed Analysis. Fast Cython procedure.

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
def watershed(short[:, :] flow, A[:, :] values, A fill_value=0):
    """
    Watershed analysis

    Fills no-data cells in `values`
    by propagating data values in the inverse (ie. upstream) flow direction
    given by `flow`.
    
    Raster `values` will be modified in place.
    
    In typical usage,
    `values` is the Strahler order for stream cells and no data elsewhere,
    and the result is a raster map of watersheds,
    identified by their Strahler order.

    Parameters
    ----------

    flow: array-like, dtype=int8, nodata=-1 (ndim=2)
        D8 flow direction raster

    values: array-like, dtype=float32, same shape as `flow`
        Values to propagate upstream

    fill_value: float
        Update only cells in `values` having value equal to `fill_value`.
        Other cells are left unchanged.
    """

    cdef:

        long height, width, i, j, ik, jk
        int k
        unsigned char[:, :] visited
        Cell cell
        CellStack stack

    height = flow.shape[0]
    width = flow.shape[1]

    visited = np.zeros((height, width), dtype=np.uint8)

    with nogil:

        for i in range(height):
            for j in range(width):

                if values[i, j] != fill_value:

                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if ingrid(height, width, ik, jk) and flow[ik, jk] == upward[k] and values[ik, jk] == fill_value:

                            cell = Cell(i, j)
                            stack.push(cell)
                            visited[i, j] = True

                            break

        while not stack.empty():

            cell = stack.top()
            stack.pop()

            i = cell.first
            j = cell.second

            for k in range(8):

                ik = i + ci[k]
                jk = j + cj[k]

                if ingrid(height, width, ik, jk) and flow[ik, jk] == upward[k] and not visited[ik, jk]:

                    if values[ik, jk] != fill_value:
                        continue
                    
                    values[ik, jk] = values[i, j]
                    cell = Cell(ik, jk)
                    stack.push(cell)
                    visited[ik, jk] = True
