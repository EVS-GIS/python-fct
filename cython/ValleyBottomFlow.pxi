# -*- coding: utf-8 -*-

"""
Valley Bottom Extraction Algorithms

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
def valley_bottom_flow(
        D8Flow[:, :] flow,
        float[:, :] reference,
        float[:, :] elevations,
        float elevations_nodata,
        float[:, :] distance=None,
        float max_dz=0.0,
        float max_distance=0.0):
    """
    Valley Bottom, based on Flow Direction

    Parameters
    ----------

    elevations: array-like, dtype=float32
        
        Digital elevation model (DEM) raster (ndim=2)

    flow: array-like, dtype=int8, nodata=-1 (ndim=2)
        
        D8 flow direction raster

    reference: array-like, dtype=float32, same shape as `elevations`
        
        Reference elevation raster,
        that is modified in place during processing.
        
        For each cell, the reference elevation is the elevation of the first cell
        on the stream network following flow direction
        from that cell.
        
        Reference raster must be initialized to a copy of `elevations`
        set to nodata everywhere except for reference (stream network) cells.

    elevations_nodata: float
        
        No data value in `elevations`

    distance: array-like, dtype=float32, same shape as `elevations`
        
        Optional raster of distance to reference cells

    max_dz: float
        
        Optional maximum elevation difference from reference cell.
        Set to 0 to disable elevation stop criteria.

    max_distance: float
        
        Optional maximum distance to reference cell.
        Set to 0 to disable distance stop criteria.

    References
    ----------

    TODO
    """

    cdef:

        long height, width, i, j, ik, jk
        int k
        unsigned char[:, :] visited
        cdef Cell cell
        cdef CellStack stack

    height = reference.shape[0]
    width = reference.shape[1]

    if max_distance > 0 and distance is None:
        distance = np.zeros((height, width), dtype='float32')

    visited = np.zeros((height, width), dtype='uint8')

    with nogil:

        for i in range(height):
            for j in range(width):

                if reference[i, j] != elevations_nodata:

                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if ingrid(height, width, ik, jk) and flow[ik, jk] == upward[k] and reference[ik, jk] == elevations_nodata:

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

                    if reference[ik, jk] != elevations_nodata:
                        continue
                    
                    visited[ik, jk] = True

                    if max_dz > 0:

                        if elevations[ik, jk] - reference[i, j] > max_dz:
                            continue
                    
                    if max_distance > 0:

                        if k % 2 == 0:
                            distance[ik, jk] = distance[i, j] + 1.0
                        else:
                            distance[ik, jk] = distance[i, j] + 1.4142135 # sqrt(2) float32

                        if distance[ik, jk] > max_distance:
                            continue

                    reference[ik, jk] = reference[i, j]
                    
                    cell = Cell(ik, jk)
                    stack.push(cell)
