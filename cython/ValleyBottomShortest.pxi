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
def valley_bottom_shortest(
        float[:, :] reference,
        float[:, :] elevations,
        float elevations_nodata,
        float[:, :] distance=None,
        float max_dz=0.0,
        float max_distance=0.0):
    """
    Valley Bottom, based on shortest distance space exploration

    Parameters
    ----------

    reference: array-like, dtype=float32, same shape as `elevations`
        
        Reference elevation raster,
        that is modified in place during processing.
        
        For each cell, the reference elevation is the elevation of
        the nearest cell on the stream network.
        
        Reference raster is initialized to a copy of `elevations`
        set to nodata everywhere except for reference (stream network) cells.

    elevations: array-like, dtype=float32
        
        Digital elevation model (DEM) raster (ndim=2)

    elevations_nodata: float
        
        No data value in `elevations`

    distance: array-like, dtype=float32, same shape as `elevations`
        
        Optional raster of distance to reference cells

    max_dz: float
        
        Optional maximum elevation difference from reference cell.
        Set to 0 to disable elevation stop criteria (default).
        Using max dz stop criteria
        changes the meaning of shortest :)
        But you can try max dz significantly greater than your target,
        and threshold your output afterwards.

    max_distance: float
        
        Optional maximum distance to reference cell.
        Set to 0 to disable distance stop criteria (default).

    References
    ----------

    TODO
    """

    cdef:

        Py_ssize_t width, height
        Py_ssize_t i, j, ik, jk
        short k
        float d

        Cell ij, ijk
        ShortestEntry entry
        ShortestQueue queue
        unsigned char[:, :] seen
        map[Cell, Cell] ancestors
        float[:, :] jitteri, jitterj

    height = reference.shape[0]
    width = reference.shape[1]
    seen = np.zeros((height, width), dtype=np.uint8)
    jitteri = np.float32(np.random.normal(0, 0.4, (height, width)))
    jitterj = np.float32(np.random.normal(0, 0.4, (height, width)))

    # if cost is None:
    #     cost = np.ones((height, width), dtype=np.float32)

    # if out is None:
    #     out = np.full((height, width), startval, dtype=np.float32)

    if distance is None:
        distance = np.zeros((height, width), dtype=np.float32)

    with nogil:

        # Clamp jitter

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
        # Search for origin cells with startvalue

        for i in range(height):
            for j in range(width):

                if reference[i, j] != elevations_nodata:

                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if ingrid(height, width, ik, jk) and reference[ik, jk] == elevations_nodata:

                            entry = ShortestEntry(-distance[i, j], Cell(i, j))
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
                
                if reference[i, j] == elevations_nodata:
                    reference[i, j] = reference[ik, jk]
                
                if i == ik or j == jk:
                    distance[i, j] = distance[ik, jk] + 1
                else:
                    distance[i, j] = distance[ik, jk] + 1.4142135 # sqrt(2) float32
                
                ancestors.erase(ij)
            
            seen[i, j] = 2 # settled

            # Stop criteria

            # Using max dz stop criteria for shortest
            # changes the meaning of shortest :)

            if max_dz > 0:

                if elevations[i, j] - reference[i, j] > max_dz:
                    continue
        
            if max_distance > 0:

                if distance[i, j] > max_distance:
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

                if elevations[ik, jk] == elevations_nodata:
                    continue

                # if k % 2 == 0:
                #     d = distance[i, j] + 1
                # else:
                #     d = distance[i, j] + 1.4142135 # sqrt(2) float32

                # d = d + d*cost[ik, jk]

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
