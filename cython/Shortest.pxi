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
def shortest_value(
        float[:, :] domain,
        float[:, :] reference,
        float nodata,
        float[:, :] distance=None,
        float max_distance=0.0):
    """
    Valley Bottom, based on shortest distance

    Parameters
    ----------

    domain: array-like, dtype=float32
        
        Domain mask, 1 belongs to domain, 0 is outside

    reference: array-like, dtype=float32, same shape as `elevations`
        
        Reference elevation raster,
        that is modified in place during processing.
        
        For each cell, the reference elevation is the elevation of the first cell
        on the stream network following flow direction
        from that cell.
        
        Reference raster must be initialized to a copy of `elevations`
        set to nodata everywhere except for reference (stream network) cells.

    nodata: float
        
        No data value in `reference`

    distance: array-like, dtype=float32, same shape as `elevations`
        
        Optional raster of distance to reference cells

    max_distance: float
        
        Optional maximum distance to reference cell.
        Set to 0 to disable distance stop criteria.

    References
    ----------

    TODO
    """

    cdef:

        long width, height, count = 0, recount = 0
        long i, j, k, ik, jk
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

    # if cost is None:
    #     cost = np.ones((height, width), dtype=np.float32)

    # if out is None:
    #     out = np.full((height, width), startval, dtype=np.float32)

    if distance is None:
        distance = np.zeros((height, width), dtype=np.float32)

    with nogil:

        # Sequential scan
        # Search for seed cells having data value
        # surrounded by at least one no-data cell

        for i in range(height):
            for j in range(width):

                if domain[i, j] == 1:
                    count += 1

                if reference[i, j] != nodata:

                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if ingrid(height, width, ik, jk) and reference[ik, jk] == nodata:

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
                
                if reference[i, j] == nodata:
                    reference[i, j] = reference[ik, jk]
                
                if i == ik or j == jk:
                    distance[i, j] = distance[ik, jk] + 1
                else:
                    distance[i, j] = distance[ik, jk] + 1.4142135 # sqrt(2) float32
                
                ancestors.erase(ij)
            
            seen[i, j] = 2 # settled

            # Stop criteria

            # Cannot use max dz stop criteria for shortest
            # withou changing the meaning of shortest :)

            # if max_dz > 0:

            #     if elevations[i, j] - reference[i, j] > max_dz:
            #         continue
        
            if max_distance > 0:

                if distance[i, j] > max_distance:
                    continue

            if domain[i, j] == 1:
                recount += 1
                if recount >= count:
                    break

            # Iterate over direct neighbor cells

            for k in range(8):

                # D4 connectivity

                # if not (ci[x] == 0 or cj[x] == 0):
                #     continue

                ik = i + ci[k]
                jk = j + cj[k]

                if not ingrid(height, width, ik, jk):
                    continue

                # if domain[ik, jk] == nodata:
                #     continue

                # if k % 2 == 0:
                #     d = distance[i, j] + 1
                # else:
                #     d = distance[i, j] + 1.4142135 # sqrt(2) float32

                d = distance[i, j] + sqrt(
                    (i + jitteri[i, j] - ik - jitteri[ik, jk])**2 +
                    (j + jitterj[i, j] - jk - jitterj[ik, jk])**2)

                # d = d + d*cost[ik, jk]

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
