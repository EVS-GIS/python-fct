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
        A[:, :] domain,
        A[:, :] values,
        A nodata,
        A reference,
        float[:, :] distance=None,
        float max_distance=0.0,
        float jitter=0.4):
    """
    Shortest path exploration of data domain from reference cells :

    - extract subdomain connected to reference cells
    - propagate refence values from 'values' on shortest path

    Parameters
    ----------

    domain: array-like, dtype=float32, same shape as `domain`

        Domain raster which should contain 3 types of cells :

        - nodata cells indicating exterior domain, not to be processed
        - reference cells, from which to start domain exploration
        - other cells, that may have any other value
          and will be modified in `values` output

    values: array-like, dtype=float32, same shape as `domain`
        
        Input raster providing reference values,
        that is modified in place during processing.

        You may provide values = domain or a copy of domain,
        in order to calculate the connected subdomain,
        that will correspond to cells with value `reference` in the output.

    nodata: float
        
        No data value in `domain`

    reference: float

        Valid data in `domain`

    distance: array-like, dtype=float32, same shape as `domain`
        
        Optional raster of distance to raster cells

    max_distance: float
        
        Optional maximum distance to raster cell.
        Set to 0 to disable distance stop criteria.

    jitter: float

        Amplitude of jitter to add to grid locations
        in order to avoid grid artefacts.
        Disable jitter with jitter = 0

    Returns
    -------

        `values` raster

    References
    ----------

    TODO
    """

    cdef:

        # long width, height, count = 0, recount = 0
        # long i, j, k, ik, jk
        Py_ssize_t width, height
        Py_ssize_t i, j, k, ik, jk
        float d

        Cell ij, ijk
        ShortestEntry entry
        ShortestQueue queue
        unsigned char[:, :] seen
        map[Cell, Cell] ancestors
        float[:, :] jitteri, jitterj

    height = values.shape[0]
    width = values.shape[1]
    seen = np.zeros((height, width), dtype=np.uint8)

    if jitter > 0:

        jitteri = np.float32(np.random.normal(0, jitter, (height, width)))
        jitterj = np.float32(np.random.normal(0, jitter, (height, width)))

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

        if jitter > 0:

            for i in range(height):
                for j in range(width):

                    if jitteri[i, j] > jitter:
                        jitteri[i, j] = jitter
                    elif jitteri[i, j] < -jitter:
                        jitteri[i, j] = -jitter

                    if jitterj[i, j] > jitter:
                        jitterj[i, j] = jitter
                    elif jitterj[i, j] < -jitter:
                        jitterj[i, j] = -jitter

        for i in range(height):
            for j in range(width):

                if domain[i, j] == reference:

                    # count += 1

                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if ingrid(height, width, ik, jk) and domain[ik, jk] != reference:

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

            if domain[i, j] == nodata:
                continue

            if seen[i, j] == 2:
                # already settled
                continue

            if distance[i, j] < d:
                continue

            if ancestors.count(ij) > 0:

                ijk = ancestors[ij]
                ik = ijk.first
                jk = ijk.second
                
                if values[i, j] != nodata:
                    values[i, j] = values[ik, jk]
                
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

            #     if elevations[i, j] - values[i, j] > max_dz:
            #         continue
        
            if max_distance > 0:

                if distance[i, j] > max_distance:
                    continue

            # if domain[i, j] != nodata:
            #     recount += 1
            #     if recount >= count:
            #         break

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

    return np.asarray(values)
