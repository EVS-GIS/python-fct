# coding: utf-8

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
def valley_bottom_initstate(
        float[:, :] reference,
        float nodata):
    """
    Find boundary cells in raster `reference`,
    ie. cells with data value neighbouring no-data cells.
    """

    cdef:

        Py_ssize_t width, height
        Py_ssize_t i, j, ik, jk
        short k
        unsigned char[:, :] state

    height = reference.shape[0]
    width = reference.shape[1]
    state = np.zeros((height, width), dtype=np.uint8)

    with nogil:

        # Sequential scan
        # Search for data cells neighbouring nodata cells

        for i in range(height):
            for j in range(width):

                if reference[i, j] != nodata:

                    for k in range(8):

                        ik = i + ci[k]
                        jk = j + cj[k]

                        if ingrid(height, width, ik, jk) and reference[ik, jk] == nodata:

                            state[i, j] = 1 # discovered
                            break

    return state

@cython.boundscheck(False)
@cython.wraparound(False)
def valley_bottom_shortest(
        float[:, :] elevations,
        unsigned char[:, :] state,
        float[:, :] reference=None,
        float[:, :] distance=None,
        float max_dz=0.0,
        float min_distance=0.0,
        float max_distance=0.0,
        float jitter=0.4):
    """
    Valley Bottom, based on shortest distance space exploration

    Input Parameters
    ----------------

    elevations: array-like, dtype=float32
        
        Digital elevation model (DEM) raster (ndim=2)

    Input/Output Parameters
    -----------------------

    state: array-like, same shape as `elevations`, dtype=uint8

        input:
        
            0 = space to explore
            1 = start cells
            255 = no data
        
        output:
        
            2 = resolved cells
            3 = height limit
            4 = distance limit
            255 = no data

    reference: array-like, dtype=float32, same shape as `elevations`
        
        Reference elevation raster,
        that is modified in place during processing.
        
        For each cell, the reference elevation is the elevation of
        the nearest cell on the stream network.
        
        If not provided, `reference` is copied from `elevations`
        where `state` == 1 before processing.

    distance: array-like, dtype=float32, same shape as `elevations`
        
        Optional raster of distance to reference cells,
        otherwise initialized to zeros.

    Other parameters
    ----------------

    max_dz: float
        
        Optional maximum elevation difference from reference cell.
        Set to 0 to disable elevation stop criteria (default).
        Using max dz stop criteria
        changes the meaning of shortest :)
        But you can try max dz significantly greater than your target,
        and threshold your output afterwards.

    min_distance: float

        Continue exploration until distance > min_distance
        whatever other stop criteria

    max_distance: float
        
        Optional maximum distance to reference cell.
        Set to 0 to disable distance stop criteria (default).

    jitter: float

        Amplitude of jitter to add to grid locations
        in order to avoid grid artefacts.
        Disable jitter with jitter = 0

    References
    ----------

    TODO
    """

    cdef:

        Py_ssize_t width, height
        Py_ssize_t i, j, ik, jk
        short k
        float dist

        Cell ij, ijk
        ShortestEntry entry
        ShortestQueue queue
        # unsigned char[:, :] state
        map[Cell, Cell] ancestors
        float[:, :] jitteri, jitterj
        bint copyref = False

    height = elevations.shape[0]
    width = elevations.shape[1]
    # state = np.zeros((height, width), dtype=np.uint8)

    assert reference.shape[0] == height and reference.shape[1] == width
    assert state.shape[0] == height and state.shape[1] == width
    assert distance.shape[0] == height and distance.shape[1] == width

    if jitter > 0:

        jitteri = np.float32(np.random.normal(0, jitter, (height, width)))
        jitterj = np.float32(np.random.normal(0, jitter, (height, width)))


    if reference is None:
        reference = np.zeros((height, width), dtype=np.float32)
        copyref = True

    if distance is None:
        distance = np.zeros((height, width), dtype=np.float32)

    with nogil:

        # Clamp jitter

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

        # Sequential scan
        # Search for origin cells with startvalue

        for i in range(height):
            for j in range(width):

                if state[i, j] == 255: # no-data

                    continue

                if state[i, j] == 1: # start

                    if copyref:
                        reference[i, j] = elevations[i, j]
                    
                    entry = ShortestEntry(-distance[i, j], Cell(i, j))
                    queue.push(entry)

                elif state[i, j] >= 2: # settled at previous iteration
                                       # -> discovered, and available for reiteration
                    state[i, j] = 1 

        # Dijkstra iteration

        while not queue.empty():

            entry = queue.top()
            queue.pop()

            dist = -entry.first
            ij = entry.second
            i = ij.first
            j = ij.second

            if state[i, j] >= 2: # already settled
                continue

            if distance[i, j] < dist:
                continue

            if ancestors.count(ij) > 0:

                ijk = ancestors[ij]
                ik = ijk.first
                jk = ijk.second
                
                # if state[i, j] != 255:
                reference[i, j] = reference[ik, jk]
                
                if i == ik or j == jk:
                    distance[i, j] = distance[ik, jk] + 1
                else:
                    distance[i, j] = distance[ik, jk] + 1.4142135 # sqrt(2) float32
                
                ancestors.erase(ij)
            
            state[i, j] = 2 # settled

            # Stop criteria

            dist = distance[i, j]
        
            if max_distance > 0 and dist > max_distance:
                state[i, j] = 4 # distance limit
                continue

            if dist > min_distance:

                # Using max dz stop criteria for shortest
                # changes the meaning of shortest :)

                if max_dz > 0 and not (-max_dz < elevations[i, j] - reference[i, j] <= max_dz):
                    state[i, j] = 3 # height limit
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

                if state[ik, jk] == 255:
                    continue

                if jitter > 0:

                    dist = distance[i, j] + sqrt(
                        (i + jitteri[i, j] - ik - jitteri[ik, jk])**2 +
                        (j + jitterj[i, j] - jk - jitterj[ik, jk])**2)

                else:

                    dist = distance[i, j] + sqrt((i - ik)**2 + (j - jk)**2)

                if state[ik, jk] == 0:

                    ijk = Cell(ik, jk)
                    entry = ShortestEntry(-dist, ijk)
                    queue.push(entry)
                    state[ik, jk] = 1 # discovered
                    distance[ik, jk] = dist
                    ancestors[ijk] = ij

                elif state[ik, jk] == 1:

                    if dist < distance[ik, jk]:

                        ijk = Cell(ik, jk)
                        entry = ShortestEntry(-dist, ijk)
                        queue.push(entry)
                        distance[ik, jk] = dist
                        ancestors[ijk] = ij

    return np.asarray(reference)
