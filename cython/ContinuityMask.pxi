# -*- coding: utf-8 -*-

"""
Shortest Maximum

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
def continuity_mask(
        unsigned char[:, :] state,
        # unsigned char[:, :] out,
        float[:, :] distance,
        float jitter=0.4):
    """
    Assign to each input cell the maximum value on the shortest path
    to the nearest origin (reference) cell.

    Input/Output Parameters
    -----------------------

    state: array-like, ndims=2, dtype=uint8

        input:
        
            0 = space to explore
            1 = start cells
            255 = no data
        
        output:
        
            0 = unreachable cells
            2 = resolved cells
            255 = no data

    # out: array-like, same shape and type as `state`
        
    #     Maximum landcover class on the path to the nearest origin cell.
    #     `out` is modified in place,
    #     origin cells value are left unchanged,
    #     and propagated to nearest neighbors.

    distance: array-like, same shape `state`, dtype=float32
        
        Shortest distance in pixels to the nearest origin cell
        in `state`.
        Must be initialized to zeros,
        but can be reused between iterations.

    Other Parameters
    ----------------

    jitter: float

        Amplitude of jitter to add to grid locations
        in order to avoid grid artefacts.
        Disable jitter with jitter = 0
    """

    cdef:

        Py_ssize_t width, height
        Py_ssize_t i, j, ik, jk
        short k
        float dist, dx

        Cell ij, ijk
        ShortestEntry entry
        ShortestQueue queue
        # unsigned char[:, :] state
        map[Cell, Cell] ancestors
        float[:, :] jitteri, jitterj

    height = state.shape[0]
    width = state.shape[1]
    # state = np.zeros((height, width), dtype=np.uint8)

    # assert heights.shape[0] == height and heights.shape[1] == width
    # assert state.shape[0] == height and state.shape[1] == width
    # assert out.shape[0] == height and out.shape[1] == width
    assert distance.shape[0] == height and distance.shape[1] == width

    # if cost is not None:

    #     assert cost.shape[0] == height and cost.shape[1] == width
    #     withcost = True

    if jitter > 0:

        jitteri = np.float32(np.random.normal(0, jitter, (height, width)))
        jitterj = np.float32(np.random.normal(0, jitter, (height, width)))

    with nogil:

        if jitter > 0:

            # Clamp jitter

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

                if state[i, j] == 255:
                    continue

                if state[i, j] == 1:

                    # if out[i, j] == 255:
                    #     out[i, j] = 1

                    entry = ShortestEntry(-distance[i, j], Cell(i, j))
                    queue.push(entry)
                    # state[i, j] = 1 # seen
                    # distance[i, j] = 0

                elif state[i, j] > 2:

                    state[i, j] = 1
                    
        # Dijkstra iteration

        while not queue.empty():

            entry = queue.top()
            queue.pop()

            dist = -entry.first
            ij = entry.second
            i = ij.first
            j = ij.second

            if state[i, j] >= 2:
                continue

            if distance[i, j] < dist:
                continue

            if ancestors.count(ij) > 0:

                ijk = ancestors[ij]
                ik = ijk.first
                jk = ijk.second
                
                # out[i, j] = 1
                
                if i == ik or j == jk:
                    distance[i, j] = distance[ik, jk] + 1
                else:
                    distance[i, j] = distance[ik, jk] + 1.4142135623730951 # sqrt(2)
                
                ancestors.erase(ij)
            
            state[i, j] = 2 # settled

            dist = distance[i, j]

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

                    dx = sqrt(
                        (i + jitteri[i, j] - ik - jitteri[ik, jk])**2 +
                        (j + jitterj[i, j] - jk - jitterj[ik, jk])**2)

                else:

                    if i == ik or j == jk:
                        dx = 1
                    else:
                        dx = 1.4142135623730951 # sqrt(2)

                dist += dx

                if state[ik, jk] == 0:

                    ijk = Cell(ik, jk)
                    entry = ShortestEntry(-dist, ijk)
                    queue.push(entry)
                    state[ik, jk] = 1 # seen
                    distance[ik, jk] = dist
                    ancestors[ijk] = ij

                elif state[ik, jk] == 1:

                    if dist < distance[ik, jk]:

                        ijk = Cell(ik, jk)
                        entry = ShortestEntry(-dist, ijk)
                        queue.push(entry)
                        distance[ik, jk] = dist
                        ancestors[ijk] = ij
