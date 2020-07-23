# -*- coding: utf-8 -*-

"""
Watershed Labeling with Depression Filling

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
cdef Cell grow_flat_region(
    D8Flow[:, :] flow,
    float[:, :] elevations,
    float nodata,
    Label[:, :] labels,
    long height,
    long width,
    Label region_label,
    CellQueue& queue):
    """
    DOCME
    """

    cdef:

        Cell c
        int count = 0, direction
        long i, j, ix, jx
        short x
        Cell outlet = Cell(-1, -1)
        float z = nodata

    while not queue.empty():

        c = queue.front()
        queue.pop_front()

        i = c.first
        j = c.second

        if z == nodata:
            z = elevations[i, j]

        for x in range(8):
                    
            ix = i + ci[x]
            jx = j + cj[x]

            if not ingrid(height, width, ix, jx) or flow[ix, jx] == -1:
                continue

            if labels[ix, jx] > 0:
                continue

            if flow[ix, jx] == 0:

                labels[ix, jx] = region_label
                queue.push_back(Cell(ix, jx))

            elif elevations[ix, jx] == z:
                
                # Outlet is conceptually part of the flat,
                # put possibly belongs to multiple flats
                
                # labels[ix, jx] = region_label
                # queue.push_back(Cell(ix, jx))
                outlet = Cell(ix, jx)

    return outlet

@cython.wraparound(False)
@cython.boundscheck(False)
def flat_labels(
        D8Flow[:, :] flow,
        float[:, :] elevations,
        float nodata,
        # float dx, float dy,
        # float minslope=1e-3,
        # float[:, :] out = None,
        Label[:, :] labels = None):

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        long i, j
        Label next_label = 1
        CellQueue pit
        vector[Cell] outlets
        Cell outlet

    if labels is None:
        labels = np.zeros((height, width), dtype=np.uint32)

    for i in range(height):
        for j in range(width):

            if flow[i, j] == -1:
                continue

            if flow[i, j] == 0 and labels[i, j] == 0:

                label = next_label
                next_label += 1
                labels[i, j] = label

                pit.push_back(Cell(i, j))
                outlet = grow_flat_region(flow, elevations, nodata, labels, height, width, label, pit)
                outlets.push_back(outlet)

    return np.asarray(labels), outlets

@cython.wraparound(False)
@cython.boundscheck(False)
def flat_boxes(Label[:, :] labels):
    """
    DOCME
    """

    cdef:

        long height = labels.shape[0], width = labels.shape[1]
        long i, j
        Label label
        map[Label, long] mini, minj, maxi, maxj, count

    for i in range(height):
        for j in range(width):

            label = labels[i, j]

            if label > 0:

                if mini.count(label) == 0:

                    mini[label] = i
                    minj[label] = j
                    maxi[label] = i
                    maxj[label] = j
                    count[label] = 1

                else:

                    mini[label] = min[long](i, mini[label])
                    minj[label] = min[long](j, minj[label])
                    maxi[label] = max[long](i, maxi[label])
                    maxj[label] = max[long](j, maxj[label])
                    count[label] += 1

    return {l: (mini[l], minj[l], maxi[l], maxj[l], count[l]) for l in dict(mini)}

@cython.wraparound(False)
@cython.boundscheck(False)
def label_areas(Label[:, :] labels):
    """
    Count pixels by label
    """

    cdef:

        long height = labels.shape[0], width = labels.shape[1]
        long i, j
        Label label
        map[Label, unsigned int] areas

    with nogil:

        for i in range(height):
            for j in range(width):

                label = labels[i, j]

                if areas.count(label) > 0:
                    areas[label] += 1
                else:
                    areas[label] = 1

    return areas
    
@cython.wraparound(False)
@cython.boundscheck(False)
def borderflat_labels(
        D8Flow[:, :] flow,
        float[:, :] elevations,
        Label[:, :] labels = None):

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        long i, j, ix, jx
        float z
        short x
        Label label
        Label next_label = 1
        CellQueue queue
        Cell cell

    if labels is None:
        labels = np.zeros((height, width), dtype=np.uint32)

    for i in range(height):
        for j in range(width):

            if i > 1 and i < height-2:
                if j > 1 and j < width-2:
                    continue

            if flow[i, j] == 0:

                queue.push_back(Cell(i, j))
                z = elevations[i, j]
                label = labels[i, j]

                if label > 0:
                    continue

                # Check (i, j) is not just an outlet to nodata
                # it must have a neighbor with same z

                for x in range(8):
                    ix = i + ci[x]
                    jx = j + cj[x]
                    if ingrid(height, width, ix, jx):
                        if elevations[ix, jx] == z:
                            break
                else:
                    continue

                label = next_label
                next_label += 1

                while not queue.empty():

                    cell = queue.front()
                    queue.pop_front()

                    ix = cell.first
                    jx = cell.second

                    if not ingrid(height, width, ix, jx):
                        continue
                    if labels[ix, jx] != 0:
                        continue
                    if elevations[ix, jx] != z:
                        continue

                    labels[ix, jx] = label

                    for x in range(8):
                        queue.push_back(Cell(ix+ci[x], jx+cj[x]))

    return np.asarray(labels)

@cython.wraparound(False)
@cython.boundscheck(False)
def label_graph(
        Label[:, :] labels, 
        D8Flow[:, :] flow,
        float[:, :] elevations):

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        long i, j, ix, jx
        map[LabelPair, float] graph

        D8Flow direction
        short x
        Label label
        Label exterior = 0
        LabelPair link
        float z

    for i in range(height):
        for j in range(width):

            direction = flow[i, j]

            if direction == -1:
                # NODATA
                for x in range(8):

                    ix = i + ci[x]
                    jx = j + cj[x]
                    
                    if ingrid(height, width, ix, jx):
                    
                        label = labels[ix, jx]

                        if label == exterior:
                            continue

                        link = LabelPair(label, exterior)
                        z = elevations[ix, jx]
                    
                        if graph.count(link) > 0:
                            graph[link] = min[float](z, graph[link])
                        else:
                            graph[link] = z
                
                continue

            if direction == 0:
                # NOFLOW
                continue

            label = labels[i, j]
            z = elevations[i, j]

            x = ilog2(direction)
            ix = i + ci[x]
            jx = j + cj[x]

            if not ingrid(height, width, ix, jx):
                continue

            if labels[ix, jx] == label or labels[ix, jx] == exterior:
                continue

            link = LabelPair(label, labels[ix, jx])
            # z is always greater than elevations[ix, jx]
            # linkz = max[float](z, elevations[ix, jx])

            if graph.count(link) > 0:

                graph[link] = min[float](z, graph[link])

            else:

                graph[link] = z

    return graph

@cython.wraparound(False)
@cython.boundscheck(False)
def count_deadcells(D8Flow[:, :] flow):
    """
    Count invalid pixels in a flow direction raster.

    A pixel is invalid if :

    - it points to a pixel that points back to it (self loop)
    - it does not flow (flow = 0) but is not surrounded by at least
      one nodata pixel, indicating an unresolved depression
    """

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        long i, j, ix, jx
        D8Flow direction
        short x
        long loops = 0, noflows = 0

    with nogil:

        for i in range(height):
            for j in range(width):

                direction = flow[i, j]

                if direction == -1:
                    continue

                if direction == 0:

                    for x in range(8):
                        ix = i + ci[x]
                        jx = j + cj[x]
                        if ingrid(height, width, ix, jx):
                            if flow[ix, jx] == -1:
                                break
                    else:
                        noflows += 1

                    continue

                x = ilog2(direction)
                ix = i + ci[x]
                jx = j + cj[x]

                if ingrid(height, width, ix, jx):
                    if flow[ix, jx] == upward[x]:
                        loops += 1

    return loops, noflows

@cython.boundscheck(False)
@cython.wraparound(False)
def minimumz(Label[:, :] labels, map[Label, float] zmap, float nodata, float[:, :] out=None):
    """
    Map labels to minimum elevation
    """

    cdef:

        long height = labels.shape[0], width = labels.shape[1]
        long i, j
        Label label

    if out is None:
        out = np.zeros((height, width), dtype=np.float32)

    with nogil:

        for i in range(height):
            for j in range(width):

                label = labels[i, j]

                if label == 0:
                    out[i, j] = nodata
                elif zmap.count(label) == 0:
                    out[i, j] = nodata
                else:
                    out[i, j] = zmap[label]

    return np.asarray(out)
