# -*- coding: utf-8 -*-

"""
Watershed Labeling with Depression Filling

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