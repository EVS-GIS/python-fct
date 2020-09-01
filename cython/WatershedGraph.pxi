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

ctypedef pair[int, Label] WatershedLabel
ctypedef pair[WatershedLabel, WatershedLabel] WatershedPair
ctypedef map[WatershedPair, float] WatershedGraph

cdef enum EdgeSide:
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

@cython.boundscheck(False)
cdef void connect(
    WatershedGraph& graph,
    WatershedLabel l1,
    float z1,
    WatershedLabel l2,
    float z2) nogil:

    cdef:
        float over_z
        # float epsilon = 0.002
        WatershedPair p

    if l1 > l2:
        l1, l2 = l2, l1

    p = WatershedPair(l1, l2)

    # if (max[float](z1, z2) - min[float](z1, z2)) < epsilon:
    #     over_z = max[float](z1, z2) + epsilon
    # else:
    #     over_z = max[float](z1, z2)

    over_z = max[float](z1, z2)

    if graph.count(p) == 0:
        graph[p] = over_z
    else:
        if over_z < graph[p]:
            graph[p] = over_z
        

@cython.boundscheck(False)
cdef void connect_exterior_edge(
    WatershedGraph& graph,
    float[:] elevations,
    float nodata,
    Label[:] labels,
    int tile_id):

    cdef:

        int k, width = elevations.shape[0]
        float z1, z2
        WatershedLabel l1, l2

    with nogil:
        for k in range(width):
            z1 = elevations[k]
            if z1 == nodata:
                continue
            l1 = WatershedLabel(tile_id, labels[k])
            l2 = WatershedLabel(-1, 1)
            z2 = nodata
            connect(graph, l1, z1, l2, z2)

@cython.boundscheck(False)
cdef void connect_edge(
    WatershedGraph& graph,
    float[:] elevations,
    float nodata,
    Label[:] labels,
    int tile_id,
    float[:] neighbor_elevations,
    Label[:] neighbor_labels,
    int neighbor_id):

    cdef:

        int k, s, width = elevations.shape[0]
        float z1, z2
        WatershedLabel l1, l2

    with nogil:
        for k in range(width):
            
            z1 = elevations[k]
            
            if z1 == nodata:
                l1 = WatershedLabel(-1, 1)
            else:
                l1 = WatershedLabel(tile_id, labels[k])
            
            for s in range(-1, 2):
                if k+s < 0 or k+s == width:
                    continue
                
                z2 = neighbor_elevations[-(k+s)-1]
                if z2 == nodata:
                    l2 = WatershedLabel(-1, 1)
                else:
                    l2 = WatershedLabel(neighbor_id, neighbor_labels[-(k+s)-1])
                
                connect(graph, l1, z1, l2, z2)

cdef void connect_corner(
    WatershedGraph& graph,
    float[:] elevations,
    float nodata,
    Label[:] labels,
    int tile_id,
    float[:] corner_elevations,
    Label[:] corner_labels,
    int corner_id):

    cdef:

        float z1, z2
        WatershedLabel l1, l2

    z1 = elevations[0]
    if z1 == nodata:
        l1 = WatershedLabel(-1, 1)
    else:
        l1 = WatershedLabel(tile_id, labels[0])
    
    z2 = corner_elevations[0]
    if z2 == nodata:
        l2 = WatershedLabel(-1, 1)
    else:
        l2 = WatershedLabel(corner_id, corner_labels[0])

    connect(graph, l1, z1, l2, z2)

def read_data(row, col, tiledatafn):
    return np.load(tiledatafn(row, col), allow_pickle=True)

def connect_tile(int row, int col, float nodata, dict tileindex, tiledatafn):
    """
    Return connection graph to neighbor tiles.
    """

    cdef:

        int tile_id, neighbor_id
        int i, j, side
        float z, z1, z2

        float[:] elevations
        Label[:] labels

        float[:] neighbor_elevations
        Label[:] neighbor_labels

        LabelPair link
        WatershedGraph graph
        WatershedLabel l1, l2
        WatershedPair p

    tile_id = tileindex[(row, col)].gid
    data = read_data(row, col, tiledatafn)

    for link, z in data['graph']:
        
        if link.first == 0:
            l1 = WatershedLabel(-1, 1)
        else:
            l1 = WatershedLabel(tile_id, link.first)
        
        l2 = WatershedLabel(tile_id, link.second)
        p = WatershedPair(l1, l2)
        graph[p] = z

    # for e in range(4):
    #     for k in range(elevations.shape[1]-1):

    #         z1 = elevations[e, k]
    #         l1 = (tile_id, labels[e, k])
    #         z2 = elevations[e, k+1]
    #         l2 = (tile_id, labels[e, k+1])
    #         if z1 == nodata or z2 == nodata:
    #             continue
    #         connect(l1, z1, l2, z2)

    for i, j, side in [
            (row-1, col, EdgeSide.TOP),
            (row, col-1, EdgeSide.LEFT)
        ]:

        if (i, j) in tileindex:

            elevations = data['z'][side]
            labels = data['labels'][side]
            neighbor_id = tileindex[(i, j)].gid
            neighbor_data = read_data(i, j, tiledatafn)
            neighbor_side = (side + 2) % 4
            neighbor_elevations = neighbor_data['z'][neighbor_side]
            neighbor_labels = neighbor_data['labels'][neighbor_side]

            connect_edge(
                graph,
                elevations, nodata, labels, tile_id,
                neighbor_elevations, neighbor_labels, neighbor_id)

        else:

            elevations = data['z'][side]
            labels = data['labels'][side]

            connect_exterior_edge(
                graph,
                elevations, nodata, labels, tile_id)

    for i, j, side in [
            (row+1, col, EdgeSide.BOTTOM),
            (row, col+1, EdgeSide.RIGHT)
        ]:

        if (i, j) not in tileindex:

            elevations = data['z'][side]
            labels = data['labels'][side]

            connect_exterior_edge(
                graph,
                elevations, nodata, labels, tile_id)
    
    if (row-1, col-1) in tileindex:

        elevations = data['z'][0]
        labels = data['labels'][0]
        neighbor_id = tileindex[(row-1, col-1)].gid
        neighbor_data = read_data(row-1, col-1, tiledatafn)
        neighbor_elevations = neighbor_data['z'][2]
        neighbor_labels = neighbor_data['labels'][2]

        connect_corner(
            graph,
            elevations, nodata, labels, tile_id,
            neighbor_elevations, neighbor_labels, neighbor_id)

    return graph