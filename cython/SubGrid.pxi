# -*- coding: utf-8 -*-

"""
Subgrid Topography

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

cdef inline bint inextent(GridExtent extent, long i, long j) nogil:

    return i >= extent.first.first and i <= extent.second.first and \
           j >= extent.first.second and j <= extent.second.second

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ContributingArea local_contributive_area(Cell pixel, short[:, :] flow, GridExtent extent):

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        long i, j, ix, jx
        Cell current
        CellStack queue
        ContributingArea lca = 0
        int x

    queue.push(pixel)

    while not queue.empty():

        current = queue.top()
        queue.pop()

        i = current.first
        j = current.second

        for x in range(8):

            ix =  i + ci[x]
            jx =  j + cj[x]

            if ingrid(height, width, ix, jx) and inextent(extent, ix, jx) and flow[ix, jx] == upward[x] :

                lca += 1
                queue.push(Cell(ix, jx))

    return lca

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Outlet subgrid_outlet(GridExtent extent, short[:, :] flow, ContributingArea[:, :] acc):
    """
    DOCME
    """

    cdef:

        
        long height = flow.shape[0], width = flow.shape[1]
        long i, j, mini, minj, maxi, maxj
        Cell pixel, other_pixel
        ZCell entry
        ZPriorityQueue queue
        ContributingArea area, other_area

    mini = extent.first.first
    minj = extent.first.second
    maxi = extent.second.first
    maxj = extent.second.second

    for i in range(mini, maxi+1):
        for j in range(minj, maxj+1):

            if not ingrid(height, width, i, j):
                continue

            pixel = Cell(i, j)

            if flow[i, j] != -1:

                area = acc[i, j]
                entry = ZCell(area, pixel)
                queue.push(entry)

    if queue.empty():
        
        pixel = Cell(-1, -1)
        area = 0
        return Outlet(pixel, area)

    entry = queue.top()
    queue.pop()

    pixel = entry.second
    area = local_contributive_area(pixel, flow, extent)

    while not queue.empty():

        entry = queue.top()
        queue.pop()

        other_pixel = entry.second
        other_area = local_contributive_area(other_pixel, flow, extent)

        if other_area > area:

            pixel = other_pixel
            area = other_area

        else:

            break

    return Outlet(pixel, area)

cdef GridExtent window_to_grid_extent(object window):

    cdef:

        long mini, minj, maxi, maxj

    mini = window.row_off
    minj = window.col_off
    maxi = mini + window.height - 1
    maxj = minj + window.width - 1

    return GridExtent(Cell(mini, minj), Cell(maxi, maxj))

@cython.boundscheck(False)
@cython.wraparound(False)
def window_outlet(object window, short[:, :] flow, ContributingArea[:, :] acc):
    """
    window: RasterIO Window object
    """

    cdef:

        GridExtent extent

    extent = window_to_grid_extent(window)

    return subgrid_outlet(extent, flow, acc)

@cython.boundscheck(False)
@cython.wraparound(False)
def region_outlet(short[:, :] flow, ContributingArea[:, :] acc):
    """
    window: RasterIO Window object
    """

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        GridExtent extent

    extent = GridExtent(Cell(0, 0), Cell(height-1, width-1))
    return subgrid_outlet(extent, flow, acc)

@cython.boundscheck(False)
@cython.wraparound(False)
def subgrid_outlets(long[:, :] bounds, short[:, :] flow, ContributingArea[:, :] acc):
    """
    Find outlets for every geometry (polygon) in `geometries`
    """

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        GridExtent extent
        list outlets
        Cell pixel
        ContributingArea area
        Outlet outlet

    outlets = list()

    for k in range(bounds.shape[0]):

        extent = GridExtent(
            Cell(bounds[k, 0], bounds[k, 1]),
            Cell(bounds[k, 2], bounds[k, 3]))

        outlet = subgrid_outlet(extent, flow, acc)
        pixel = outlet.first
        area = outlet.second

        if pixel.first != -1:
            outlets.append(outlet)

    return outlets
