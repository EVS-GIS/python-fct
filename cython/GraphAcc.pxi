# coding: utf-8

"""
Graph accumulation / drainage area calculation

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

@cython.wraparound(False)
@cython.boundscheck(False)
def graph_acc(dict graph_in, float coeff=25e-6):
    """
    Calculate cumulative drained areas
    for each pixel represented in the input graph ;
    the input represents the connection between outlet pixels
    and inlet pixels at tile borders.

    Parameters
    ----------

    graph_in: dict
        outlet pixel -> destination pixel + received contributing area
        (tile, i, j) -> (tile, ti, tj, area)
        pixel: triple (int, int, int) = (tile id, pixel row, pixel column)
        rows and columns reference the global dataset
        area: local contributing area in pixel count, ie.
        the area drained by the outlet pixel _within_ the tile it belongs to.

    coeff: float
        coefficient to use to convert contributing areas in pixel count
        to real world surfaces in km^2

    Returns
    -------

    areas: dict
        pixel (row, col) -> cumulative drained area in km^2
        rows and columns reference the global dataset
    """

    cdef:

        long tile, i, j, t, ti, tj
        long area, count = 0
        Graph graph
        GraphItem item
        Degree indegree
        Pixel pixel, target
        ContributingPixel value
        PixTracker seen
        CumAreas areas
        PixQueue queue
        GraphIterator it

    for tile, i, j in graph_in:
        
        t, ti, tj, area = graph_in[(tile, i, j)]
        pixel = Pixel(i, j)
        target = Pixel(ti, tj)
        
        graph[pixel] = ContributingPixel(target, area)

        if indegree.count(target) == 0:
            indegree[target] = 1
        else:
            indegree[target] += 1

        count += 1

    it = graph.begin()
    while it != graph.end():
        item = dereference(it)
        if indegree.count(item.first) == 0:
            queue.push_back(item.first)
        preincrement(it)

    with click.progressbar(length=count) as progress:
    
        while not queue.empty():

            pixel = queue.front()
            queue.pop_front()

            if seen.count(pixel) > 0:
                continue

            progress.update(1)
            seen[pixel] = True

            if graph.count(pixel) > 0:

                value = graph[pixel]
                target = value.first
                area = value.second
                indegree[target] -= 1

                if areas.count(target) > 0:
                    areas[target] += areas[pixel] + area*coeff # convert to km^2
                else:
                    areas[target] = areas[pixel] + area*coeff # convert to km^2

                if indegree[target] == 0:
                    queue.push_back(target)

    return areas, indegree

@cython.wraparound(False)
@cython.boundscheck(False)
def raster_acc(float[:, :] raster, dict graph_in, float[:, :] out=None, float coeff=25e-6):
    """
    Calculate cumulative values
    for each pixel represented in the input graph,
    with input values from `raster`.

    Parameters
    ----------

    raster: array-like, 2D, dtype=float32
        
        Input values to be accumulated

    graph_in: dict
        
        Graph of outlet pixel (i, j) to destination pixel (ti, tj)
        pixel: tuple (int, int) = (pixel row, pixel column)
        rows and columns reference pixel coordinates in `raster`

    coeff: float
        
        coefficient to use to convert contributing areas from raster values

    Returns
    -------

    out: array-like, 2D, dtype=float32, same shape as `raster`
        
        Cumulative values
    """

    cdef:

        long height = raster.shape[0], width = raster.shape[1]
        long i, j, ti, tj
        count = 0
        ContributingArea area
        PixelGraph graph
        PixelGraphItem item
        Degree indegree
        Pixel pixel, target
        ContributingPixel value
        PixTracker seen
        CumAreas areas
        PixQueue queue
        PixelGraphIterator it

    if out is None:
        out = np.zeros_like(raster, dtype='float32')

    for i, j in graph_in:
        
        ti, tj = graph_in[(i, j)]
        pixel = Pixel(i, j)
        target = Pixel(ti, tj)
        
        graph[pixel] =target

        if indegree.count(target) == 0:
            indegree[target] = 1
        else:
            indegree[target] += 1

        count += 1

    it = graph.begin()
    while it != graph.end():
        item = dereference(it)
        if indegree.count(item.first) == 0:
            queue.push_back(item.first)
        preincrement(it)
    
    while not queue.empty():

        pixel = queue.front()
        queue.pop_front()

        if seen.count(pixel) > 0:
            continue

        seen[pixel] = True

        if graph.count(pixel) > 0:

            target = graph[pixel]

            i = pixel.first
            j = pixel.second

            ti = target.first
            tj = target.second

            # if ti == 311 and (tj == 179 or tj == 180):
            #     print(ti, tj, out[ti, tj], out[i, j], raster[i, j])

            if ingrid(height, width, i, j) and ingrid(height, width, ti, tj):

                area = raster[i, j]
                out[ti, tj] += out[i, j] + area*coeff

            indegree[target] -= 1

            if indegree[target] == 0:
                queue.push_back(target)

    return np.asarray(out)

@cython.wraparound(False)
@cython.boundscheck(False)
def multiband_raster_acc(float[:, :, :] raster, dict graph_in, float[:, :, :] out=None, float coeff=25e-6):
    """
    Calculate cumulative values
    for each pixel represented in the input graph,
    with input values from `raster`.

    Parameters
    ----------

    raster: array-like, 2D, dtype=float32
        
        Input values to be accumulated

    graph_in: dict
        
        Graph of outlet pixel (i, j) to destination pixel (ti, tj)
        pixel: tuple (int, int) = (pixel row, pixel column)
        rows and columns reference pixel coordinates in `raster`

    coeff: float
        
        coefficient to use to convert contributing areas from raster values

    Returns
    -------

    out: array-like, 2D, dtype=float32, same shape as `raster`
        
        Cumulative values
    """

    cdef:

        int bands = raster.shape[0], k
        long height = raster.shape[1], width = raster.shape[2]
        long i, j, ti, tj
        count = 0
        ContributingArea area
        PixelGraph graph
        PixelGraphItem item
        Degree indegree
        Pixel pixel, target
        ContributingPixel value
        PixTracker seen
        CumAreas areas
        PixQueue queue
        PixelGraphIterator it

    if out is None:
        out = np.zeros_like(raster, dtype='float32')

    for i, j in graph_in:
        
        ti, tj = graph_in[(i, j)]
        pixel = Pixel(i, j)
        target = Pixel(ti, tj)
        
        graph[pixel] =target

        if indegree.count(target) == 0:
            indegree[target] = 1
        else:
            indegree[target] += 1

        count += 1

    it = graph.begin()
    while it != graph.end():
        item = dereference(it)
        if indegree.count(item.first) == 0:
            queue.push_back(item.first)
        preincrement(it)
    
    while not queue.empty():

        pixel = queue.front()
        queue.pop_front()

        if seen.count(pixel) > 0:
            continue

        seen[pixel] = True

        if graph.count(pixel) > 0:

            target = graph[pixel]

            i = pixel.first
            j = pixel.second

            ti = target.first
            tj = target.second

            if ingrid(height, width, i, j) and ingrid(height, width, ti, tj):

                for k in range(bands):

                    area = raster[k, i, j]
                    out[k, ti, tj] += out[k, i, j] + area*coeff

            indegree[target] -= 1

            if indegree[target] == 0:
                queue.push_back(target)

    return np.asarray(out)
