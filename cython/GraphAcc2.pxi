# coding: utf-8

"""
Graph accumulation / drainage area calculation
Same as graph_acc, but with (row, col) tile coordinates
rather than index tile coordinate.

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

ctypedef pair[int, int] Tile
ctypedef pair[int, int] IPixel
ctypedef pair[Tile, IPixel] TileCoord
ctypedef pair[TileCoord, float] Contribution
ctypedef map[TileCoord, Contribution] TileGraph
ctypedef pair[TileCoord, Contribution]  TileGraphItem
ctypedef map[TileCoord, Contribution].iterator TileGraphIterator

@cython.wraparound(False)
@cython.boundscheck(False)
def graph_acc2(dict graph_in):
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

        int row, col, i, j
        int trow, tcol, ti, tj
        float area
        long count = 0
        
        TileGraph graph
        TileGraphItem item
        TileCoord pixel, target
        Contribution value

        map[TileCoord, int] indegree
        map[TileCoord, int] seen
        deque[TileCoord] queue
        TileGraphIterator it

        map[TileCoord, double] areas

    click.echo('Convert graph')

    with click.progressbar(graph_in) as iterator:
        for row, col, i, j in iterator:
            
            trow, tcol, ti, tj, area = graph_in[(row, col, i, j)]

            pixel = TileCoord(Tile(row, col), IPixel(i, j))
            target = TileCoord(Tile(trow, tcol), IPixel(ti, tj))
            
            graph[pixel] = Contribution(target, area)

            if indegree.count(target) == 0:
                indegree[target] = 1
            else:
                indegree[target] += 1

            count += 1

    click.echo('Find source nodes')

    it = graph.begin()
    while it != graph.end():
        item = dereference(it)
        if indegree.count(item.first) == 0:
            queue.push_back(item.first)
        preincrement(it)

    click.echo('Accumulate')

    with click.progressbar(length=count) as iterator:
    
        while not queue.empty():

            pixel = queue.front()
            queue.pop_front()

            if seen.count(pixel) > 0:
                continue

            iterator.update(1)
            seen[pixel] = True

            if graph.count(pixel) > 0:

                value = graph[pixel]
                target = value.first
                area = value.second
                indegree[target] -= 1

                if areas.count(target) > 0:
                    areas[target] += areas[pixel] + area
                else:
                    areas[target] = areas[pixel] + area

                if indegree[target] == 0:
                    queue.push_back(target)

    return areas
