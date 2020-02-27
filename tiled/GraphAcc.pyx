%%cython
# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii, embedsignature=True

import cython
import click
cimport cython
from libcpp.pair cimport pair
from libcpp.deque cimport deque
from libcpp.map cimport map
from cython.operator cimport dereference, preincrement

ctypedef pair[long, long] Pixel
ctypedef pair[Pixel, long] ContributingPixel
ctypedef map[Pixel, ContributingPixel] Graph
ctypedef pair[Pixel, ContributingPixel]  GraphItem
ctypedef map[Pixel, ContributingPixel].iterator GraphIterator
ctypedef map[Pixel, int] Degree
ctypedef map[Pixel, float] CumAreas
ctypedef map[Pixel, bint] PixTracker
ctypedef deque[Pixel] PixQueue
# ctypedef deque[Pixel].iterator PixQueueIterator

@cython.wraparound(False)
@cython.boundscheck(False)
def graph_acc(dict graph_in):

    cdef:

        long tile, i, j, t, ti, tj
        long area
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

    it = graph.begin()
    while it != graph.end():
        item = dereference(it)
        if indegree.count(item.first) == 0:
            queue.push_back(item.first)
        preincrement(it)

    with click.progressbar(length=len(indegree)) as progress:
    
        while not queue.empty():

            pixel = queue.front()
            queue.pop_front()

            if seen.count(pixel) > 0:
                continue

            progress.update(1)
            seen[pixel] = True

            if graph.count(pixel) > 0:

                value = graph[pixel]
                pixel = value.first
                area = value.second
                indegree[pixel] -= 1

                if areas.count(pixel) > 0:
                    areas[pixel] += area*25e-6 # convert to km^2
                else:
                    areas[pixel] = area*25e-6 # convert to km^2

                if indegree[pixel] == 0:
                    queue.push_back(pixel)

    return areas