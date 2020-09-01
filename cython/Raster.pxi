# coding: utf-8

"""
Generic raster processing/aggregate algorithms

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
def count_by_value(numpy.int64_t[:, :] raster):
    """
    Returns a dict of the count of pixels for each value in `raster`
    """

    cdef:

        long height = raster.shape[0], width = raster.shape[1]
        long i, j
        numpy.int64_t value
        map[numpy.int64_t, long] counts

    for i in range(height):
        for j in range(width):

            value = raster[i, j]
            if counts.count(value) == 0:
                counts[value] = 1
            else:
                counts[value] += 1

    return counts

@cython.boundscheck(False)
@cython.wraparound(False)
def cumulate_by_id(A[:, :] values, unsigned int[:, :] ids, A nodata):
    """
    Sum up values by unique IDs
    """

    cdef:

        Py_ssize_t height = values.shape[0], width = values.shape[1]
        Py_ssize_t i, j, count
        unsigned int oid
        unsigned int[:] unique_ids
        map[unsigned int, A] cumulated

    unique_ids = np.unique(ids)
    count = unique_ids.shape[0]

    with nogil:

        for i in range(count):

            oid = unique_ids[i]
            cumulated[oid] = 0

        for i in range(height):
            for j in range(width):

                if values[i, j] != nodata:

                    oid = ids[i, j]
                    cumulated[oid] += values[i, j]

    return cumulated

@cython.boundscheck(False)
@cython.wraparound(False)
def cumulate_by_id2(A[:, :] values, unsigned int[:, :] ids1, unsigned int[:, :] ids2, A nodata):
    """
    Sum up values by unique IDs
    """

    cdef:

        Py_ssize_t height = values.shape[0], width = values.shape[1]
        Py_ssize_t i, j, count1, count2
        unsigned int oid1, oid2
        Cell oid
        unsigned int[:] unique_ids1, unique_ids2
        map[Cell, A] cumulated

    unique_ids1 = np.unique(ids1)
    count1 = unique_ids1.shape[0]

    unique_ids2 = np.unique(ids2)
    count2 = unique_ids2.shape[0]

    with nogil:

        for i in range(count1):
            
            oid1 = unique_ids1[i]

            for j in range(count2):

                oid2 = unique_ids2[j]
                oid = Cell(oid1, oid2)
                cumulated[oid] = 0

        for i in range(height):
            for j in range(width):

                if values[i, j] != nodata:

                    oid1 = ids1[i, j]
                    oid2 = ids2[i, j]
                    oid = Cell(oid1, oid2)
                    cumulated[oid] += values[i, j]

    return cumulated

@cython.wraparound(False)
@cython.boundscheck(False)
def count_by_uint8(numpy.uint8_t[:, :] raster, numpy.uint8_t nodata, int n):
    """
    Returns an array of the count of pixels for each value in `raster`
    """

    cdef:

        long height = raster.shape[0], width = raster.shape[1]
        long i, j
        numpy.uint8_t value
        numpy.int64_t[:] counts

    # last position counts nodata
    counts = np.zeros(n+1, dtype='int')

    for i in range(height):
        for j in range(width):

            value = raster[i, j]

            if value == nodata:
                counts[n] += 1
                continue

            if value < n:
                counts[value] += 1

    return np.asarray(counts)

@cython.wraparound(False)
@cython.boundscheck(False)
def spread_connected(numpy.uint8_t[:, :] raster, numpy.uint8_t value):
    """
    Spread value to connected pixels with value+1
    """

    cdef:

        long height = raster.shape[0], width = raster.shape[1]
        long i, j, ix, jx, ik, jk
        short x
        Cell cell
        CellQueue queue

    for i in range(height):
        for j in range(width):

            if raster[i, j] == (value+1):

                for x in range(0, 8, 2):

                    ix = i + ci[x]
                    jx = j + cj[x]

                    if ingrid(height, width, ix, jx):
                        if raster[ix, jx] == value:
                            queue.push_back(Cell(i, j))
                            break
                else:

                    continue

                while not queue.empty():

                    cell = queue.front()
                    queue.pop_front()
                    ik = cell.first
                    jk = cell.second

                    if raster[ik, jk] == value:
                        continue

                    raster[ik, jk] = value

                    for x in range(0, 8, 2):

                        ix = ik + ci[x]
                        jx = jk + cj[x]

                        if ingrid(height, width, ix, jx):
                            if raster[ix, jx] == (value+1):
                                queue.push_back(Cell(ix, jx))
