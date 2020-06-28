# coding: utf-8

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
