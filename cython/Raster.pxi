# coding: utf-8

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
