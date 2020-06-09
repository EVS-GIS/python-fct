# coding: utf-8

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_inflow(short[:,:] flow, unsigned char[:, :] inflow = None):
    """
    Calculate cell's inflow degree
    """

    cdef:

        long height = flow.shape[0], width = flow.shape[1]
        short nodata = -1, noflow = 0
        long i, j, ix, jx
        short direction

    if inflow is None:
        inflow = np.zeros((height, width), dtype=np.uint8)

    with nogil:
        for i in range(height):
            for j in range(width):

                direction = flow[i, j]

                if direction == nodata or direction == noflow:
                    continue

                x = ilog2(direction)
                ix = i + ci[x]
                jx = j + cj[x]

                if ingrid(height, width, ix, jx):
                    inflow[ix, jx] += 1

    return inflow

@cython.boundscheck(False)
@cython.wraparound(False)
def flow_accumulation(short[:,:] flow, ContributingArea[:, :] out=None):
    """ Flow accumulation from D8 flow direction.

    Parameters
    ----------

    flow: array-like
        D8 Flow direction raster (ndim=2, dtype=np.int8), nodata=-1

    out: array-like
        Same shape as flow, dtype=np.uint32, initialized to 0

    feedback: QgsProcessingFeedback-like object
        or None to disable feedback

    Returns
    -------

    Flow accumulation raster, dtype=np.float32, nodata=0
    """

    cdef:

        long width, height
        short nodata = -1, noflow = 0
        long i, j, ix, jx, count
        int x
        unsigned char[:, :] inflow
        unsigned char inflowij
        short direction
        long ncells = 0
        Cell cell
        CellQueue queue

    height = flow.shape[0]
    width = flow.shape[1]

    if out is None:
        out = np.ones((height, width), dtype=np.float32)

    inflow = calc_inflow(flow)

    for i in range(height):
        for j in range(width):

            if inflow[i, j] == 0:
                queue.push_back(Cell(i, j))

    while not queue.empty():

        cell = queue.front()
        queue.pop_front()
        i = cell.first
        j = cell.second

        direction = flow[i, j]

        if direction == nodata or direction == noflow:
            continue

        x = ilog2(direction)
        ix = i + ci[x]
        jx = j + cj[x]

        if not ingrid(height, width, ix, jx):
            continue

        out[ix, jx] = out[ix, jx] + out[i, j]
        inflow[ix, jx] -= 1

        if inflow[ix, jx] == 0:
            queue.push_back(Cell(ix, jx))

    return np.asarray(out)