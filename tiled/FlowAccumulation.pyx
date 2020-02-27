%%cython
# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii, embedsignature=True

import cython
import click
cimport cython

import numpy as np
cimport numpy

import array
from cpython cimport array

from libcpp.pair cimport pair
from libcpp.deque cimport deque

ctypedef float ContributingArea
ctypedef pair[int, int] Cell
ctypedef deque[Cell] CellQueue

#                                    0   1   2   3   4   5   6   7
#                                    N  NE   E  SE   S  SW   W  NW
cdef int[8] ci = array.array('i', [ -1, -1,  0,  1,  1,  1,  0, -1 ])
cdef int[8] cj = array.array('i', [  0,  1,  1,  1,  0, -1, -1, -1 ])


# upward = np.power(2, np.array([ 4,  5,  6,  7,  0,  1,  2,  3 ], dtype=np.uint8))
cdef unsigned char[8] upward = array.array('B', [ 16,  32,  64,  128,  1,  2,  4,  8 ])

cdef inline bint ingrid(long height, long width, long i, long j) nogil:

    return (i >= 0) and (i < height) and (j >= 0) and (j < width)

cdef inline int ilog2(unsigned char x) nogil:

    cdef int r = 0

    if x == 0:
        return -1

    while x != 1:
        r += 1
        x = x >> 1

    return r

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

    Flow accumulation raster, dtype=np.uint32, nodata=0
    """

    cdef:

        long width, height
        short nodata = -1, noflow = 0
        long i, j, ix, jx, count
        int x
        signed char[:, :] inflow
        signed char inflowij
        short direction
        long ncells = 0
        Cell cell
        CellQueue queue

    height = flow.shape[0]
    width = flow.shape[1]

    if out is None:
        out = np.ones((height, width), dtype=np.float32)

    inflow = np.zeros((height, width), dtype=np.int8)

    click.echo('Find source cells ...')
    # progress = click.progressbar(length=width*height)

    # with nogil:

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

        # progress.update(1)

    for i in range(height):
        for j in range(width):

            if inflow[i, j] == 0:
                queue.push_back(Cell(i, j))

    click.echo('Accumulate ...')
    # progress = click.progressbar(length=width*height)

    while not queue.empty():

        cell = queue.front()
        queue.pop_front()
        i = cell.first
        j = cell.second

        direction = flow[i, j]

        if direction == nodata and direction == noflow:
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