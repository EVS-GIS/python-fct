"""
Compute distance to drainage outlet
for every point in space
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def pix_distance_to_outlet(D8Flow[:, :] flow, Py_ssize_t i, Py_ssize_t j):
    """
    Compute distance from pixel (i, j) to local outlet according to flow raster

    Parameters
    ----------

    flow: array-like
        D8 Flow direction raster (ndim=2)

    i, j: int
        Coordinates of pixel (i, j)

    Returns
    -------

    ik, jk: int
        Coordinate of outlet pixel (ik, jk)
    distance: float
        Distance in pixels from pixel (i, j) to local outlet
    """

    cdef:

        Py_ssize_t height, width
        Py_ssize_t it, jt, ik, jk
        D8Flow direction
        int x, xk
        float d

    height = flow.shape[0]
    width = flow.shape[1]

    with nogil:

        d = 0.0
        it, jt = i, j
        direction = flow[it, jt]

        while direction != -1 and direction != 0:

            x = ilog2(direction)
            ik = it + ci[x]
            jk = jt + cj[x]

            if (0 <= ik < height) and (0 <= jk < width):

                # distance in pixels
                d += sqrt((ik-it)**2 + (jk-jt)**2)
                it, jt = ik, jk
                direction = flow[it, jt]

            else:

                break

    return it, jt, d

@cython.boundscheck(False)
@cython.wraparound(False)
def distance_to_outlet(D8Flow[:, :] flow, float[:, :] distance, float conversion=1.0):
    """
    Compute distance to drainage outlets

    Parameters
    ----------

    flow: array-like
        D8 Flow direction raster (ndim=2)

    distance: array-like
        Known distance of outlets ;
        pixels with distance > 0 are considered to be outlets

    conversion: float
        Optional pixel-distance to output unit conversion factor,
        defaults to 1.0 (no conversion)
    """

    cdef:

        Py_ssize_t height, width
        Py_ssize_t i, j, ik, jk
        Cell c
        CellStack stack
        unsigned char[:, :] seen
        D8Flow direction
        int x, xk
        float d

    height = flow.shape[0]
    width = flow.shape[1]
    seen = np.zeros((height, width), dtype=np.uint8)

    with nogil:

        for i in range(height):
            for j in range(width):

                if distance[i, j] > 0:

                    stack.push(Cell(i, j))
                    seen[i, j] = True

        while not stack.empty():

            c = stack.top()
            stack.pop()
            i = c.first
            j = c.second

            for x in range(8):
                
                ik = i - ci[x]
                jk = j - cj[x]

                if (0 <= ik < height) and (0 <= jk < width):

                    direction = flow[ik, jk]

                    if direction != -1 and direction != 0:

                        xk = ilog2(direction)
                        if xk == x:

                            # distance in pixels
                            d = sqrt((ik-i)**2 + (j-jk)**2)
                            distance[ik, jk] = distance[i, j] + d*conversion

                            if not seen[ik, jk]:

                                stack.push(Cell(ik, jk))
                                seen[ik, jk] = True
