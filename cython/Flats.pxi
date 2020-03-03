# coding: utf-8

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Cell grow_flat_region(
    float[:, :] elevations,
    float[:, :] flats,
    float nodata,
    Label[:, :] labels,
    long height,
    long width,
    Label region_label,
    CellQueue& queue):
    """
    DOCME
    """

    cdef:

        Cell c
        int count = 0
        long i, j, ix, jx, x
        float flatz = nodata, outz = nodata
        Cell outlet = Cell(-1, -1)

    while not queue.empty():

        c = queue.front()
        queue.pop_front()

        i = c.first
        j = c.second

        if flatz == nodata:
            flatz = elevations[i, j]
        elif elevations[i, j] < flatz:
            flatz = elevations[i, j]

        for x in range(8):
                    
            ix = i + ci[x]
            jx = j + cj[x]

            if not ingrid(height, width, ix, jx) or flats[ix, jx] == nodata:
                continue

            if labels[ix, jx] > 0:
                continue

            labels[ix, jx] = region_label

            if flats[ix, jx] > 0:

                queue.push_back(Cell(ix, jx))

            elif elevations[ix, jx] < flatz:

                if outz == nodata:

                    outlet = Cell(ix, jx)
                    outz = elevations[ix, jx]
                    count += 1

                elif elevations[ix, jx] > outz:

                    outlet = Cell(ix, jx)
                    outz = elevations[ix, jx]
                    count += 1

    if count > 1:
        click.echo('%d outlets for region %d' % (count, region_label))

    return outlet

def flat_labels(
        float[:, :] elevations,
        float[:, :] flats,
        float nodata,
        # float dx, float dy,
        # float minslope=1e-3,
        # float[:, :] out = None,
        Label[:, :] labels = None):

    cdef:

        long height = flats.shape[0], width = flats.shape[1]
        long i, j
        Label next_label = 1
        CellQueue pit
        vector[Cell] outlets
        Cell outlet

    if labels is None:
        labels = np.zeros((height, width), dtype=np.uint32)

    for i in range(height):
        for j in range(width):

            if flats[i, j] == nodata:
                continue

            if flats[i, j] > 0 and labels[i, j] == 0:

                label = next_label
                next_label += 1
                labels[i, j] = label

                pit.push_back(Cell(i, j))
                outlet = grow_flat_region(elevations, flats, nodata, labels, height, width, label, pit)
                outlets.push_back(outlet)

    return np.asarray(labels), outlets

def flat_boxes(Label[:, :] labels):
    """
    DOCME
    """

    cdef:

        long height = labels.shape[0], width = labels.shape[1]
        long i, j
        Label label
        map[Label, long] mini, minj, maxi, maxj, count

    for i in range(height):
        for j in range(width):

            label = labels[i, j]

            if label > 0:

                if mini.count(label) == 0:

                    mini[label] = i
                    minj[label] = j
                    maxi[label] = i
                    maxj[label] = j
                    count[label] = 1

                else:

                    mini[label] = min[long](i, mini[label])
                    minj[label] = min[long](j, minj[label])
                    maxi[label] = max[long](i, maxi[label])
                    maxj[label] = max[long](j, maxj[label])
                    count[label] += 1

    return {l: (mini[l], minj[l], maxi[l], maxj[l], count[l]) for l in dict(mini)}