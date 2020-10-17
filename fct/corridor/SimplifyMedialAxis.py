from heapq import heappop, heappush, heapify
from functools import total_ordering
import numpy as np
import rasterio as rio
import fiona
from ..config import config

def triangle_area(a, b, c):

    return abs((a.x - c.x) * (b.y - a.y) - (a.x - b.x) * (c.y - a.y)) / 2

class Point(object):

    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        # do not store z, we don't need it
        # self.z = z

    def __repr__(self):
        return '(%.1f, %.1f)' % (self.x, self.y)

class Triangle(object):

    def __init__(self, index, a, b, c):
        self.a = Point(*a)
        self.b = Point(*b)
        self.c = Point(*c)
        # Store original b for later output
        self.bo = b
        self.weight = self.area
        self.entry = None
        self.index = index
        self.previous = index - 1
        self.next = index + 1

    @property
    def area(self):
        return triangle_area(self.a, self.b, self.c)

    def __repr__(self):
        return 'A%s B%s C%s' % (self.a, self.b, self.c)

def middle(a, b):

    return Point(0.5*(a.x + b.x), 0.5*(a.y + b.y))

@total_ordering
class QueueEntry(object):

    def __init__(self, triangle):
        self.triangle = triangle
        self.removed = False
        triangle.entry = self

    def __lt__(self, other):
        return self.triangle.weight < other.triangle.weight

    def __eq__(self, other):
        return self.triangle.weight == other.triangle.weight

def simplify(linestring):
    """
    Visvalingam's non-destructive line simplification
    [1] Visvalingam, M. and J. D. Whyatt. (1992)
        Line Generalisation By Repeated Elimination of Smallest Area.
        Cartographic Information Systems Research Group, University of Hull.
    [2] https://bost.ocks.org/mike/simplify/
    [3] https://github.com/topojson/topojson-simplify/blob/9c893b2/src/presimplify.js
        BSD-3 Licensed
    """

    if len(linestring) <= 2:

        return zip(linestring, [float('inf')] * len(linestring))

    triangles = [
        Triangle(k, a, b, c)
        for k, (a, b, c)
        in enumerate(zip(linestring[:-2], linestring[1:-1], linestring[2:]))
    ]

    heap = [QueueEntry(t) for t in triangles]
    heapify(heap)
    max_weight = 0

    while heap:

        entry = heappop(heap)
        if entry.removed:
            continue

        triangle = entry.triangle
        weight = triangle.weight

        if weight < max_weight:
            triangle.weight = max_weight
        else:
            max_weight = weight

        if triangle.previous > 0:

            t = triangles[triangle.previous]
            t.next = triangle.next
            t.c = triangle.c
            t.weight = t.area
            t.entry.removed = True

            heappush(heap, QueueEntry(t))

        if triangle.next < len(triangles)-1:

            t = triangles[triangle.next]
            t.previous = triangle.previous
            t.a = triangle.a
            t.weight = t.area
            t.entry.removed = True

            heappush(heap, QueueEntry(t))

    start = (linestring[0], float('inf'))
    end = (linestring[-1], float('inf'))
    return [start] + [(t.bo, t.weight) for t in triangles] + [end]


def mask_simplify(linestring, mask_file):
    """
    Visvalingam's non-destructive line simplification
    [1] Visvalingam, M. and J. D. Whyatt. (1992)
        Line Generalisation By Repeated Elimination of Smallest Area.
        Cartographic Information Systems Research Group, University of Hull.
    [2] https://bost.ocks.org/mike/simplify/
    [3] https://github.com/topojson/topojson-simplify/blob/9c893b2/src/presimplify.js
        BSD-3 Licensed
    """

    if len(linestring) <= 2:

        return zip(linestring, [float('inf')] * len(linestring))

    triangles = [
        Triangle(k, a, b, c)
        for k, (a, b, c)
        in enumerate(zip(linestring[:-2], linestring[1:-1], linestring[2:]))
    ]

    heap = [QueueEntry(t) for t in triangles]
    heapify(heap)
    max_weight = 0

    def midpoints(a, b):

        dx = b.x - a.x
        dy = b.y - a.y

        for k in [0.2, .35, .5, .65, .8]:
            yield (a.x + k*dx, a.y + k*dy)

    with rio.open(mask_file) as ds:

        while heap:

            entry = heappop(heap)
            if entry.removed:
                continue

            triangle = entry.triangle
            weight = triangle.weight

            # ---------------------------------------------------
            # mask conservative modification

            # midpoints = middle(triangle.a, triangle.c)
            values = ds.sample(list(midpoints(triangle.a, triangle.c)), 1)

            for value in values:

                if value == ds.nodata:

                    triangle.weight = float('inf')
                    break

            else:

                # ---------------------------------------------------

                if weight < max_weight:
                    triangle.weight = max_weight
                else:
                    max_weight = weight

                if triangle.previous > 0:

                    t = triangles[triangle.previous]
                    t.next = triangle.next
                    t.c = triangle.c
                    t.weight = t.area
                    t.entry.removed = True

                    heappush(heap, QueueEntry(t))

                if triangle.next < len(triangles)-1:

                    t = triangles[triangle.next]
                    t.previous = triangle.previous
                    t.a = triangle.a
                    t.weight = t.area
                    t.entry.removed = True

                    heappush(heap, QueueEntry(t))

    start = (linestring[0], float('inf'))
    end = (linestring[-1], float('inf'))
    return [start] + [(t.bo, t.weight) for t in triangles] + [end]

def SimplifyMedialAxis(axis, distance, mask_file=None):
    """
    Simplify medial axis using Visvalingam & Whyatt algorithm.

    Distance is converted to the area of a triangle
    of side length equals to distance,
    and this area is used as the simplification threshold.

    If mask_file is provided,
    ensure the resulting linestrings remain
    within the mask defined region.
    """

    shapefile = config.filename('ax_medialaxis', axis=axis)
    output = config.filename('ax_medialaxis_simplified', axis=axis)

    # sqrt(3)/4 = 0.4330 ~ 0.5
    threshold = 0.5 * distance**2

    if mask_file:

        def do_simplify(linestring):
            return mask_simplify(linestring, mask_file)

    else:

        do_simplify = simplify

    with fiona.open(shapefile) as fs:

        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

        with fiona.open(output, 'w', **options) as fst:

            for feature in fs:

                linestring = np.array(feature['geometry']['coordinates'])

                coordinates = [
                    p for p, weight in do_simplify(linestring)
                    if weight >= threshold
                ]

                feature['geometry']['coordinates'] = coordinates

                fst.write(feature)
