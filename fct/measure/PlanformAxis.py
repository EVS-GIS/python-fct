# -*- coding: utf-8 -*-

"""
Planform Metrics

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import math
from heapq import heappush, heappop, heapify
from functools import total_ordering
from operator import itemgetter

import numpy as np
import click
import fiona
import fiona.crs
from shapely.geometry import asShape, LineString, Point
from shapely.ops import nearest_points

from ..config import (
    DatasetParameter,
    LiteralParameter
)

coordx = itemgetter(0)
coordy = itemgetter(1)

class Parameters:
    """
    Planform axis parameters
    """

    planform = DatasetParameter('stream/talweg polyline', type='input')
    inflection_points = DatasetParameter('inflection points', type='output')
    segments = DatasetParameter('segments between inflection points', type='output')
    amplitude_stems = DatasetParameter('maximum amplitude stems', type='output')
    planform_axis = DatasetParameter('planform axis passing through inflection points', type='output')

    amplitude_min = LiteralParameter('minimum amplitude value to retain')
    distance_max = LiteralParameter('maximum interdistance between inflection points')

    def __init__(self):
        """
        Default parameter values
        """

        self.planform = 'network-cartography-ready'
        self.inflection_points = 'planform_inflection_points'
        self.segments = 'planform_segments'
        self.amplitude_stems = 'amplitude_stems'
        self.planform_axis = 'planform_axis'

        self.amplitude_min = 20.0
        self.distance_max = 2000.0

def vector_angle(a, b):
    """
    Return angle change from direction a to direction b
    """

    return (
        np.arctan2(coordy(b), coordx(b)) -
        np.arctan2(coordy(a), coordx(a))
    )

def angle_sign(a, b, c):
    """
    Angle sign between vectors AB and AC
    """

    xab = coordx(b) - coordx(a)
    yab = coordy(b) - coordy(a)
    xac = coordx(c) - coordx(a)
    yac = coordy(c) - coordy(a)
    dot = (xab * yac) - (yab * xac)

    if dot == 0:
        return 0

    if dot > 0:
        return 1

    return -1

def project_point(a, b, c):
    """ Project point C on line (A, B)

    a, b, c: QgsPointXY
    returns QgsPointXY
    """

    xab = coordx(b) - coordx(a)
    yab = coordy(b) - coordy(a)

    A = np.array([
        [-yab, xab],
        [xab, yab]
    ])

    B = np.array([
        coordx(a)*yab + coordy(a)*xab,
        coordx(c)*xab + coordy(c)*yab
    ])

    x, y = np.linalg.inv(A).dot(B)

    return x, y

def distance(a, b):
    """
    Euclidean distance between points A and B
    """

    return np.linalg.norm(b - a)

def distance_to_line(a, b, c):
    """ Euclidean distance from point C to line (A, B)

    a, b, c: QgsPointXY
    returns distance (float)
    """

    projection = project_point(a, b, c)
    return distance(projection, c)

# def qgs_vector(p0, p1):
#     return QgsVector(p1.x() - p0.x(), p1.y() - p0.y())

class Bend(object):

    def __init__(self, points, measure):
        self.points = points
        self.measure = measure

    @classmethod
    def merge(cls, bend1, bend2):

        # assert(bend1.points[-1] == bend2.points[0])
        return cls(bend1.points[:-1] + bend2.points, bend2.measure)

    @property
    def p_origin(self):
        return self.points[0]

    @property
    def p_end(self):
        return self.points[-1]

    def npoints(self):
        return len(self.points)

    def amplitude(self):

        axis = LineString([self.p_origin, self.p_end])
        amp = max([Point(p).distance(axis) for p in self.points])
        # amp = max([ distance_to_line(self.p_origin, self.p_end, p) for p in self.points ])
        return amp

    def max_amplitude_stem(self):
        """

        Returns:
        - max amplitude stem as QgsGeometry (Line with 2 points)
        - index of max amplitude point
        """
        axis = LineString([self.p_origin, self.p_end])
        max_amp = 0.0
        max_idx = 0
        stem = None

        for idx, p in enumerate(self.points):
            pt = Point(p)
            amp = pt.distance(axis)
            if amp > max_amp:
                a, b = nearest_points(axis, pt)
                stem = LineString([a, b])
                max_amp = amp
                max_idx = idx

        return stem, max_idx

    def wavelength(self):
        axis = LineString([self.p_origin, self.p_end])
        return 2 * axis.length

    def length(self):
        return LineString(self.points).length

    def sinuosity(self):
        return 2 * self.length() / self.wavelength()

    def omega_origin(self):

        axis_direction = self.p_end - self.p_origin
        p0 = self.points[0]
        p1 = self.points[1]
        return vector_angle(axis_direction, p1 - p0) * 180 / math.pi

    def omega_end(self):

        axis_direction = self.p_end - self.p_origin
        p0 = self.points[-2]
        p1 = self.points[-1]
        return vector_angle(axis_direction, p1 - p0) * 180 / math.pi

    def curvature_radius(self):
        # formula reference ?
        return self.wavelength() * pow(self.sinuosity(), 1.5) / (13 * pow(self.sinuosity() - 1, 0.5))

def clamp_angle(angle):
    """ Return angle between -180 and +180 degrees
    """

    while angle <= -180:
        angle = angle + 360
    while angle > 180:
        angle = angle - 360
    return angle

@total_ordering
class QueueEntry(object):

    def __init__(self, index):

        self.index = index
        self.priority = float('inf')
        self.previous = None
        self.next = None
        # self.interdistance = 0.0
        self.duplicate = False
        self.removed = False

    def __lt__(self, other):
        
        return self.priority < other.priority

    def __eq__(self, other):
        
        return self.priority == other.priority

    def __repr__(self):

        return (
            'QueueEntry %d previous = %s, next = %s, priority = %f, interdistance = %f' %
            (self.index, self.previous, self.next, self.priority, self.interdistance)
        )

    def axis_angle(self, inflection_points):

        if self.previous is None or self.next is None:
            return 0.0

        a = inflection_points[self.previous]
        b = inflection_points[self.index]
        c = inflection_points[self.next]
        ab = b - a
        bc = c - b

        angle = vector_angle(ab, bc)

        # return clamp_angle(math.degrees(ab.angle(bc)))
        return clamp_angle(math.degrees(angle))

    def interdistance(self, inflection_points):

        l1 = l2 = 0.0

        if self.previous:

            a = inflection_points[self.previous]
            b = inflection_points[self.index]
            l1 = np.linalg.norm(b - a)

        if self.next:

            a = inflection_points[self.index]
            b = inflection_points[self.next]
            l2 = np.linalg.norm(b - a)

        return l1 + l2

def PlanformAxis(params: Parameters, **kwargs):
    """
    Disaggregate stream polyline by inflection points,
    and compute planform metrics.
    """

    resolution = params.amplitude_min
    lmax = params.distance_max

    def sink_axis_segment(filename, **options):

        schema = {
            'geometry': 'LineString',
            'properties': [
                ('fid', 'int')
            ]
        }

        with fiona.open(filename, 'w', schema=schema, **options) as fst:
            while True:

                fid, p0, p1, feature = (yield)

                geometry = {'type': 'LineString', 'coordinates': [p0, p1]}
                properties = {
                    'fid': fid
                }

                fst.write(dict(geometry=geometry, properties=properties))

    def sink_segment(filename, **options):

        schema = {
            'geometry': 'LineString',
            'properties': [
                ('fid', 'int'),
                ('measure', 'float:7.1'),
                ('npts', 'int'),
                ('lbend', 'float:7.1'),
                ('lwave', 'float:7.1'),
                ('sinuo', 'float:5.3'),
                ('ampli', 'float:6.2'),
                ('omeg0', 'float'),
                ('omeg1', 'float')
            ]
        }

        with fiona.open(filename, 'w', schema=schema, **options) as fst:
            while True:

                fid, bend, feature = (yield)

                geometry = {'type': 'LineString', 'coordinates': bend.points}
                properties = {
                    'fid': fid,
                    'measure': bend.measure,
                    'npts': bend.npoints(),
                    'lbend': bend.length(),
                    'lwave': bend.wavelength(),
                    'sinuo': bend.sinuosity(),
                    'ampli': bend.amplitude(),
                    'omeg0': bend.omega_origin(),
                    'omeg1': bend.omega_end()
                }

                fst.write(dict(geometry=geometry, properties=properties))

    def sink_amplitude_stem(filename, **options):

        schema = {
            'geometry': 'LineString',
            'properties': [
                ('fid', 'int'),
                ('ampli', 'float:6.2')
            ]
        }

        with fiona.open(filename, 'w', schema=schema, **options) as fst:
            while True:

                fid, bend = (yield)

                stem, _ = bend.max_amplitude_stem()

                if stem is None:
                    # ProcessingLog.addToLog(ProcessingLog.LOG_INFO, str(points))
                    continue

                geometry = stem.__geo_interface__
                properties = {
                    'fid': fid,
                    'ampli': stem.length
                }

                fst.write(dict(geometry=geometry, properties=properties))

    def sink_inflection_point(filename, **options):

        schema = {
            'geometry': 'Point',
            'properties': [
                ('gid', 'int'),
                ('angle', 'float'),
                ('interdist', 'float:6.2')
            ]
        }

        with fiona.open(filename, 'w', schema=schema, **options) as fst:
            while True:

                point_id, point, angle, interdistance = (yield)
                geometry = {'type': 'Point', 'coordinates': point}
                properties = {
                    'gid': point_id,
                    'angle': angle,
                    'interdist': interdistance
                }

                fst.write(dict(geometry=geometry, properties=properties))

    # total = 100.0 / layer.featureCount() if layer.featureCount() else 0
    fid = 0
    point_id = 0
    # Total count of detected inflection points
    detected = 0
    # Total count of retained inflection points
    retained = 0

    with fiona.open(params.planform.filename(tileset=None, **kwargs)) as fs:

        options = dict(driver=fs.driver, crs=fs.crs)

        output_segment = sink_segment(
            params.segments.filename(tileset=None, **kwargs),
            **options)

        output_inflection_point = sink_inflection_point(
            params.inflection_points.filename(tileset=None, **kwargs),
            **options)

        output_axis_segment = sink_axis_segment(
            params.planform_axis.filename(tileset=None, **kwargs),
            **options)

        output_amplitude_stem = sink_amplitude_stem(
            params.amplitude_stems.filename(tileset=None, **kwargs),
            **options)

        next(output_segment)
        next(output_inflection_point)
        next(output_axis_segment)
        next(output_amplitude_stem)

        with click.progressbar(fs) as iterator:
            for feature in iterator:

                # points = feature.geometry().asPolyline()
                points = np.asarray(feature['geometry']['coordinates'])
                points_iterator = iter(points)

                a = next(points_iterator)
                b = next(points_iterator)
                current_sign = 0
                current_segment = [a]
                current_axis_direction = None

                bends = list()
                inflection_points = list()

                # write_inflection_point(point_id, a)
                point_id = point_id + 1
                measure = 0.0

                for c in points_iterator:

                    sign = angle_sign(a, b, c)

                    if current_sign * sign < 0:

                        p0 = current_segment[0]
                        # pi = QgsPointXY(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()))
                        pk = 0.5 * (a + b)
                        current_segment.append(pk)
                        # measure += qgs_vector(p0, pk).length()
                        measure += distance(p0, pk)

                        if current_axis_direction is not None:
                            angle = vector_angle(current_axis_direction, pk - p0) * 180 / np.pi
                        else:
                            angle = 0.0

                        # write_axis_segment(fid, p0, pi, feature, angle)
                        # write_segment(fid, current_segment, feature)
                        # write_inflection_point(point_id, pi)

                        bend = Bend(current_segment, measure)
                        bends.append(bend)
                        inflection_points.append(p0)

                        current_sign = sign
                        current_segment = [pk, b]
                        measure += distance(pk, b)
                        current_axis_direction = pk - p0
                        fid = fid + 1
                        point_id = point_id + 1

                    else:

                        current_segment.append(b)
                        measure += distance(a, b)

                    if current_sign == 0:
                        current_sign = sign

                    a, b = b, c

                p0 = current_segment[0]

                if current_axis_direction is not None:
                    angle = vector_angle(current_axis_direction, b - p0) * 180 / math.pi
                else:
                    angle = 0.0

                # write_axis_segment(fid, p0, b, feature, angle)
                current_segment.append(b)
                measure += distance(a, b)

                # write_segment(fid, current_segment, feature)
                # write_inflection_point(point_id, b)
                bend = Bend(current_segment, measure)
                bends.append(bend)
                inflection_points.append(p0)
                inflection_points.append(b)

                fid = fid + 1
                point_id = point_id + 1

                detected = detected + len(inflection_points)

                # ProcessingLog.addToLog(ProcessingLog.LOG_INFO, 'Inflections points = %d' % len(inflection_points))
                # ProcessingLog.addToLog(ProcessingLog.LOG_INFO, 'Bends = %d' % len(bends))

                # Filter out smaller bends

                entries = list()

                entry = QueueEntry(0)
                entry.previous = None
                entry.next = 1
                entry.priority = float('inf')
                # entry.interdistance = float('inf')
                entries.append(entry)

                for k in range(1, len(inflection_points)-1):

                    entry = QueueEntry(k)
                    entry.previous = k-1
                    entry.next = k+1
                    entry.priority = bends[k-1].amplitude() + bends[k].amplitude()
                    # entry.interdistance = qgs_vector(inflection_points[k-1], inflection_points[k]).length() + \
                    #                       qgs_vector(inflection_points[k], inflection_points[k+1]).length()
                    entries.append(entry)

                k = len(inflection_points) - 1
                entry = QueueEntry(k)
                entry.previous = k-1
                entry.next = None
                entry.priority = float('inf')
                # entry.interdistance = float('inf')
                entries.append(entry)

                queue = list(entries)
                heapify(queue)

                while queue:

                    entry = heappop(queue)
                    k = entry.index

                    # ProcessingLog.addToLog(ProcessingLog.LOG_INFO, 'Entry = %s' % entry)

                    if entry.priority > 2*resolution:
                        break

                    if entry.duplicate or entry.removed:
                        continue

                    if entry.interdistance(inflection_points) > lmax:
                        continue

                    previous_entry = entries[entry.previous]
                    next_entry = entries[entry.next]

                    if previous_entry.previous is None:
                        continue

                    if next_entry.next is None:
                        continue

                    new_entry = QueueEntry(k)

                    entries[previous_entry.previous].next = k
                    new_entry.previous = previous_entry.previous

                    entries[next_entry.next].previous = k
                    new_entry.next = next_entry.next

                    before_bend = Bend.merge(bends[new_entry.previous], bends[entry.previous])
                    after_bend = Bend.merge(bends[k], bends[entry.next])

                    bends[new_entry.previous] = before_bend
                    bends[k] = after_bend

                    new_entry.priority = before_bend.amplitude() + after_bend.amplitude()
                    # new_entry.interdistance = qgs_vector(inflection_points[new_entry.previous], inflection_points[k]).length() + \
                    #                       qgs_vector(inflection_points[k], inflection_points[new_entry.next]).length()

                    heappush(queue, new_entry)

                    entries[k] = new_entry
                    previous_entry.removed = True
                    next_entry.removed = True
                    entry.duplicate = True

                # Output results

                index = 0

                while True:

                    entry = entries[index]
                    point = inflection_points[index]

                    if entry.next is None:

                        point_id = point_id + 1
                        angle = entry.axis_angle(inflection_points)
                        dist = entry.interdistance(inflection_points)
                        output_inflection_point.send((point_id, point, angle, dist))
                        retained = retained + 1
                        break

                    bend = bends[index]
                    point_id = point_id + 1
                    fid = fid + 1

                    # ProcessingLog.addToLog(ProcessingLog.LOG_INFO, 'Points = %s' % bend.points)
                    angle = entry.axis_angle(inflection_points)
                    dist = entry.interdistance(inflection_points)
                    output_inflection_point.send((point_id, point, angle, dist))
                    retained = retained + 1
                    output_axis_segment.send((fid, bend.p_origin, bend.p_end, feature))
                    output_segment.send((fid, bend, feature))
                    output_amplitude_stem.send((fid, bend))

                    index = entry.next

    output_inflection_point.close()
    output_segment.close()
    output_axis_segment.close()
    output_amplitude_stem.close()
