# distutils: language=c
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, embedsignature=True

"""
GeoTransform:
replacement for RasterIO .index() and .xy() methods,
and vectorized transformation of array of coordinates.

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np
import cython
from affine import Affine

cimport numpy as np
cimport cython

from libc.math cimport lround

cdef struct Point:

    float x
    float y

cdef struct Pixel:

    int i
    int j

cdef struct GeoTransform:

    float origin_x
    float origin_y
    float scale_x
    float scale_y
    float shear_x
    float shear_y

# see https://github.com/cython/cython/issues/1642

cdef Point make_point(float x, float y) nogil:
    cdef Point p
    p.x = x
    p.y = y
    return p

cdef Pixel make_pixel(int i, int j) nogil:
    cdef Pixel p
    p.i = i
    p.j = j
    return p

cdef tuple xytuple(Point p):
    return p.x, p.y

cdef tuple ijtuple(Pixel p):
    return p.i, p.j

cdef GeoTransform transform_from_gdal(tuple gdal_transform):
    """
    Convert GDAL GeoTransform Tuple to internal GeoTransform
    """

    cdef GeoTransform transform

    transform.origin_x = gdal_transform[0]
    transform.origin_y = gdal_transform[3]
    transform.scale_x = gdal_transform[1]
    transform.scale_y = gdal_transform[5]
    transform.shear_x = gdal_transform[2]
    transform.shear_y = gdal_transform[4]

    return transform

cdef GeoTransform transform_from_rasterio(object rio_transform):
    """
    Convert RasterIO Affine Transform Object to internal GeoTransform
    """

    cdef GeoTransform transform

    transform.origin_x = rio_transform.c
    transform.origin_y = rio_transform.f
    transform.scale_x = rio_transform.a
    transform.scale_y = rio_transform.e
    transform.shear_x = rio_transform.d
    transform.shear_y = rio_transform.b

    return transform

cdef GeoTransform get_transform(object transform):

    if isinstance(transform, Affine):
        return transform_from_rasterio(transform)

    if isinstance(transform, tuple):
        return transform_from_gdal(transform)

    raise ValueError('Invalid transform instance')

cdef Point pixeltopoint(Pixel pixel, GeoTransform transform) nogil:
    """
    Transform raster pixel coordinates (py, px)
    into real world coordinates (x, y)
    """

    cdef float x, y

    if transform.shear_x == 0 and transform.shear_y == 0:

        x = (pixel.j + 0.5) * transform.scale_x + transform.origin_x
        y = (pixel.i + 0.5) * transform.scale_y + transform.origin_y

    else:

        # complete affine transform formula
        x = (pixel.i + 0.5) * transform.scale_x + (pixel.j + 0.5) * transform.shear_y + transform.origin_x
        y = (pixel.j + 0.5) * transform.scale_y + (pixel.i + 0.5) * transform.shear_x + transform.origin_y

    return make_point(x, y)

@cython.cdivision(True) 
cdef Pixel pointtopixel(Point p, GeoTransform transform) nogil:
    """
    Transform real world coordinates (x, y)
    into raster pixel coordinates (py, px)
    """

    cdef long i, j
    cdef float det

    if transform.shear_x == 0 and transform.shear_y == 0:

        j = lround((p.x - transform.origin_x) / transform.scale_x - 0.5)
        i = lround((p.y - transform.origin_y) / transform.scale_y - 0.5)

    else:

        # complete affine transform formula
        det = transform.scale_x*transform.scale_y - transform.shear_x*transform.shear_y
        j = lround((p.x*transform.scale_y - p.y*transform.shear_x + \
            transform.shear_x*transform.origin_y - transform.origin_x*transform.scale_y) / det - 0.5)
        i = lround((-p.x*transform.shear_y + p.y*transform.scale_x + \
             transform.origin_x*transform.shear_y - transform.scale_x*transform.origin_y) / det - 0.5)
    
    return make_pixel(i, j)

def index(float x, float y, transform):
    """
    Transform real world coordinates (x, y)
    into raster pixel coordinates (py, px)

    Parameters
    ----------

    x, y: float
        x and y real world coordinates

    transform: object
        GDAL GeoTransform or RasterIO Affine Transform Object

    Returns
    -------

    (row, col): int
        raster pixel coordinates
    """

    cdef GeoTransform gt
    gt = get_transform(transform)
    return ijtuple(pointtopixel(make_point(x, y), gt))

def xy(int row, int col, transform):
    """
    Transform raster pixel coordinates (py, px)
    into real world coordinates (x, y)

    Parameters
    ----------

    row, col: int
        raster pixel coordinates

    transform: object
        GDAL GeoTransform or RasterIO Affine Transform Object

    Returns
    -------

    (x, y): float
        x and y real world coordinates
        
    """

    cdef GeoTransform gt
    gt = get_transform(transform)
    return xytuple(pixeltopoint(make_pixel(row, col), transform))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def worldtopixel(np.float32_t[:, :] coordinates, transform):
    """
    Transform real world coordinates (x, y)
    into raster pixel coordinates (py, px)

    Parameters
    ----------

    coordinates: array, shape (n, 2), dtype=float32
        array of (x, y) coordinates

    transform: object
        GDAL GeoTransform or RasterIO Affine Transform Object

    gdal: boolean
        True if `transform` is a GDAL GeoTransform,
        False if it is a Rasterio Affine Transform 

    Returns
    -------

    Raster pixel coordinates
    as an array of shape (n, 2), dtype=int32
    """

    cdef:

        long length = coordinates.shape[0], k
        np.int32_t[:, :] pixels
        GeoTransform gt
        # Point point
        # Pixel pixel
        float x, y
        float det, a, b, c, d, e, f

    gt = get_transform(transform)

    pixels = np.zeros((length, 2), dtype=np.int32)

    with nogil:

        if gt.shear_x == 0 and gt.shear_y == 0:

            for k in range(length):

                # point =  Point(coordinates[k, 0], coordinates[k, 1])
                x = coordinates[k, 0]
                y = coordinates[k, 1]
                
                # pixel = pointtopixel(point, gt)
                pixels[k, 0] = lround((y - gt.origin_y) / gt.scale_y - 0.5)
                pixels[k, 1] = lround((x - gt.origin_x) / gt.scale_x - 0.5)

        else:

            # Compute inverse transform only once

            det = gt.scale_x*gt.scale_y - gt.shear_x*gt.shear_y
            a = -gt.shear_y / det
            b = gt.scale_x / det
            c = (gt.origin_x*gt.shear_y - gt.scale_x*gt.origin_y) / det
            d = gt.scale_y / det
            e = -gt.shear_x / det
            f = (gt.shear_x*gt.origin_y - gt.origin_x*gt.scale_y) / det

            for k in range(length):

                # point =  Point(coordinates[k, 0], coordinates[k, 1])
                x = coordinates[k, 0]
                y = coordinates[k, 1]
                
                pixels[k, 0] = lround((a*x + b*y + c) - 0.5)
                pixels[k, 1] = lround((d*x + e*y + f) - 0.5)
                
    return np.asarray(pixels)

@cython.boundscheck(False)
@cython.wraparound(False)
def pixeltoworld(np.int32_t[:, :] pixels, transform):
    """
    Transform raster pixel coordinates (py, px)
    into real world coordinates (x, y)

    Parameters
    ----------

    pixels: array, shape (n, 2), dtype=int32
        array of (row, col) raster coordinates

    transform: tuple or object
        GDAL GeoTransform or RasterIO Affine Transform Object

    Returns
    -------

    Real world coordinates
    as an array of shape (n, 2), dtype=float32
    """

    cdef:

        long length = pixels.shape[0], k
        np.float32_t[:, :] coordinates
        GeoTransform gt
        Point point
        Pixel pixel

    gt = get_transform(transform)

    coordinates = np.zeros((length, 2), dtype=np.float32)

    with nogil:

        for k in range(length):

            pixel = make_pixel(pixels[k, 0], pixels[k, 1])
            point = pixeltopoint(pixel, gt)
            coordinates[k, 0] = point.x
            coordinates[k, 1] = point.y

    return np.asarray(coordinates)
