# -*- coding: utf-8 -*-

"""
Fast Simple Raster Filters, with nodata support

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mean_filter(float[:, :] data, float nodata, int window=3, bint inplace=True):
    """
    Mean filter on a square window of size `window`.
    No-data cells are left out in the calculation process
    and returned unchanged in the output.

    Parameters
    ----------

    data: array-like, 2d, dtype=float32

        Raster data to be processed

    nodata: float

        No-data value in `data`

    window: int

        Filtering window size, must be odd

    inplace: bool

        If True, modify `data` in place

    Returns
    -------

    filtered: array-like, 2d, dtype=float32

        Filtered raster, with nodata
        
    """

    cdef:

        Py_ssize_t height, width, padded_height, padded_width
        Py_ssize_t i, j, ik, jk
        int size, count
        double value
        float[:, :] padded
        float[:, :] filtered

    height = data.shape[0]
    width = data.shape[1]
    size = (window-1) // 2

    if inplace:
        filtered = data
    else:
        filtered = np.copy(data)

    padded = np.pad(data, size, constant_values=nodata)
    padded_height = padded.shape[0]
    padded_width = padded.shape[1]

    with nogil:
        for i in range(padded_height):
            for j in range(padded_width):

                if ingrid(height, width, i-size, j-size):

                    if filtered[i-size, j-size] == nodata:
                        continue

                    value = 0.0
                    count = 0

                    for ik in range(i-size, i+size+1):
                        for jk in range(j-size, j+size+1):

                            if padded[ik, jk] != nodata:
                                value += padded[ik, jk]
                                count += 1
                
                    filtered[i-size, j-size] = value / count

    return np.asarray(filtered)

@cython.boundscheck(False)
@cython.wraparound(False)
def max_filter(float[:, :] data, float nodata, int window=3, bint inplace=True):
    """
    Maximum filter on a square window of size `window`.
    No-data cells are left out in the calculation process
    and returned unchanged in the output.

    Parameters
    ----------

    data: array-like, 2d, dtype=float32

        Raster data to be processed

    nodata: float

        No-data value in `data`

    window: int

        Filtering window size, must be odd

    inplace: bool

        If True, modify `data` in place

    Returns
    -------

    filtered: array-like, 2d, dtype=float32

        Filtered raster, with nodata
        
    """

    cdef:

        Py_ssize_t height, width, padded_height, padded_width
        Py_ssize_t i, j, ik, jk
        int size
        float value
        float[:, :] padded
        float[:, :] filtered

    height = data.shape[0]
    width = data.shape[1]
    size = (window-1) // 2

    if inplace:
        filtered = data
    else:
        filtered = np.copy(data)

    padded = np.pad(data, size, constant_values=nodata)
    padded_height = padded.shape[0]
    padded_width = padded.shape[1]

    with nogil:
        for i in range(padded_height):
            for j in range(padded_width):

                if ingrid(height, width, i-size, j-size):

                    if filtered[i-size, j-size] == nodata:
                        continue

                    value = padded[ik, jk]

                    for ik in range(i-size, i+size+1):
                        for jk in range(j-size, j+size+1):

                            if padded[ik, jk] != nodata:
                                value = max[float](padded[ik, jk], value)
                
                    filtered[i-size, j-size] = value

    return np.asarray(filtered)

@cython.boundscheck(False)
@cython.wraparound(False)
def min_filter(float[:, :] data, float nodata, int window=3, bint inplace=True):
    """
    Minimum filter on a square window of size `window`.
    No-data cells are left out in the calculation process
    and returned unchanged in the output.

    Parameters
    ----------

    data: array-like, 2d, dtype=float32

        Raster data to be processed

    nodata: float

        No-data value in `data`

    window: int

        Filtering window size, must be odd

    inplace: bool

        If True, modify `data` in place

    Returns
    -------

    filtered: array-like, 2d, dtype=float32

        Filtered raster, with nodata
        
    """

    cdef:

        Py_ssize_t height, width, padded_height, padded_width
        Py_ssize_t i, j, ik, jk
        int size
        float value
        float[:, :] padded
        float[:, :] filtered

    height = data.shape[0]
    width = data.shape[1]
    size = (window-1) // 2

    if inplace:
        filtered = data
    else:
        filtered = np.copy(data)

    padded = np.pad(data, size, constant_values=nodata)
    padded_height = padded.shape[0]
    padded_width = padded.shape[1]

    with nogil:
        for i in range(padded_height):
            for j in range(padded_width):

                if ingrid(height, width, i-size, j-size):

                    if filtered[i-size, j-size] == nodata:
                        continue

                    value = padded[ik, jk]

                    for ik in range(i-size, i+size+1):
                        for jk in range(j-size, j+size+1):

                            if padded[ik, jk] != nodata:
                                value = min[float](padded[ik, jk], value)
                
                    filtered[i-size, j-size] = value

    return np.asarray(filtered)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def median_filter(float[:, :] data, float nodata, int window=3, bint inplace=True):
    """
    Median filter on a square window of size `window` using STD C++ quickselect algorithm.
    No-data cells are left out in the calculation process
    and returned unchanged in the output.

    Parameters
    ----------

    data: array-like, 2d, dtype=float32

        Raster data to be processed

    nodata: float

        No-data value in `data`

    window: int

        Filtering window size, must be odd

    inplace: bool

        If True, modify `data` in place

    Returns
    -------

    filtered: array-like, 2d, dtype=float32

        Filtered raster, with nodata
        
    """

    cdef:

        Py_ssize_t height, width, padded_height, padded_width
        Py_ssize_t i, j, ik, jk
        int size
        vector[float] values
        float[:, :] padded
        float[:, :] filtered

    height = data.shape[0]
    width = data.shape[1]
    size = (window-1) // 2

    if inplace:
        filtered = data
    else:
        filtered = np.copy(data)

    padded = np.pad(data, size, constant_values=nodata)
    padded_height = padded.shape[0]
    padded_width = padded.shape[1]

    with nogil:
        for i in range(padded_height):
            for j in range(padded_width):

                if ingrid(height, width, i-size, j-size):

                    if filtered[i-size, j-size] == nodata:
                        continue

                    values.clear()

                    for ik in range(i-size, i+size+1):
                        for jk in range(j-size, j+size+1):

                            if padded[ik, jk] != nodata:
                                values.push_back(padded[ik, jk])
                
                    nth_element[vector[float].iterator](
                        values.begin(),
                        values.begin() + values.size()/2,
                        values.end())

                    filtered[i-size, j-size] = values[values.size()/2]

    return np.asarray(filtered)
