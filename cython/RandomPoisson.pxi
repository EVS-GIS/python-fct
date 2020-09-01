# coding: utf-8

"""
Random Poisson-disc sampling procedure

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

ctypedef pair[float, float] Point

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def random_poisson(float height, float width, float distance):
    """
    2D Poisson Disc Sampler

    Generate samples from a blue noise distribution [4],
    ie. randomly but maintaining a minimal distance between samples.

    [1] Bridson (2007). Fast Poisson disk sampling in arbitrary dimensions.
        http://dl.acm.org/citation.cfm?doid=1278780.1278807

    [2] Mike Bostock Poisson Disc explanation
        https://bl.ocks.org/mbostock/dbb02448b0f93e4c82c3

    [3] Javascript implementation and demo
        https://bl.ocks.org/mbostock/19168c663618b7f07158

    [4] Blue noise definition
        https://en.wikipedia.org/wiki/Colors_of_noise#Blue_noise
    """

    cdef:

        int i, j, k, t, di, dj, max_tries = 30
        float x, y, sqdist, angle, r
        Point sample, other
        vector[Point] samples
        deque[Point] queue
        float cell_size
        int grid_height, grid_width
        unsigned int[:, :] grid
        bint farenough

    sqdist = distance**2
    cell_size = distance * sqrt(0.5)
    grid_height = <int>ceil(height / cell_size)
    grid_width = <int>ceil(width / cell_size)
    grid = np.zeros((grid_height, grid_width), dtype='uint32')

    while True:

        if samples.empty():

            x = np.random.random_sample() * width
            y = np.random.random_sample() * height
            sample = Point(x, y)
            samples.push_back(sample)
            queue.push_back(sample)
            i = <int>(y / cell_size)
            j = <int>(x / cell_size)
            grid[i, j] = samples.size()

        else:

            while not queue.empty():

                k = np.random.random_sample() * queue.size()
                sample = queue.at(k)

                for t in range(max_tries):

                    angle = 2 * M_PI * np.random.random_sample()
                    r = sqrt(np.random.sample() * 3 * sqdist + sqdist)
                    x = sample.first + r * cos(angle)
                    y = sample.second + r * sin(angle)

                    if x >= 0 and x < width and y >= 0 and y < height:

                        i = <int>(y / cell_size)
                        j = <int>(x / cell_size)

                        farenough = True

                        for di in range(-1, 2):
                            for dj in range(-1, 2):

                                if (i+di) >= 0 and (i+di) < grid_height and (j+dj) >=0 and (j+dj) < grid_width:
                                    if grid[i+di, j+dj] != 0:
                                        
                                        other = samples[grid[i+di, j+dj] - 1]
                                        
                                        if ((x - other.first)**2 + (y - other.second)**2) < sqdist:
                                            farenough = False
                                            break

                            if not farenough:
                                break

                        else:

                            sample = Point(x, y)
                            samples.push_back(sample)
                            queue.push_back(sample)
                            grid[i, j] = samples.size()

                            break

                else:

                    queue[k] = queue.back()
                    queue.pop_back()

            break

    return np.array(samples, dtype='float32')
