import os
from multiprocessing import Pool

import numpy as np
import click

import rasterio as rio
from rasterio.windows import Window
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

import terrain_analysis as ta
from ransac import LinearModel, ransac
from Command import starcall

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset + 1
    width = col_end - col_offset + 1

    return Window(col_offset, row_offset, width, height)

def TileCropInvalidRegions(axis, row, col):

    invalid_shapefile = os.path.join(workdir, 'AX%03d_DGO_DISCONNECTED.shp' % axis)
    distance_raster = os.path.join(workdir, 'AX%03d_AXIS_DISTANCE_%02d_%02d.tif' % (axis, row, col))
    # distance_raster = os.path.join(workdir, 'AX%03d_NEAREST_DISTANCE_%02d_%02d.tif' % (axis, row, col))

    if not os.path.exists(distance_raster):
        return

    with rio.open(distance_raster) as ds:

        data = ds.read(1)
        transform = ds.transform
        nodata = ds.nodata
        profile = ds.profile.copy()

    with fiona.open(invalid_shapefile) as fs:

        def accept(feature):
            return all([
                feature['properties']['AXIS'] == axis,
                feature['properties']['ROW'] == row,
                feature['properties']['COL'] == col
            ])

        mask = features.rasterize(
            [f['geometry'] for f in fs if accept(f)],
            out_shape=data.shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8')

    data[mask == 1] = nodata

    with rio.open(distance_raster, 'w', **profile) as dst:
        dst.write(data, 1)

def CropInvalidRegions(axis, processes=1):

    tileindex = os.path.join(workdir, 'TILES.shp')
    kwargs = dict()
    arguments = list()

    with fiona.open(tileindex) as fs:
        for feature in fs:
            row = feature['properties']['ROW']
            col = feature['properties']['COL']
            arguments.append([TileCropInvalidRegions, axis, row, col, kwargs])

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)
        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def SwathProfile(axis, gid, bounds):
    """
    Calculate Elevation Swath Profile for Valley Unit (axis, gid)
    """

    dgo_raster = os.path.join(workdir, 'AX%03d_DGO.vrt' % axis)
    measure_raster = os.path.join(workdir, 'AX%03d_AXIS_MEASURE.vrt' % axis)
    distance_raster = os.path.join(workdir, 'AX%03d_AXIS_DISTANCE.vrt' % axis)
    # distance_raster = os.path.join(workdir, 'AX%03d_NEAREST_DISTANCE.vrt' % axis)
    # elevation_raster = '/var/local/fct/RMC/DEM_RGE5M_TILES.vrt'
    elevation_raster = '/var/local/fct/RMC/RGEALTI.tif'
    relz_raster = os.path.join(workdir, 'AX%03d_NEAREST_RELZ.vrt' % axis)

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(elevation_raster) as ds1:
            window1 = as_window(bounds, ds1.transform)
            elevations = ds1.read(1, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(measure_raster) as ds2:
            measure = ds2.read(1, window=window, boundless=True, fill_value=ds2.nodata)

        with rio.open(relz_raster) as ds3:
            relz = ds3.read(1, window=window, boundless=True, fill_value=ds3.nodata)

        with rio.open(dgo_raster) as ds4:
            mask = (ds4.read(1, window=window, boundless=True, fill_value=ds4.nodata) == gid)
            mask = mask & (elevations != ds1.nodata) & (distance != ds.nodata)

        assert(all([
            distance.shape == elevations.shape,
            measure.shape == elevations.shape,
            mask.shape == elevations.shape
        ]))

        xbins = np.arange(np.min(distance[mask]), np.max(distance[mask]), 10.0)
        binned = np.digitize(distance, xbins)
        x = 0.5*(xbins[1:] + xbins[:-1])

        # Absolute elevation swath profile

        swath_absolute = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        for i in range(1, len(xbins)):
            
            swath_elevations = elevations[mask & (binned == i)]
            if swath_elevations.size:
                swath_absolute[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # Relative-to-stream elevation swath profile

        swath_rel_stream = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        for i in range(1, len(xbins)):

            swath_elevations = relz[mask & (binned == i)]
            if swath_elevations.size:
                swath_rel_stream[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # Relative-to-valley-floor elevation swath profile

        swath_rel_valley = np.full((len(xbins)-1, 5), np.nan, dtype='float32')
        error_threshold = 1.0

        size = elevations.shape[0]*elevations.shape[1]
        matrix = np.stack([
            measure.reshape(size),
            np.ones(size, dtype='float32'),
            elevations.reshape(size)
        ]).T
        matrix = matrix[mask.reshape(size), :]
        samples = matrix.shape[0] // 10
        model = LinearModel([0, 1], [2])

        try:

            (slope, z0), _, _ = ransac(matrix, model, samples, 100, error_threshold, 2*samples)

            relative = elevations - (z0 + slope*measure)

            for i in range(1, len(xbins)):

                swath_elevations = relative[mask & (binned == i)]
                if swath_elevations.size:
                    swath_rel_valley[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        except RuntimeError:

            swath_rel_valley = np.array([])

        return axis, gid, x, swath_absolute, swath_rel_stream, swath_rel_valley


def SwathProfiles(axis, processes=1):

    dgo_shapefile = os.path.join(workdir, 'AX%03d_DGO.shp' % axis)
    relative_errors = 0
    
    def output(axis, gid):
        return os.path.join(workdir, 'SWATH', 'AX%03d_SWATH_%04d.npz' % (axis, gid))

    if processes == 1:

        with fiona.open(dgo_shapefile) as fs:
            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    gid = feature['properties']['GID']
                    measure = feature['properties']['M']
                    geometry = asShape(feature['geometry'])
                    _, _, x, swath_absolute, swath_rel_stream, swath_rel_valley = SwathProfile(axis, gid, geometry.bounds)

                    if swath_rel_valley.size == 0:
                        relative_errors += 1

                    np.savez(
                        output(axis, gid),
                        profile=(axis, gid, measure),
                        x=x,
                        swath_abs=swath_absolute,
                        swath_rel=swath_rel_stream,
                        swath_vb=swath_rel_valley)

    else:

        kwargs = dict()
        profiles = dict()
        arguments = list()

        with fiona.open(dgo_shapefile) as fs:
            for feature in fs:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                geometry = asShape(feature['geometry'])

                profiles[axis, gid] = [axis, gid, measure]
                arguments.append([SwathProfile, axis, gid, geometry.bounds, kwargs])

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)

            with click.progressbar(pooled, length=len(arguments)) as iterator:

                for axis, gid, x, swath_absolute, swath_rel_stream, swath_rel_valley in iterator:

                    profile = profiles[axis, gid]

                    if swath_rel_valley.size == 0:
                        relative_errors += 1

                    np.savez(
                        output(axis, gid),
                        profile=profile,
                        x=x,
                        swath_abs=swath_absolute,
                        swath_rel=swath_rel_stream,
                        swath_vb=swath_rel_valley)

    if relative_errors:
        click.secho('%d DGO without relative-to-valley-bottom profile' % relative_errors, fg='yellow')

def PlotSwath(axis, gid, kind='absolute', filename=None):

    from PlotSwath import plot_swath

    filename = os.path.join(workdir, 'SWATH', 'AX%03d_SWATH_%04d.npz' % (axis, gid))

    if os.path.exists(filename):

        data = np.load(filename, allow_pickle=True)

        x = data['x']
        _, _,  measure = data['profile']

        if kind == 'absolute':
            swath = data['swath_abs']
        elif kind == 'relative':
            swath = data['swath_rel']
        elif kind == 'valley bottom':
            swath = data['swath_vb']
            if swath.size == 0:
                click.secho('No relative-to-valley-bottom swath profile for DGO (%d, %d)' % (axis, gid), fg='yellow')
                click.secho('Using relative-to-nearest-drainage profile', fg='yellow')
                swath = data['swath_rel']
        else:
            click.secho('Unknown swath kind: %s' % kind)
            return

        if swath.shape[0] == x.shape[0]:
            title = 'Swath Profile PK %.0f m (DGO #%d)' % (measure, gid)
            output = os.path.join(workdir, 'SWATH', 'AX%03d_SWATH_%04d.pdf' % (axis, gid))
            plot_swath(-x, swath, kind in ('relative', 'valley bottom'), title, output)
        else:
            click.secho('Invalid swath data')
