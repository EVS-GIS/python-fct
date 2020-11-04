import os
import click
import rasterio as rio
from fct.config import config

def SoustractForeignValleyMask(ax1, ax2, side):

    ax_tiles = 'ax_shortest_tiles'
    tilefile = config.tileset().filename(ax_tiles, axis=ax1)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def tiles():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))
                yield row, col

    with click.progressbar(tiles(), length=length()) as iterator:
        for row, col in iterator:

            valley_mask_raster = config.tileset().tilename('ax_valley_mask', axis=ax1, row=row, col=col)
            other_height_raster = config.tileset().tilename('ax_nearest_height', axis=ax2, row=row, col=col)
            other_distance_raster = config.tileset().tilename('ax_nearest_distance', axis=ax2, row=row, col=col)

            if os.path.exists(other_height_raster):

                with rio.open(other_height_raster) as ds:
                    other_height = ds.read(1)
                    other_height_nodata = ds.nodata

                with rio.open(other_distance_raster) as ds:
                    other_distance = ds.read(1)

                with rio.open(valley_mask_raster) as ds:

                    valley_mask = ds.read(1)
                    crop_mask = (other_height != other_height_nodata) & (other_distance < 0)
                    valley_mask[crop_mask] = ds.nodata

                    profile = ds.profile.copy()

                profile.update(compress='deflate')

                with rio.open(valley_mask_raster, 'w', **profile) as dst:
                    dst.write(valley_mask, 1)

