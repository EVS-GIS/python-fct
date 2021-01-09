"""
Some older code from FCT Terrain Analysis
"""

# -- deprecated usage
# ta.index
# ta.xy
# ta.worldtopixel
# ta.pixeltoworld

# -- not reimplemented methods
# [x] ta.resolve_flat
# [x] ta.flat_mask_flowdir
# [x] ta.watershed_labels
# [x] ta.tile_outlets
# [x] ta.outlet
# [x] ta.flowdir
# [x] ta.signed_distance
# [x] ta.disaggregate

# -- FlatMap deprecated methods ?
# ta.watershed_max
# ta.shortest_max

from typing import Optional, Tuple, List
import numpy as np
from affine import Affine
import fct.terrain_analysis as ta

def resolve_flat(
        elevations: np.ndarray,
        flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a pseudo-height raster for DEM flats,
    suitable to calculate a realistic drainage direction raster.

    Arguments:

        elevations:
            
            Digital elevation model (DEM) raster,
            dtype 'float32',
            preprocessed for depression filling.
            Flats must have constant elevation.

        flow:

            D8 flow firection raster (int16 array)

    Returns:

        - flat_mask:

            Pseudo-height raster (float32 array),
            nodata = 0

        - labels:

            Flat label raster (uint32 array),
            nodata = 0
    """

    return ta.resolve_flat(elevations, flow)

def flat_mask_flowdir(
    mask: np.ndarray,
    flow: np.ndarray,
    labels: np.ndarray) -> np.ndarray:
    """
    Assign drainage direction to flat areas, according to pseudo-height in `mask`.
    Input `flow` raster is modified in place.

    See also: [resolve_flat()][fct.algorithms.terrain.resolve_flat]

    Arguments:

        mask:
            
            Pseudo-height raster (float32 array)

        flow:
            
            D8 flow firection raster (int16 array)

        labels:
            
            Flat label raster (uint32 array)

    Returns:

        Modified D8 flow direction raster (int16 array),
        nodata = -1
    """

    return ta.flat_mask_flowdir(mask, flow, labels)

def watershed_labels(
        elevations: np.ndarray,
        nodata: float,
        noout: float,
        # float dx, float dy,
        # float minslope=1e-3,
        # float[:, :] out = None,
        labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Fill sinks of digital elevation model (DEM),
    based on the algorithm of Wang & Liu (2006).

    Arguments:

        elevations:
            
            Digital elevation model (DEM) raster,
            dtype 'float32'

        nodata:
            
            no-data value in `elevations`

        noout:
            
            no-out value in `elevations`
            Provide max(elevations) if not applicable

        labels:

            Same shape and dtype as elevations, initialized to nodata

    Returns:

        - 'uint32' aster map of watershed labels
          starting from 1, with nodata = 0

        - watershed graph:
          dict {(label1, label2): minz}
          where label1 < label2 and (label1, label2) denotes an undirected link
          between watershed 1 and watershed 2,
          and minz is the minimum connecting elevation
          between the two waterhseds.
    """

    return ta.watershed_labels(elevations, nodata, noout, labels)

def tile_outlets(flow: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int]]]:
    """
    Find all outlet pixels according to flow
    and calculate their local contributive area.
    
    An optional mask raster may be used
    when the tile boundaries do not align with the raster grid.

    Arguments:

        flow:
            
            D8 flow firection raster (int16 array)

        mask:

            validity mask (uint8 array),
            where value > 0 indicates a valid pixel

    Returns:

        - list of tile outlets (i, j, area)
          where area is the local area drained by pixel (i, j)

        - list of target pixels (ix, jx) such as
          pixel (i, j) flows to pixel (ix, jx)
    """

    if mask is None:
        mask = np.ones_like(flow, dtype='uint8')

    return ta.tile_outlets(flow, mask)

def outlet(flow: np.ndarray, i0: int, j0: int) -> Tuple[int, int]:
    """
    Find the outlet pixel draining pixel (i0, j0)

    Arguments:

        flow:
            
            D8 flow firection raster (int16 array)

        i0:

            pixel row index

        j0:

            pixel column index

    Returns:

        (row, col) coordinates of outlet pixel
        draining origin pixel (i0, j0)
    """
    
    return ta.outlet(flow, i0, j0)

def flowdir(
        elevations: np.ndarray,
        nodata: float,
        flow: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Flow direction from elevation data.
    Assign flow direction toward the lower neighbouring cell.

    Arguments:

        elevations:
            Elevation raster (DEM),
            array-like, ndims=2, dtype=float32

        nodata:
            No data value for elevation

        flow:
            Output array initialized to -1

    Returns:

        D8 flow direction array,
        nodata = -1
    """

    return ta.flowdir(elevations, nodata, flow)

def signed_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Signed distance from points C to segments [AB].

    Arguments:

        a:
            vector of A points,
            array-like of pairs of float32 coordinates (x, y)
        
        b:
            vector of B points,
            array-like of pairs of float32 coordinates (x, y)
        
        c:
            vector of C points,
            array-like of pairs of float32 coordinates (x, y)

    Returns:

        tuple of 3 float32 arrays :

        - distance from C to segment [AB]
        - signed distance from C to infinite line (AB)
        - normalized distance of C's nearest point on [AB] from A,
          between 0.0 (C' == A) and 1.0 (C' == B)
    """

    return ta.signed_distance(a, b, c)

def disaggregate(
    geometry: np.ndarray,
    zone: np.ndarray,
    value: float,
    increment: float,
    transform: Affine,
    mask: np.ndarray,
    out: np.ndarray):
    """
    Disaggregate uniformly `value`
    over the extent given by `geometry`(must be a polygon)
    onto target pixels in raster `zone`.

    Arguments:

        geometry:

            Sequence of float32 coordinate pairs (x, y),
            defining a polygon exterior ring,
            first and last point must be the same

        zone:
            
            'int8' raster which defines where the targets pixels are :
            values in `zone`should be such as
            target pixels = 2, fallback pixels = 1, nodata = 0

        value:

            The value to disaggregate over the extent of `geometry`

        transform: 
            
            RasterIO transform from geometry coordinate system
            to raster pixel coordinates.

        mask: 
            
            temporary 'uint8' raster that can be reused between successive
            calls  to `disaggregate` ;
            must be initialized to zeros.

        out: 
            
            Output float32 raster, receiving disaggregated increments
            that sum up to `value`. The disaggregated values are added
            to pixel values in `out`
    """

    return ta.disaggregate(geometry, zone, value, increment, transform, mask, out)
