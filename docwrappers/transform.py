"""
GeoTransform module :

- replacement for RasterIO .index() and .xy() methods
- vectorized transformation of NumPy array of coordinates
"""

import numpy as np
import fct.transform
from affine import Affine
from typing import Tuple, Union

GeoTransform = Union[Tuple, Affine]

def index(x: float, y: float, transform: GeoTransform) -> Tuple[int, int]:
    """
    Transform real world coordinates (x, y)
    into raster pixel coordinates (py, px)

    Arguments:

        x:
            x real world coordinates
        
        y:
            y real world coordinates

        transform:
            GDAL GeoTransform tuple or RasterIO Affine Transform object

    Returns:
        
        Raster pixel coordinates (row, col)
    """
    
    return fct.transform.index(x, y, transform)

def xy(row: int, col: int, transform: GeoTransform) -> Tuple[float, float]:
    """
    Transform raster pixel coordinates (py, px)
    into real world coordinates (x, y)

    Arguments:

        row:
            pixel row index

        col:
            pixel column index

        transform:
            GDAL GeoTransform tuple or RasterIO Affine Transform object

    Returns:

        Real world coordinates (x, y)
    """

    return fct.transform.xy(row, col, transform)

def worldtopixel(coordinates: np.ndarray, transform: GeoTransform) -> np.ndarray:
    """
    Transform real world coordinates (x, y)
    into raster pixel coordinates (py, px)

    Arguments:

        coordinates:
            array of (x, y) coordinates
            with dtype 'float32'

        transform:
            GDAL GeoTransform tuple
            or RasterIO Affine Transform object

    Returns:

        Raster pixel coordinates array
        with dtype 'int32'
    """

    return fct.transform.worldtopixel(coordinates, transform)

def pixeltoworld(pixels: np.ndarray, transform: GeoTransform) -> np.ndarray:
    """
    Transform raster pixel coordinates (py, px)
    into real world coordinates (x, y)

    Arguments:

        pixels: 
            
            array of (row, col) raster coordinates
            with dtype 'int32'

        transform:
            
            GDAL GeoTransform tuple
            or RasterIO Affine Transform object

    Returns:

        Real world coordinates array
        with dtype 'float32'
    """

    return fct.transform.pixeltoworld(pixels, transform)
