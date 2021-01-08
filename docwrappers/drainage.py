"""
D8 drainage algorithms
"""

from typing import Tuple, List, Optional
import numpy as np
import fct.speedup


def flow_accumulation(flow: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Flow accumulation from D8 flow direction.

    Arguments:

        flow:
            D8 flow direction raster,
            ndim=2,
            dtype='int8',
            nodata=-1

        out:
            output float32 array
            initialized to 0

    Returns:

        float32 accumulation raster
    """

    return fct.speedup.flow_accumulation(flow, out)

def outlets(flow: np.ndarray) -> Tuple[List, List]:
    """
    Find all cells flowing outside of raster

    Arguments:

        flow:
            D8 flow direction raster,
            ndim=2,
            dtype='int8',
            nodata=-1

    Returns:

        - list of outlet pixel coordinates (row, col)
        - list of corresponding target pixel coordinates (row, col)
          outside of raster range
    """

    return fct.speedup.outlets(flow)
