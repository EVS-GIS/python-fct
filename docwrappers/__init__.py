"""
Mkdocstrings documentation wrappers
"""

from . import (
    transform,
    raster,
    drainage,
    continuity,
    terrain
)

# set fake module name for pytkdocs
transform.__name__ = 'fct.transform'
raster.__name__ = 'fct.speedup'
drainage.__name__ = 'fct.speedup'
continuity.__name__ = 'fct.speedup'
terrain.__name__ = 'fct.terrain_analysis'
