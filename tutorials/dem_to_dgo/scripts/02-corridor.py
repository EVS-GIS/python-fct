# Setup axis directory structure
from fct.axis import SetupAxes
SetupAxes.config.from_file('./tutorials/dem_to_dgo/config.ini')
SetupAxes.SetupAxes()

# Shortest Height
from fct.height import ShortestHeight
ShortestHeight.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ShortestHeight.Parameters()

ShortestHeight.ShortestHeight(params)

# Height above nearest drainage
from fct.height import HeightAboveNearestDrainage
HeightAboveNearestDrainage.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = HeightAboveNearestDrainage.Parameters()

HeightAboveNearestDrainage.HeightAboveNearestDrainage(params)

# Valley mask for each axis
from fct.corridor import ValleyBottomMask
ValleyBottomMask.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ValleyBottomMask.Parameters()
params.output = 'ax_valley_bottom_mask'

for axis in ValleyBottomMask.config.axes('refaxis'):
    ValleyBottomMask.ValleyBottomMask(axis=axis, params=params)




