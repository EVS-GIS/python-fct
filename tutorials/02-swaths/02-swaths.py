# Copy the Hydrographic Reference to outputs/GLOBAL/REFHYDRO

import glob, shutil, os

if not os.path.isdir('../outputs/GLOBAL/REFHYDRO/'):
    os.mkdir('../outputs/GLOBAL/REFHYDRO/')

for f in glob.glob('../inputs/REFERENTIEL_HYDRO.*'):
    shutil.copy(f, '../outputs/GLOBAL/REFHYDRO/')

# Shortest Height
from fct.height import ShortestHeight
params = ShortestHeight.Parameters()
params.scale_distance = 25.0
params.mask_height_max = 100.0
params.height_max = 100.0
params.distance_min = 20.0
params.distance_max = 2000.0
params.jitter = 0.4

shutil.rmtree('../outputs/NETWORK/HEIGHT/10K/SHORTEST_HEIGHT/')
ShortestHeight.ShortestHeight(params, processes=16)

from fct.tileio import buildvrt
buildvrt('10k', 'shortest_height')

# Height above nearest drainage
from fct.height import HeightAboveNearestDrainage
params = HeightAboveNearestDrainage.Parameters()
params.mask_height_max = 100.0
params.buffer_width = 2000
params.resolution = 25.0

shutil.rmtree('../outputs/NETWORK/HEIGHT/10K/NEAREST_HEIGHT/')
HeightAboveNearestDrainage.HeightAboveNearestDrainage(params, processes=16)

from fct.tileio import buildvrt
buildvrt('10k', 'nearest_height')
buildvrt('10k', 'nearest_distance')
buildvrt('10k', 'nearest_drainage_axis')

# Disaggregate along refaxis
from fct.measure import SwathMeasurement
params = SwathMeasurement.Parameters()
params.mdelta = 200.0

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=16)
swaths_bounds = SwathMeasurement.WriteSwathsBounds(params, attrs=swaths)

from fct.tileio import buildvrt
buildvrt('10k', 'axis_measure')
buildvrt('10k', 'axis_nearest')
buildvrt('10k', 'axis_distance')
buildvrt('10k', 'swaths_refaxis')

# Swath drainage
from fct.corridor import SwathDrainage
params = SwathDrainage.Parameters()

swath_drainage = SwathDrainage.SwathDrainage(params, processes=16)

# Valley bottom features
from fct.corridor import ValleyBottomFeatures
params = ValleyBottomFeatures.Parameters()
params.height_max = 20.0
params.swath_length = 200.0
params.patch_min_pixels = 100

params.thresholds = [
    # drainage area km², distance min, distance max, max height (depth), max slope (%)
    ValleyBottomFeatures.ValleyBottomThreshold(0, 20.0, 100.0, 2.0, 10.0),
    ValleyBottomFeatures.ValleyBottomThreshold(30, 20.0, 400.0, 4.0, 7.0),
    ValleyBottomFeatures.ValleyBottomThreshold(250, 20.0, 1500.0, 5.0, 5.0),
    ValleyBottomFeatures.ValleyBottomThreshold(1000, 20.0, 2000.0, 6.0, 3.5),
    ValleyBottomFeatures.ValleyBottomThreshold(5000, 20.0, 2500.0, 6.5, 3.0),
    ValleyBottomFeatures.ValleyBottomThreshold(11500, 20.0, 4000.0, 7.5, 2.5),
    ValleyBottomFeatures.ValleyBottomThreshold(13000, 20.0, 4000.0, 8.5, 2.5),
    ValleyBottomFeatures.ValleyBottomThreshold(45000, 20.0, 7500.0, 10.5, 2.0)
]

shutil.rmtree('../outputs/NETWORK/TEMP/10K/VALLEY_BOTTOM_FEATURES/')
ValleyBottomFeatures.ClassifyValleyBottomFeatures(params, swath_drainage, processes=16)

from fct.tileio import buildvrt
buildvrt('10k', 'valley_bottom_features')

# Connected Valley bottom
from fct.corridor import ValleyBottomFinal
params = ValleyBottomFinal.Parameters()
params.distance_max = 2000
params.jitter = 0.9

shutil.rmtree('../outputs/NETWORK/TEMP/10K/VALLEY_BOTTOM_CONNECTED/')
shutil.rmtree('../outputs/NETWORK/TEMP/10K/VALLEY_BOTTOM_DISTANCE_CONNECTED/')
ValleyBottomFinal.ConnectedValleyBottom(params, processes=16)

shutil.rmtree('../outputs/NETWORK/HEIGHT/10K/VALLEY_BOTTOM/')
ValleyBottomFinal.TrueValleyBottom(params, processes=16)

from fct.tileio import buildvrt
buildvrt('10k', 'valley_bottom_final')

# Vectorize Refaxis Swaths
from fct.measure import SwathPolygons
params = SwathPolygons.Parameters()

swaths = SwathPolygons.Swaths(params, processes=16)
SwathPolygons.VectorizeSwaths(swaths, swath_drainage, params, processes=16)

# Calculate medial axes
from fct.corridor import MedialAxis2
params = MedialAxis2.Parameters()

medial_axis = MedialAxis2.MedialAxis(params)
MedialAxis2.SimplifyMedialAxis(params, medial_axis)

# Disaggregate along medial axes
from fct.measure import SwathMeasurement
params = SwathMeasurement.Parameters()
params.reference = 'medialaxis_simplified'
params.output_distance = 'medialaxis_distance'
params.output_measure = 'medialaxis_measure'
params.output_nearest = 'medialaxis_nearest'
params.output_swaths_raster = 'swaths_medialaxis'
params.output_swaths_shapefile = 'swaths_medialaxis_polygons'
params.output_swaths_bounds = 'swaths_medialaxis_bounds'
params.mdelta = 200.0

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=16)
swaths_bounds = SwathMeasurement.WriteSwathsBounds(params, attrs=swaths)

from fct.tileio import buildvrt
buildvrt('10k', 'swaths_medialaxis')

# Swath drainage
from fct.corridor import SwathDrainage
params = SwathDrainage.Parameters()

swath_drainage = SwathDrainage.SwathDrainage(params, processes=16)

# Vectorize Medialaxis Swaths
from fct.measure import SwathPolygons
params = SwathPolygons.Parameters()
params.swaths = 'swaths_medialaxis'
params.polygons = 'swaths_medialaxis_polygons'

swaths = SwathPolygons.Swaths(params, processes=16)
SwathPolygons.VectorizeSwaths(swaths, swath_drainage, params, processes=16)

# SimplifySwathsPolygons to get a clean vectorial output
from fct.measure import SimplifySwathPolygons2
params = SimplifySwathPolygons2.Parameters()

SimplifySwathPolygons2.SimplifySwathPolygons(params)