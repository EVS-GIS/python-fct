# Copy the Hydrographic Reference to outputs/GLOBAL/REFHYDRO
# cp ./tutorials/dem_to_dgo/inputs/REFERENTIEL_HYDRO.* ./tutorials/dem_to_dgo/outputs/GLOBAL/INPUT/

# Shortest Height
from fct.height import ShortestHeight
ShortestHeight.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ShortestHeight.Parameters()

ShortestHeight.ShortestHeight(params, processes=4)

# Height above nearest drainage
from fct.height import HeightAboveNearestDrainage
HeightAboveNearestDrainage.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = HeightAboveNearestDrainage.Parameters()

HeightAboveNearestDrainage.HeightAboveNearestDrainage(params, processes=4)

# Disaggregate along referentiel hydro
from fct.measure import SwathMeasurement
SwathMeasurement.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathMeasurement.Parameters()
params.reference = 'stream-network-cartography-in'
params.mdelta = 200.0

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=4)

# Swath drainage
from fct.corridor import SwathDrainage
SwathDrainage.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathDrainage.Parameters()

swath_drainage = SwathDrainage.SwathDrainage(params, processes=4)

# Vectorize Refaxis Swaths
from fct.measure import SwathPolygons
SwathPolygons.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathPolygons.Parameters()

swaths = SwathPolygons.Swaths(params, processes=4)
SwathPolygons.VectorizeSwaths(swaths, swath_drainage, params, processes=4)

# Valley bottom features
from fct.corridor import ValleyBottomFeatures
ValleyBottomFeatures.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ValleyBottomFeatures.Parameters()
params.height_max = 20.0

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

ValleyBottomFeatures.ClassifyValleyBottomFeatures(params, swath_drainage, processes=4)

# Connected Valley bottom
from fct.corridor import ValleyBottomFinal
ValleyBottomFinal.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ValleyBottomFinal.Parameters()

ValleyBottomFinal.ConnectedValleyBottom(params, processes=4)
ValleyBottomFinal.TrueValleyBottom(params, processes=4)

# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k nearest_distance
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k swaths_refaxis
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k valley_bottom_final
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k nearest_drainage_axis
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k axis_measure
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k axis_distance

# Calculate medial axes
from fct.corridor import MedialAxis2
MedialAxis2.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = MedialAxis2.Parameters()

medial_axis = MedialAxis2.MedialAxis(params)
MedialAxis2.SimplifyMedialAxis(params, medial_axis)

# Disaggregate along medial axes
# TODO: create separate outputs for medialaxis
from fct.measure import SwathMeasurement
SwathMeasurement.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathMeasurement.Parameters()
params.reference = 'medialaxis_simplified'
params.output_distance = 'axis_distance'
params.output_measure = 'axis_measure'
params.output_nearest = 'axis_nearest'
params.output_swaths_raster = 'swaths_medialaxis'
params.output_swaths_shapefile = 'swaths_medialaxis_polygons'
params.output_swaths_bounds = 'swaths_medialaxis_bounds'
params.mdelta = 200.0

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=4)

# Swath drainage
from fct.corridor import SwathDrainage
SwathDrainage.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathDrainage.Parameters()

swath_drainage = SwathDrainage.SwathDrainage(params, processes=4)

# Vectorize Medialaxis Swaths
from fct.measure import SwathPolygons
SwathPolygons.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathPolygons.Parameters()
params.swaths = 'swaths_medialaxis'
params.polygons = 'swaths_medialaxis_polygons'

swaths = SwathPolygons.Swaths(params, processes=4)
SwathPolygons.VectorizeSwaths(swaths, swath_drainage, params, processes=4)

#TODO: SimplifySwathsPolygons to get a clean vectorial output
