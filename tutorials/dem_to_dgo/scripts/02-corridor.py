# Copy the Hydrographic Reference to outputs/GLOBAL/REFHYDRO
# cp ./tutorials/dem_to_dgo/inputs/REFERENTIEL_HYDRO.* ./tutorials/dem_to_dgo/outputs/GLOBAL/INPUT/

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

# Disaggregate along referentiel hydro
from fct.measure import SwathMeasurement
SwathMeasurement.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathMeasurement.Parameters()
params.reference = 'stream-network-cartography-in'

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=4)

# Swath drainage
from fct.corridor import SwathDrainage
SwathDrainage.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathDrainage.Parameters()

swath_drainage = SwathDrainage.SwathDrainage(params)

# Valley bottom features
from fct.corridor import ValleyBottomFeatures
ValleyBottomFeatures.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ValleyBottomFeatures.Parameters()

ValleyBottomFeatures.ClassifyValleyBottomFeatures(params, swath_drainage)

# Connected Valley bottom
from fct.corridor import ValleyBottomFinal
ValleyBottomFinal.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ValleyBottomFinal.Parameters()

ValleyBottomFinal.ConnectedValleyBottom(params)
ValleyBottomFinal.TrueValleyBottom(params)

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

# Disaggregate along medial axes
from fct.measure import SwathMeasurement
SwathMeasurement.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathMeasurement.Parameters()
params.reference = 'medialaxis'

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=4)

# SwathProfile
from fct.profiles import SwathProfile
SwathProfile.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathProfile.Parameters()

swath_profiles = SwathProfile.SwathProfile(params, processes=4)
swath_profiles.to_netcdf('./tutorials/dem_to_dgo/outputs/NETWORK/METRICS/SWATHS_ELEVATION.nc')

# ValleyBottomWidth
from fct.metrics import ValleyBottomWidth2
ValleyBottomWidth2.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = ValleyBottomWidth2.Parameters()

vbw = ValleyBottomWidth2.ValleyBottomWidth(swath_profiles.set_index(sample=('axis', 'measure', 'distance')), params, processes=4)
vbw.to_netcdf('./tutorials/dem_to_dgo/outputs/NETWORK/METRICS/WIDTH_VALLEY_BOTTOM.nc')







# Simplify medial axes
from fct.corridor import MedialAxis2
MedialAxis2.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = MedialAxis2.Parameters()

medial_axis = MedialAxis2.SimplifyMedialAxis(params)










from fct.profiles import ValleyBottomElevationProfile

params = ValleyBottomElevationProfile.Parameters()

ValleyBottomElevationProfile.ValleyBottomElevationProfile(params)










# Valley swath
from fct.measure import SwathPolygons
params = SwathPolygons.Parameters()

SwathPolygons.Swaths(params)















for axis in SwathMeasurement.config.axes('refaxis'):
    SwathMeasurement.WriteSwathsBounds(params, swaths, axis=axis)





# from fct.measure import Measurement2
# Measurement2.config.from_file('./tutorials/dem_to_dgo/config.ini')
# params = Measurement2.Parameters()

# Measurement2.MeasureNetwork(params)



# Vectorize Refaxis Swaths
from fct.measure import SwathPolygons
SwathPolygons.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = SwathPolygons.Parameters()

swaths = SwathPolygons.Swaths(params)
SwathPolygons.VectorizeSwaths(swaths, swath_drainage, params)