import shutil, os
from fct.config import config
from multiprocessing import cpu_count
p = cpu_count()/2

# Copy the Hydrographic Reference to outputs/GLOBAL/REFHYDRO
if not os.path.isdir(f'{config.workdir}/GLOBAL/REFHYDRO/'):
    os.mkdir(f'{config.workdir}/GLOBAL/REFHYDRO/')

# COPY input REFHYDRO to {config.workdir}/GLOBAL/REFHYDRO/REFERENTIEL_HYDRO.shp

# Shortest path exploration
from fct.height import ShortestHeight
params = ShortestHeight.Parameters()
params.dem = 'dem-drainage-resolved' # elevation raster (DEM)
#params.reference = 'network-cartography-ready'
params.mask = 'off' # 'nearest_height' height raster defining domain mask
params.scale_distance = 5.0 # scale distance output with given scale factor, corresponding to pixel resolution
params.mask_height_max = 100.0 # maximum height defining domain mask
params.height_max = 100.0 # stop at maximum height above reference
params.distance_min = 4.0 # minimum distance before applying stop criteria, expressed in pixels)
params.distance_max = 800.0 # stop at maximum distance, expressed in pixels
params.jitter = 0.4 # apply jitter on performing shortest path exploration

if os.path.isdir(f'{config.workdir}/NETWORK/HEIGHT/DEFAULT/SHORTEST_HEIGHT/'):
    shutil.rmtree(f'{config.workdir}/NETWORK/HEIGHT/DEFAULT/SHORTEST_HEIGHT/')
ShortestHeight.ShortestHeight(params, processes=p) 

from fct.tileio import buildvrt
buildvrt('default', 'shortest_height')
buildvrt('default', 'shortest_distance')
buildvrt('default', 'shortest_state')

# Height above nearest drainage
from fct.height import HeightAboveNearestDrainage
params = HeightAboveNearestDrainage.Parameters()
params.dem = 'dem-drainage-resolved'
params.tiles = 'shortest_tiles'
params.drainage = 'network-cartography-ready'
# params.drainage = 'refaxis'
params.mask = 'shortest_height'
params.height = 'nearest_height'
params.distance = 'nearest_distance'
params.nearest = 'nearest_drainage_axis'
params.mask_height_max = 100.0 # maximum height defining domain mask
params.buffer_width = 2000 # enlarge domain mask by buffer width expressed in real distance unit (eg. meters)
params.resolution = 5.0 # raster resolution, ie. pixel size, in real distance unit (eg. meters)

if os.path.isdir(f'{config.workdir}/NETWORK/HEIGHT/DEFAULT/NEAREST_HEIGHT/'):
    shutil.rmtree(f'{config.workdir}/NETWORK/HEIGHT/DEFAULT/NEAREST_HEIGHT/')
HeightAboveNearestDrainage.HeightAboveNearestDrainage(params, processes=p)

from fct.tileio import buildvrt
buildvrt('default', 'nearest_height')
buildvrt('default', 'nearest_distance')
buildvrt('default', 'nearest_drainage_axis')

# Disaggregate along refaxis
from fct.measure import SwathMeasurement
params = SwathMeasurement.Parameters()
# params.reference = 'network-cartography-ready'
params.reference = 'refaxis'
params.mdelta = 200.0

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=p)
swaths_bounds = SwathMeasurement.WriteSwathsBounds(params, attrs=swaths)

from fct.tileio import buildvrt
buildvrt('default', 'axis_measure')
buildvrt('default', 'axis_nearest')
buildvrt('default', 'axis_distance')
buildvrt('default', 'swaths_refaxis')

# Swath drainage
from fct.corridor import SwathDrainage
params = SwathDrainage.Parameters()
params.axis = 'axis_nearest'

swath_drainage = SwathDrainage.SwathDrainage(params, processes=p)
SwathDrainage.WriteDrainageToDisk(swath_drainage, params)
swath_drainage = SwathDrainage.ReadDrainageFromDisk(params.output.filename())

# Valley bottom features
from fct.corridor import ValleyBottomFeatures
params = ValleyBottomFeatures.Parameters()
params.dem = 'dem-drainage-resolved'
params.axis = 'axis_nearest'
params.distance = 'axis_distance'
params.measure = 'axis_measure'
params.height_max = 20.0
params.swath_length = 200.0
params.patch_min_pixels = 100

params.thresholds = [
    # drainage area km², distance min, distance max, max height (depth), max slope (%)
    ValleyBottomFeatures.ValleyBottomThreshold(0, 5.0, 50.0, 0.5, 12.0),
    ValleyBottomFeatures.ValleyBottomThreshold(1, 5.0, 60.0, 0.7, 11.8),
    ValleyBottomFeatures.ValleyBottomThreshold(2, 5.0, 70.0, 0.9, 11.6),
    ValleyBottomFeatures.ValleyBottomThreshold(3, 5.0, 80.0, 1.1, 11.4),
    ValleyBottomFeatures.ValleyBottomThreshold(4, 5.0, 90.0, 1.3, 11.2),
    ValleyBottomFeatures.ValleyBottomThreshold(5, 5.0, 100.0, 1.5, 11.0),
    ValleyBottomFeatures.ValleyBottomThreshold(6, 5.0, 110.0, 1.7, 10.8),
    ValleyBottomFeatures.ValleyBottomThreshold(7, 5.0, 120.0, 1.9, 10.6),
    ValleyBottomFeatures.ValleyBottomThreshold(8, 5.0, 130.0, 2.1, 10.4),
    ValleyBottomFeatures.ValleyBottomThreshold(9, 5.0, 140.0, 2.4, 10.2),
    ValleyBottomFeatures.ValleyBottomThreshold(10, 10.0, 150.0, 2.6, 10.0),
    ValleyBottomFeatures.ValleyBottomThreshold(11, 11.0, 160.0, 2.8, 9.8),
    ValleyBottomFeatures.ValleyBottomThreshold(12, 12.0, 170.0, 3.0, 9.6),
    ValleyBottomFeatures.ValleyBottomThreshold(13, 13.0, 180.0, 3.0, 9.4),
    ValleyBottomFeatures.ValleyBottomThreshold(14, 14.0, 190.0, 3.0, 9.2),
    ValleyBottomFeatures.ValleyBottomThreshold(15, 15.0, 200.0, 3.0, 9.0),
    ValleyBottomFeatures.ValleyBottomThreshold(30, 20.0, 400.0, 4.0, 7.0),
    ValleyBottomFeatures.ValleyBottomThreshold(250, 20.0, 1500.0, 5.0, 5.0),
    ValleyBottomFeatures.ValleyBottomThreshold(1000, 20.0, 2000.0, 6.5, 3.5),
    ValleyBottomFeatures.ValleyBottomThreshold(3000, 20.0, 2000.0, 7.0, 3.5),
    ValleyBottomFeatures.ValleyBottomThreshold(5000, 20.0, 2500.0, 8.0, 3.0),
    ValleyBottomFeatures.ValleyBottomThreshold(11500, 20.0, 4000.0, 8.5, 2.5),
    ValleyBottomFeatures.ValleyBottomThreshold(13000, 20.0, 4000.0, 9.0, 2.5),
    ValleyBottomFeatures.ValleyBottomThreshold(45000, 20.0, 7500.0, 14, 2.0)
]


if os.path.isdir(f'{config.workdir}/NETWORK/TEMP/DEFAULT/VALLEY_BOTTOM_FEATURES/'):
    shutil.rmtree(f'{config.workdir}/NETWORK/TEMP/DEFAULT/VALLEY_BOTTOM_FEATURES/')

ValleyBottomFeatures.ClassifyValleyBottomFeatures(params, swath_drainage, processes=p)

from fct.tileio import buildvrt
buildvrt('default', 'valley_bottom_features')
buildvrt('default', 'slope')

# Connected Valley bottom
from fct.corridor import ValleyBottomFinal
params = ValleyBottomFinal.Parameters()
params.distance = 'axis_distance'
# params.distance_max = 2000
params.jitter = 0.4

if os.path.isdir(f'{config.workdir}/NETWORK/TEMP/DEFAULT/VALLEY_BOTTOM_CONNECTED/'):
    shutil.rmtree(f'{config.workdir}/NETWORK/TEMP/DEFAULT/VALLEY_BOTTOM_CONNECTED/')
if os.path.isdir(f'{config.workdir}/NETWORK/TEMP/DEFAULT/VALLEY_BOTTOM_DISTANCE_CONNECTED/'):
    shutil.rmtree(f'{config.workdir}/NETWORK/TEMP/DEFAULT/VALLEY_BOTTOM_DISTANCE_CONNECTED/')
ValleyBottomFinal.ConnectedValleyBottom(params, processes=p)

if os.path.isdir(f'{config.workdir}/NETWORK/HEIGHT/DEFAULT/VALLEY_BOTTOM/'):
    shutil.rmtree(f'{config.workdir}/NETWORK/HEIGHT/DEFAULT/VALLEY_BOTTOM/')
ValleyBottomFinal.TrueValleyBottom(params, processes=p)

from fct.tileio import buildvrt
buildvrt('default', 'valley_bottom_final')

# Calculate medial axes
from fct.corridor import MedialAxis2
params = MedialAxis2.Parameters()
params.swath_length = 100.0
params.nearest = 'axis_nearest'
params.distance = 'axis_distance'

medial_axis = MedialAxis2.MedialAxis(params)
MedialAxis2.SimplifyMedialAxis(params, medial_axis)
medial_axis.to_dataframe(name='vbw').to_csv(f'{config.workdir}/VBW_TEMP.csv')

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

swaths = SwathMeasurement.DisaggregateIntoSwaths(params, processes=p, tileset='default')
swaths_bounds = SwathMeasurement.WriteSwathsBounds(params, attrs=swaths)

from fct.tileio import buildvrt
buildvrt('default', 'medialaxis_measure')
buildvrt('default', 'medialaxis_nearest')
buildvrt('default', 'medialaxis_distance')
buildvrt('default', 'swaths_medialaxis')

# Swath drainage
from fct.corridor import SwathDrainage
params = SwathDrainage.Parameters()
params.drainage = 'acc'
params.axis = 'medialaxis_nearest'
params.measure = 'medialaxis_measure'
params.swath_length = 200.0

swath_drainage = SwathDrainage.SwathDrainage(params, processes=p)
SwathDrainage.WriteDrainageToDisk(swath_drainage, params)
swath_drainage = SwathDrainage.ReadDrainageFromDisk(params.output.filename(tileset='default'))

# Vectorize Medialaxis Swaths
from fct.measure import SwathPolygons
params = SwathPolygons.Parameters()
#params.nearest = 'nearest_drainage_axis'
params.nearest = 'medialaxis_nearest'
params.distance = 'medialaxis_distance'
params.swaths = 'swaths_medialaxis'
params.polygons = 'swaths_medialaxis_polygons'
params.measure = 'medialaxis_measure'
params.swath_length = 200.0

swaths = SwathPolygons.Swaths(params, processes=p)
SwathPolygons.VectorizeSwaths(swaths, swath_drainage, params, processes=p)

# SimplifySwathsPolygons to get a clean vectorial outputq
from fct.measure import SimplifySwathPolygons2
params = SimplifySwathPolygons2.Parameters()

SimplifySwathPolygons2.SimplifySwathPolygons(params)
