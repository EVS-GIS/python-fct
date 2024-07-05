from fct.config import config
from multiprocessing import cpu_count
p = cpu_count()/2

######
# Random Poisson Samples
from fct.measure import RandomPoissonSamples
params = RandomPoissonSamples.Parameters()

RandomPoissonSamples.RandomPoissonSamples(params, processes=p)

from fct.tileio import buildvrt
buildvrt('default', 'poisson_samples')

######
# Reload swath bounds dataset
from fct.measure import SwathBounds
from fct.config.descriptors import DatasetResolver
swath_bounds = SwathBounds.SwathBounds(DatasetResolver('swaths_medialaxis_polygons'))

######
# Swath Elevation Profile
from fct.profiles import SwathProfile
params = SwathProfile.Parameters()
params.values = 'dem-drainage-resolved'
params.nearest = 'medialaxis_nearest'
params.axis_distance = 'medialaxis_distance'
params.swaths = 'swaths_medialaxis'
params.is_continuous = True
params.polygons = 'swaths_medialaxis_polygons'

swath_profiles = SwathProfile.SwathProfile(params, processes=p)
swath_profiles.to_netcdf(f'{config.workdir}/NETWORK/METRICS/SWATHS_ELEVATION.nc')
swath_profiles.to_dataframe().to_csv(f'{config.workdir}/SWATH_ELEVATION_PROFILES.csv')

# Reload swath_profiles if needed
import xarray as xr
swath_profiles = xr.load_dataset(f'{config.workdir}/NETWORK/METRICS/SWATHS_ELEVATION.nc')

######
# ValleyBottomWidth
from fct.metrics import ValleyBottomWidth2
params = ValleyBottomWidth2.Parameters()

vbw = ValleyBottomWidth2.ValleyBottomWidth(swath_profiles.set_index(sample=('axis', 'measure', 'distance')), params, processes=p)
vbw.to_netcdf(f'{config.workdir}/NETWORK/METRICS/WIDTH_VALLEY_BOTTOM.nc')
vbw.to_dataframe().to_csv(f'{config.workdir}/WIDTH_VALLEY_BOTTOM.csv')

######
# Valley Bottom Height
from fct.profiles import ValleyBottomElevationProfile
params = ValleyBottomElevationProfile.Parameters()
ValleyBottomElevationProfile.RefaxisSamplePoints(params)

from fct.profiles import ValleyBottomHeight
params = ValleyBottomHeight.Parameters()
params.talweg = dict(key='metrics_refaxis_points', tiled=False, subdir='NETWORK/METRICS')
params.measure = 'medialaxis_measure'
params.nearest = 'medialaxis_nearest'

river_profile = ValleyBottomHeight.ValleyBottomHeight(swath_bounds=swath_bounds, params=params, processes=p)
river_profile.to_netcdf(f'{config.workdir}/NETWORK/METRICS/HEIGHT_FLOODPLAIN.nc')
river_profile.to_dataframe().to_csv(f'{config.workdir}/HEIGHT_FLOODPLAIN.csv')

######
# Talweg metrics

# Talweg Elevation Profile
from fct.profiles import TalwegElevationProfile
params = TalwegElevationProfile.Parameters()
# params.talweg = 'refaxis'
params.measure = 'medialaxis_measure'
params.nearest = 'medialaxis_nearest'
params.sample_distance = 10.0

talweg_points = TalwegElevationProfile.TalwegElevation(params, processes=p)
talweg_points.to_netcdf(f'{config.workdir}/NETWORK/METRICS/REFAXIS_POINTS.nc')

swath_profiles = TalwegElevationProfile.TalwegElevationProfile(data=talweg_points, 
                                                                swath_bounds=swath_bounds,
                                                                params=params,
                                                                processes=p)
swath_profiles.to_netcdf(f'{config.workdir}/NETWORK/METRICS/TALWEG_METRICS.nc')
swath_profiles.to_dataframe().to_csv(f'{config.workdir}/TALWEG_METRICS.csv')

######
# LANDCOVER AND CONTINUITY ANALYSIS
# NB: Landcover raster should be perfectly aligned with DEM and nodata value should be set in raster properties

# Prepare the tiles and VRT

from fct.cli import Tiles
Tiles.DatasourceToTiles('landcover5m', 'default', 'landcover-default', processes=p, overwrite=True)

from fct.tileio import buildvrt
buildvrt('default', 'landcover-default')

# Extract ValleyBottom Landcover
from fct.corridor import ValleyBottomLandcover
params = ValleyBottomLandcover.Parameters()

params.landcover = 'landcover-default'

ValleyBottomLandcover.ValleyBottomLandcover(params, processes=p)

from fct.tileio import buildvrt
buildvrt('default', 'landcover_valley_bottom')

# Continuity analysis
from fct.corridor import ContinuityAnalysisMax
params = ContinuityAnalysisMax.Parameters()

ContinuityAnalysisMax.ContinuityAnalysisMax(params, processes=p)

# Continuity remapping
from fct.corridor import ContinuityAnalysisRemap
params = ContinuityAnalysisRemap.Parameters()
params.output = 'continuity'

ContinuityAnalysisRemap.RemapContinuityRaster(params, processes=p, tag='MAX')

from fct.tileio import buildvrt
buildvrt('default', 'continuity') 

###
# LANDCOVER METRICS

# Swath Landcover Profile
from fct.profiles import SwathProfile
params = SwathProfile.Parameters()

params.values = 'landcover_valley_bottom'
params.nearest = 'medialaxis_nearest'
params.axis_distance = 'medialaxis_distance'
params.talweg_distance = 'nearest_distance'
params.swaths = 'swaths_medialaxis'
params.polygons = 'swaths_medialaxis_polygons'
params.output = 'swath_landcover'
params.is_continuous = False        
params.swath_length = 200.0
params.distance_delta = 10.0
params.percentiles = None
params.labels = {
        0: 'Water Channel', 
        1: 'Gravel Bars', 
        2: 'Natural Open', 
        3: 'Forest', 
        4: 'Grassland', 
        5: 'Crops', 
        6: 'Diffuse Urban', 
        7: 'Dense Urban', 
        8: 'Infrastructures'
    }

swath_profiles = SwathProfile.SwathProfile(params, processes=p)
swath_profiles.to_netcdf(f'{config.workdir}/NETWORK/METRICS/SWATHS_LANDCOVER.nc')

# Landcover width
import xarray as xr
from fct.metrics import DiscreteClassesWidth
from pathlib import Path

data = (
    xr.open_dataset(f'{config.workdir}/NETWORK/METRICS/SWATHS_LANDCOVER.nc')
    .set_index(sample=('axis', 'measure', 'distance'))
    .load()
)

params = DiscreteClassesWidth.Parameters()
params.resolution = 1.0
width_landcover = DiscreteClassesWidth.DiscreteClassesWidth(data, params, processes=p)
width_sum = width_landcover.sum(['label', 'side'])

width_landcover['width_landcover'] = (
    width_landcover.width2 / width_sum.width2 * width_sum.width1
) 
# Pour info :
# width1 = area / swath_length
# width2 = np.sum(data.profile / density_ref, axis=0) * distance_delta

width_landcover.to_netcdf(f'{config.workdir}/NETWORK/METRICS/WIDTH_LANDCOVER.nc')

# Export metrics to csv
width_landcover.to_dataframe().to_csv(f'{config.workdir}/WIDTH_LANDCOVER.csv')

### 
# CONTINUITY METRICS

# Swath Continuity Profile
from fct.profiles import SwathProfile
params = SwathProfile.Parameters()

params.values = 'continuity'
params.nearest = 'medialaxis_nearest'
params.axis_distance = 'medialaxis_distance'
params.talweg_distance = 'nearest_distance'
params.swaths = 'swaths_medialaxis'
params.polygons = 'swaths_medialaxis_polygons'
params.output = 'swath_continuity'
params.is_continuous = False        
params.swath_length = 200.0
params.distance_delta = 10.0
params.percentiles = None
params.labels = {
    0: 'Water Channel', 
    1: 'Active Channel', 
    10: 'Riparian Buffer', 
    20: 'Connected Meadows', 
    30: 'Connected Cultivated', 
    40: 'Disconnected', 
    50: 'Built', 
    255: 'No Data'}

swath_profiles = SwathProfile.SwathProfile(params, processes=p, tag='MAX')
swath_profiles.to_netcdf(f'{config.workdir}/NETWORK/METRICS/SWATHS_CONTINUITY.nc')

# Continuity width
import xarray as xr
from fct.metrics import DiscreteClassesWidth

data = (
    xr.open_dataset(f'{config.workdir}/NETWORK/METRICS/SWATHS_CONTINUITY.nc')
    .set_index(sample=('axis', 'measure', 'distance'))
    .load()
)

params = DiscreteClassesWidth.Parameters()
params.resolution = 1.0
width_continuity = DiscreteClassesWidth.DiscreteClassesWidth(data, params, processes=p)
width_sum = width_continuity.sum(['label', 'side'])

width_continuity['width_continuity'] = (
    width_continuity.width2 / width_sum.width2 * width_sum.width1
) 

width_continuity.to_netcdf(f'{config.workdir}/NETWORK/METRICS/WIDTH_CONTINUITY.nc')

# Export metrics to csv
width_continuity.to_dataframe().to_csv(f'{config.workdir}/WIDTH_CONTINUITY.csv')
