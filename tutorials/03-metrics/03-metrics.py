# Random Poisson Samples
from fct.measure import RandomPoissonSamples
params = RandomPoissonSamples.Parameters()

RandomPoissonSamples.RandomPoissonSamples(params, processes=16)

from fct.tileio import buildvrt
buildvrt('10k', 'poisson_samples')

# Talweg Elevation Profile
from fct.profiles import TalwegElevationProfile
params = TalwegElevationProfile.Parameters()
params.talweg = 'refaxis'
params.nearest = 'axis_nearest'
params.measure = 'axis_measure'

talweg_points = TalwegElevationProfile.TalwegElevation(params, processes=16)
talweg_points.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/REFAXIS_POINTS.nc')

# Reload swath bounds dataset
from fct.measure import SwathMeasurement
params = SwathMeasurement.Parameters()
params.output_swaths_bounds = 'swaths_medialaxis_bounds'
swath_bounds = SwathMeasurement.ReadSwathsBounds(params)







# Valley Bottom Height
# from fct.profiles import ValleyBottomHeight
# params = ValleyBottomHeight.Parameters()
# params.talweg = dict(key='metrics_refaxis_points', tiled=False, subdir='NETWORK/METRICS')

# river_profile = ValleyBottomHeight.ValleyBottomHeight(swath_bounds=swath_bounds, params=params, processes=16)
# river_profile.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/HEIGHT_FLOODPLAIN.nc')

# # VB Elevation Profile
# from fct.profiles import ValleyBottomElevationProfile
# params = ValleyBottomElevationProfile.Parameters()

# ValleyBottomElevationProfile.RefaxisSamplePoints(params)
# ValleyBottomElevationProfile.ValleyBottomElevationProfile(params)


# Swath Elevation Profile
from fct.profiles import SwathProfile
params = SwathProfile.Parameters()
params.swaths = 'swaths_medialaxis'

swath_profiles = SwathProfile.SwathProfile(params, processes=16)
swath_profiles.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/SWATHS_ELEVATION.nc')

# # Swath elevation metrics
# from fct.metrics import TalwegMetrics
# params = TalwegMetrics.Parameters()

# params.metrics_talweg


# Plot swath elevation profiles
from fct.plotting import PlotElevationSwath2
fig, ax = PlotElevationSwath2.setup_plot()
ax = PlotElevationSwath2.plot_profile_quantiles(ax, swath_profiles)
PlotElevationSwath2.finalize_plot(fig, ax)

# ValleyBottomWidth
from fct.metrics import ValleyBottomWidth2
params = ValleyBottomWidth2.Parameters()

vbw = ValleyBottomWidth2.ValleyBottomWidth(swath_profiles.set_index(sample=('axis', 'measure', 'distance')), params, processes=16)
vbw.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/WIDTH_VALLEY_BOTTOM.nc')





from fct.profiles import ValleyBottomElevationProfile

params = ValleyBottomElevationProfile.Parameters()

ValleyBottomElevationProfile.ValleyBottomElevationProfile(params)





######
# LANDCOVER AND CONTINUITY ANALYSIS
# NB: Landcover raster should be perfectly aligned with DEM and nodata value should be set in raster properties

# Prepare the tiles and VRT

from fct.cli import Tiles
Tiles.DatasourceToTiles('landcover', '10k', 'landcover-default', processes=64)

from fct.tileio import buildvrt
buildvrt('10k', 'landcover-default')

# Extract ValleyBottom Landcover
from fct.corridor import ValleyBottomLandcover
params = ValleyBottomLandcover.Parameters()

params.landcover = 'landcover-default'

ValleyBottomLandcover.ValleyBottomLandcover(params, processes=64)

from fct.tileio import buildvrt
buildvrt('10k', 'landcover_valley_bottom')

# # Continuity analysis
# from fct.corridor import ContinuityAnalysisWeighted
# params = ContinuityAnalysisWeighted.Parameters()

# ContinuityAnalysisWeighted.ContinuityAnalysisWeighted(params, processes=16)

# Continuity analysis
from fct.corridor import ContinuityAnalysisMax
params = ContinuityAnalysisMax.Parameters()

ContinuityAnalysisMax.ContinuityAnalysisMax(params, processes=16)

# Continuity remapping
from fct.corridor import ContinuityAnalysisRemap
params = ContinuityAnalysisRemap.Parameters()
params.output = 'continuity'

ContinuityAnalysisRemap.RemapContinuityRaster(params, processes=16, tag='MAX')

from fct.tileio import buildvrt
buildvrt('10k', 'continuity') 

###
# LANDCOVER METRICS

# Swath Landcover Profile
from fct.profiles import SwathProfile
params = SwathProfile.Parameters()
params.swaths = 'swaths_medialaxis'
params.values = 'landcover_valley_bottom'
params.is_continuous = False
params.labels = {0: 'Water Channel', 1: 'Gravel Bars', 2: 'Natural Open', 3: 'Forest', 4: 'Grassland', 5: 'Crops', 6: 'Diffuse Urban', 7: 'Dense Urban', 8: 'Infrastructures'}

swath_profiles = SwathProfile.SwathProfile(params, processes=16)
swath_profiles.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/SWATHS_LANDCOVER.nc')

# Landcover width (TODO: Update CorridorWidth2.py)
import xarray as xr
from fct.metrics import DiscreteClassesWidth
from pathlib import Path

data = (
    xr.open_dataset('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/SWATHS_LANDCOVER.nc')
    .set_index(sample=('axis', 'measure', 'distance'))
    .load()
)

params = DiscreteClassesWidth.Parameters()
params.resolution = 1.0
width_landcover = DiscreteClassesWidth.DiscreteClassesWidth(data, params, processes=16)
width_sum = width_landcover.sum(['label', 'side'])

width_landcover['width_landcover'] = (
    width_landcover.width2 / width_sum.width2 * width_sum.width1
) 
# width1 = area / swath_length
# width2 = np.sum(data.profile / density_ref, axis=0) * distance_delta

width_landcover.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/WIDTH_LANDCOVER.nc')

# Export metrics to csv
width_landcover.to_dataframe().to_csv('/data/sdunesme/fct/tests_1m/fct_workdir/WIDTH_LANDCOVER.csv')

### 
# CONTINUITY METRICS

# Swath Continuity Profile
from fct.profiles import SwathProfile
params = SwathProfile.Parameters()
params.swaths = 'swaths_medialaxis'
params.values = 'continuity'
params.is_continuous = False
params.labels = {0: 'Water Channel', 1: 'Active Channel', 10: 'Riparian Buffer', 20: 'Connected Meadows', 30: 'Connected Cultivated', 40: 'Disconnected', 50: 'Built', 255: 'No Data'}

swath_profiles = SwathProfile.SwathProfile(params, processes=16, tag='MAX')
swath_profiles.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/SWATHS_CONTINUITY.nc')

# Continuity width (TODO: Update CorridorWidth2.py)
import xarray as xr
from fct.metrics import DiscreteClassesWidth
from pathlib import Path

data = (
    xr.open_dataset('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/SWATHS_CONTINUITY.nc')
    .set_index(sample=('axis', 'measure', 'distance'))
    .load()
)

params = DiscreteClassesWidth.Parameters()
params.resolution = 1.0
width_continuity = DiscreteClassesWidth.DiscreteClassesWidth(data, params, processes=16)
width_sum = width_continuity.sum(['label', 'side'])

width_continuity['width_continuity'] = (
    width_continuity.width2 / width_sum.width2 * width_sum.width1
) 
# width1 = area / swath_length
# width2 = np.sum(data.profile / density_ref, axis=0) * distance_delta

width_continuity.to_netcdf('/data/sdunesme/fct/tests_1m/fct_workdir/NETWORK/METRICS/WIDTH_CONTINUITY.nc')

# Export metrics to csv
width_continuity.to_dataframe().to_csv('/data/sdunesme/fct/tests_1m/fct_workdir/WIDTH_CONTINUITY.csv')
