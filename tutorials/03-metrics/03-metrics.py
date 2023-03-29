# Random Poisson Samples
from fct.measure import RandomPoissonSamples
params = RandomPoissonSamples.Parameters()

RandomPoissonSamples.RandomPoissonSamples(params, processes=8)

from fct.tileio import buildvrt
buildvrt('10k', 'poisson_samples')

# Talweg Elevation Profile
from fct.profiles import TalwegElevationProfile
params = TalwegElevationProfile.Parameters()
params.talweg = 'refaxis'
params.nearest = 'axis_nearest'
params.measure = 'axis_measure'

talweg_points = TalwegElevationProfile.TalwegElevation(params, processes=8)
talweg_points.to_netcdf('../outputs/NETWORK/METRICS/REFAXIS_POINTS.nc')

# Reload swath bounds dataset
from fct.measure import SwathMeasurement
params = SwathMeasurement.Parameters()
params.output_swaths_bounds = 'swaths_medialaxis_bounds'
swath_bounds = SwathMeasurement.ReadSwathsBounds(params)







# Valley Bottom Height
from fct.profiles import ValleyBottomHeight
params = ValleyBottomHeight.Parameters()
params.talweg = dict(key='metrics_refaxis_points', tiled=False, subdir='NETWORK/METRICS')

river_profile = ValleyBottomHeight.ValleyBottomHeight(swath_bounds=swath_bounds, params=params, processes=8)
river_profile.to_netcdf('../outputs/NETWORK/METRICS/HEIGHT_FLOODPLAIN.nc')

# VB Elevation Profile
from fct.profiles import ValleyBottomElevationProfile
params = ValleyBottomElevationProfile.Parameters()

ValleyBottomElevationProfile.RefaxisSamplePoints(params)
ValleyBottomElevationProfile.ValleyBottomElevationProfile(params)


# # SwathProfile
# from fct.profiles import SwathProfile
# params = SwathProfile.Parameters()
# params.swaths = 'swaths_medialaxis'

# swath_profiles = SwathProfile.SwathProfile(params, processes=8)
# swath_profiles.to_netcdf('../outputs/NETWORK/METRICS/SWATHS_ELEVATION.nc')

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

vbw = ValleyBottomWidth2.ValleyBottomWidth(swath_profiles.set_index(sample=('axis', 'measure', 'distance')), params, processes=8)
vbw.to_netcdf('../outputs/NETWORK/METRICS/WIDTH_VALLEY_BOTTOM.nc')


















from fct.profiles import ValleyBottomElevationProfile

params = ValleyBottomElevationProfile.Parameters()

ValleyBottomElevationProfile.ValleyBottomElevationProfile(params)







