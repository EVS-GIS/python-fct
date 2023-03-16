# SwathProfile
from fct.profiles import SwathProfile
params = SwathProfile.Parameters()
params.swaths = 'swaths_medialaxis'

swath_profiles = SwathProfile.SwathProfile(params, processes=8)
swath_profiles.to_netcdf('../outputs/NETWORK/METRICS/SWATHS_ELEVATION.nc')

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







