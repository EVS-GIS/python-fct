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


















from fct.profiles import ValleyBottomElevationProfile

params = ValleyBottomElevationProfile.Parameters()

ValleyBottomElevationProfile.ValleyBottomElevationProfile(params)







