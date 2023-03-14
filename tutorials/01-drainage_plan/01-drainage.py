# coding: utf-8

'''
Goals of step 01-drainage : 
    - Get a clean Flow Accumulation raster in km2
    - 
'''

# Prepare the DEM tiles and VRT

'''bash
fct-tiles -c ./config.ini extract bdalti 10k dem
fct-tiles -c ./config.ini extract bdalti 10kbis dem

fct-tiles -c ./config.ini buildvrt 10k dem
fct-tiles -c ./config.ini buildvrt 10kbis dem
'''

# If you have two different scales DEM, you can fill the precise one with the less precise
# First step when you have only one DEM : Smoothing
from fct.drainage import PrepareDEM
params = PrepareDEM.SmoothingParameters()
params.window=2

PrepareDEM.MeanFilter(params, overwrite=True, processes=4, tileset='10k')
PrepareDEM.MeanFilter(params, overwrite=True, processes=4, tileset='10kbis')

# Fill sinks
from fct.drainage import DepressionFill
params = DepressionFill.Parameters()
params.elevations = 'smoothed'
params.exterior_data = 0.0
DepressionFill.LabelWatersheds(params, overwrite=True, processes=4)
DepressionFill.LabelWatersheds(params, overwrite=True, processes=4, tileset='10kbis')

DepressionFill.ResolveWatershedSpillover(params, overwrite=True)
DepressionFill.ResolveWatershedSpillover(params, overwrite=True, tileset='10kbis')

DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=4)
DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=4, tileset='10kbis')
 
# Resolve flats
from fct.drainage import BorderFlats
params = BorderFlats.Parameters()
BorderFlats.LabelBorderFlats(params=params, processes=4) 
BorderFlats.LabelBorderFlats(params=params, processes=4, tileset='10kbis') 
    
BorderFlats.ResolveFlatSpillover(params=params)
BorderFlats.ResolveFlatSpillover(params=params, tileset='10kbis')

BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=4)
BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=4, tileset='10kbis')
    
#FlatMap.DepressionDepthMap ?

# Flow direction
from fct.drainage import FlowDirection
params = FlowDirection.Parameters()
params.exterior = 'off'
FlowDirection.FlowDirection(params=params, overwrite=True, processes=4)
FlowDirection.FlowDirection(params=params, overwrite=True, processes=4, tileset='10kbis')

'''bash
fct-tiles -c ./config.ini buildvrt 10k flow
fct-tiles -c ./config.ini buildvrt 10kbis flow
'''

# Flow tiles inlets/outlets graph
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=4)
Accumulate.Outlets(params=params, processes=4, tileset='10kbis')

Accumulate.AggregateOutlets(params)
Accumulate.AggregateOutlets(params, tileset='10kbis')

'''bash
fct-tiles -c ./config.ini buildvrt 10k dem-drainage-resolved
fct-tiles -c ./config.ini buildvrt 10kbis dem-drainage-resolved
'''

# Resolve inlets/outlets graph
Accumulate.InletAreas(params=params)
Accumulate.InletAreas(params=params, tileset='10kbis')

# Flow accumulation
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=4) 
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=4, tileset='10kbis')

'''bash
fct-tiles -c ./config.ini buildvrt 10k acc
fct-tiles -c ./config.ini buildvrt 10kbis acc
'''

# Stream Network from sources
from fct.drainage import StreamSources

StreamSources.InletSources(params)
StreamSources.InletSources(params, tileset='10kbis')

StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=4)
StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=4, tileset='10kbis')

StreamSources.AggregateStreamsFromSources()
StreamSources.AggregateStreamsFromSources(tileset='10kbis')

# Find NoFlow pixels
StreamSources.NoFlowPixels(min_drainage=500, processes=4)
StreamSources.NoFlowPixels(min_drainage=500, processes=4, tileset='10kbis')
    
StreamSources.AggregateNoFlowPixels()
StreamSources.AggregateNoFlowPixels(tileset='10kbis')

# Fix NoFlow pixels ok 10k tileset witk 10kbis tileset
from fct.drainage import FixNoFlow
params = FixNoFlow.Parameters()

FixNoFlow.DrainageRaster(params=params, processes=4)
FixNoFlow.DrainageRaster(params=params, processes=4, tileset='10kbis')
    
'''bash
fct-tiles -c ./config.ini buildvrt 10k drainage-raster-from-sources
fct-tiles -c ./config.ini buildvrt 10kbis drainage-raster-from-sources
'''

FixNoFlow.NoFlowPixels(params=params, processes=4)
FixNoFlow.NoFlowPixels(params=params, processes=4, tileset='10kbis')

FixNoFlow.AggregateNoFlowPixels(params)
FixNoFlow.AggregateNoFlowPixels(params, tileset='10kbis')

FixNoFlow.FixNoFlow(params, tileset1='default', tileset2='10kbis', fix=True)

# Remake FlowAccumulation with NoFlow pixels fixed
from fct.drainage import Accumulate
params = Accumulate.Parameters()

Accumulate.FlowAccumulation(params=params, overwrite=True, processes=4) 
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=4, tileset='10kbis')

# Remake stream Network from sources with NoFlow pixels fixed
from fct.drainage import StreamSources

StreamSources.InletSources(params)
StreamSources.InletSources(params, tileset='10kbis')

StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=4)
StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=4, tileset='10kbis')

StreamSources.AggregateStreamsFromSources()
StreamSources.AggregateStreamsFromSources(tileset='10kbis')

# Run identify Network Nodes with the QGIS Toolbox on RHTS_10K_NOATTR.shp and save the result in outputs/RHTS_Network.gpkg

# TODO: Fix longest path finding

from fct.drainage import JoinNetworkAttributes
JoinNetworkAttributes.JoinNetworkAttributes('./inputs/sources.gpkg', './outputs/RHTS_Network.gpkg', './outputs/RHTS.shp')
JoinNetworkAttributes.AggregateByAxis('./outputs/RHTS.shp', './outputs/GLOBAL/MEASURE/REFAXIS.shp')
