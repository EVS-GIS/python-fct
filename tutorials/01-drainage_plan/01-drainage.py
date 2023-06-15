# coding: utf-8

'''
Goals of step 01-drainage : 
    - Get a clean Flow Accumulation raster in km2
    - 
'''

# Create your tileset

from fct.cli import Tiles
Tiles.CreateTileset('bdalti', 10000.0)

# Prepare the DEM tiles and VRT

from fct.cli import Tiles
Tiles.DatasourceToTiles('bdalti', '10k', 'dem', processes=8)
Tiles.DatasourceToTiles('bdalti', '10kbis', 'dem', processes=8)

from fct.tileio import buildvrt
buildvrt('10k', 'dem')
buildvrt('10kbis', 'dem')

# If you have two different scales DEM, you can fill the precise one with the less precise
# First step when you have only one DEM : Smoothing
from fct.drainage import PrepareDEM
params = PrepareDEM.SmoothingParameters()
params.window=5

PrepareDEM.MeanFilter(params, overwrite=True, processes=8, tileset='10k')
PrepareDEM.MeanFilter(params, overwrite=True, processes=8, tileset='10kbis')

# Prepare hydrologic network
# get parameters
from fct.drainage import PrepareNetwork
params = PrepareNetwork.Parameters()
# network preparation with strahler order and buffer based on strahler
PrepareNetwork.PrepareStrahlerAndBuffer(params, buffer_factor=50, overwrite=True)
# create sources from network
PrepareNetwork.CreateSources(params, overwrite=True)

# Burn DEM resolved with buffered hydro network
# get parameters
from fct.drainage import Burn
params = Burn.Parameters()
params.elevations = 'smoothed'
# create buffer around hydro network 
Burn.HydroBuffer(params=params, overwrite=True)

# Burn resolved DEM with buffer
Burn.BurnBuffer(params=params, burn_delta = 10, overwrite=True, processes=8)
Burn.BurnBuffer(params=params, burn_delta = 10, overwrite=True, processes=8, tileset='10kbis')

# Fill sinks
from fct.drainage import DepressionFill
params = DepressionFill.Parameters()
params.elevations = 'burned-dem'
params.exterior_data = 9000.0
DepressionFill.LabelWatersheds(params, overwrite=True, processes=8)
DepressionFill.LabelWatersheds(params, overwrite=True, processes=8, tileset='10kbis')

DepressionFill.ResolveWatershedSpillover(params, overwrite=True)
DepressionFill.ResolveWatershedSpillover(params, overwrite=True, tileset='10kbis')

DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=8)
DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=8, tileset='10kbis')
 
# Resolve flats
from fct.drainage import BorderFlats
params = BorderFlats.Parameters()
BorderFlats.LabelBorderFlats(params=params, processes=8) 
BorderFlats.LabelBorderFlats(params=params, processes=8, tileset='10kbis') 
    
BorderFlats.ResolveFlatSpillover(params=params)
BorderFlats.ResolveFlatSpillover(params=params, tileset='10kbis')

BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=8)
BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=8, tileset='10kbis')
    
# FlatMap.DepressionDepthMap is useful if you want to check which flat areas have been resolved

# Flow direction
from fct.drainage import FlowDirection
params = FlowDirection.Parameters()
# params.exterior = 'off'
params.elevations = 'dem-drainage-resolved'

FlowDirection.exterior_mask(params)
FlowDirection.FlowDirection(params=params, overwrite=True, processes=8)
FlowDirection.FlowDirection(params=params, overwrite=True, processes=8, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'flow')
buildvrt('10kbis', 'flow')
buildvrt('10k', 'dem-drainage-resolved')
buildvrt('10kbis', 'dem-drainage-resolved')

# Flow tiles inlets/outlets graph
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=8)
Accumulate.Outlets(params=params, processes=8, tileset='10kbis')

Accumulate.AggregateOutlets(params)
Accumulate.AggregateOutlets(params, tileset='10kbis')

# Resolve inlets/outlets graph
Accumulate.InletAreas(params=params)
Accumulate.InletAreas(params=params, tileset='10kbis')

# Flow accumulation
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=8) 
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=8, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'acc')
buildvrt('10kbis', 'acc')

# Stream Network from sources
from fct.drainage import StreamSources
params = StreamSources.Parameters()

StreamSources.InletSources(params)
StreamSources.InletSources(params, tileset='10kbis')

StreamSources.StreamToFeatureFromSources(params, min_drainage=500, processes=8)
StreamSources.StreamToFeatureFromSources(params, min_drainage=500, processes=8, tileset='10kbis')

StreamSources.AggregateStreamsFromSources(params)
StreamSources.AggregateStreamsFromSources(params, tileset='10kbis')

# Find NoFlow pixels from RHTS
from fct.drainage import FixNoFlow
params = FixNoFlow.Parameters()

params.noflow = 'noflow'
params.fixed = 'noflow-targets'

FixNoFlow.DrainageRaster(params=params, processes=8)
FixNoFlow.DrainageRaster(params=params, processes=8, tileset='10kbis')
   
from fct.tileio import buildvrt 
buildvrt('10k', 'drainage-raster-from-sources')
buildvrt('10kbis', 'drainage-raster-from-sources')

FixNoFlow.NoFlowPixels(params=params, processes=8)
FixNoFlow.NoFlowPixels(params=params, processes=8, tileset='10kbis')

FixNoFlow.AggregateNoFlowPixels(params)
FixNoFlow.AggregateNoFlowPixels(params, tileset='10kbis')

# Fix NoFlow (create TARGETS shapefile and fix Flow Direction data)
FixNoFlow.FixNoFlow(params, tileset1='10k', tileset2='10kbis', fix=True)

# Remake FlowAccumulation with NoFlow pixels fixed
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=8)
Accumulate.AggregateOutlets(params)
Accumulate.InletAreas(params=params)
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=8) 

# Remake stream Network from sources with NoFlow pixels fixed
from fct.drainage import StreamSources
params = StreamSources.Parameters()

StreamSources.InletSources(params)
StreamSources.StreamToFeatureFromSources(params, min_drainage=500, processes=8)
StreamSources.AggregateStreamsFromSources(params)

# Identify network nodes
from fct.drainage import IdentifyNetworkNodes
params = IdentifyNetworkNodes.Parameters()

IdentifyNetworkNodes.IdentifyNetworkNodes(params)

#7 TODO: Update JoinNetworkAttributes
import os
if not os.path.isdir('../outputs/GLOBAL/MEASURE'):
    os.mkdir('../outputs/GLOBAL/MEASURE')

from fct.drainage import JoinNetworkAttributes
JoinNetworkAttributes.JoinNetworkAttributes('../inputs/sources.gpkg', '../outputs/GLOBAL/DEM/NETWORK_IDENTIFIED_10K.shp', '../outputs/GLOBAL/DEM/RHTS.shp')
# JoinNetworkAttributes.UpdateLengthOrder('../outputs/GLOBAL/DEM/RHTS.shp', '../outputs/GLOBAL/DEM/RHTS.shp')
JoinNetworkAttributes.AggregateByAxis('../outputs/GLOBAL/DEM/RHTS.shp', '../outputs/GLOBAL/MEASURE/REFAXIS.shp')
