# coding: utf-8

'''
Goals of step 01-drainage : 
    - Get a clean Flow Accumulation raster in km2
    - 
'''

core = 8

# Create your tileset

from fct.cli import Tiles
Tiles.CreateTileset('bdalti', 10000.0)

# Prepare the DEM tiles and VRT

from fct.cli import Tiles
Tiles.DatasourceToTiles('bdalti', '10k', 'dem', processes=core)
Tiles.DatasourceToTiles('bdalti', '10kbis', 'dem', processes=core)

from fct.tileio import buildvrt
buildvrt('10k', 'dem')
buildvrt('10kbis', 'dem')

# If you have two different scales DEM, you can fill the precise one with the less precise
# First step when you have only one DEM : Smoothing
from fct.drainage import PrepareDEM
params = PrepareDEM.SmoothingParameters()
params.window=5

PrepareDEM.MeanFilter(params, overwrite=True, processes=core, tileset='10k')
PrepareDEM.MeanFilter(params, overwrite=True, processes=core, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'smoothed')
buildvrt('10kbis', 'smoothed')

# Prepare hydrologic network
from fct.drainage import PrepareNetwork
params = PrepareNetwork.Parameters()
# Copy input reference network to compute Strahler order with any network
PrepareNetwork.CopyRefHydroNetwork(params)
# Compute Strahler order on input network
PrepareNetwork.StrahlerOrder(params, overwrite=True)
# Add buffer field on network based on Strahler order
PrepareNetwork.BufferFieldOnStrahler(params, buffer_factor=50, overwrite=True)
# create sources from network
PrepareNetwork.CreateSources(params, overwrite=True)
# create sources and confluences file from network
PrepareNetwork.CreateSourcesAndConfluences(params, node_id_field = 'GID', axis_field = 'liens_vers', hydro_id_field='code_hydro', toponym_field='cpx_topony', overwrite=True)

# Burn DEM resolved with buffered hydro network
# get parameters
from fct.drainage import Burn
params = Burn.Parameters()
params.elevations = 'smoothed'
# create buffer around hydro network 
Burn.HydroBuffer(params=params, overwrite=True)

# Burn resolved DEM with buffer
Burn.BurnBuffer(params=params, burn_delta = 20, overwrite=True, processes=core)
Burn.BurnBuffer(params=params, burn_delta = 20, overwrite=True, processes=core, tileset='10kbis')


#######################
core=8
from fct.drainage import Burn
params = Burn.Parameters()
params.elevations = 'dem'
Burn.BurnLines(params, burn_delta=50, overwrite=True, processes=core)
Burn.BurnLines(params, burn_delta=50, overwrite=True, processes=core, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'burned-dem')
buildvrt('10kbis', 'burned-dem')


#######################
from fct.tileio import buildvrt
buildvrt('10k', 'burned-dem')
buildvrt('10kbis', 'burned-dem')

# Fill sinks
from fct.drainage import DepressionFill
params = DepressionFill.Parameters()
params.elevations = 'burned-dem'
params.exterior_data = 9000.0
DepressionFill.LabelWatersheds(params, overwrite=True, processes=core)
DepressionFill.LabelWatersheds(params, overwrite=True, processes=core, tileset='10kbis')

DepressionFill.ResolveWatershedSpillover(params, overwrite=True)
DepressionFill.ResolveWatershedSpillover(params, overwrite=True, tileset='10kbis')

DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=core)
DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=core, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'dem-filled-resolved')
buildvrt('10kbis', 'dem-filled-resolved')
 
# Resolve flats
from fct.drainage import BorderFlats
params = BorderFlats.Parameters()
BorderFlats.LabelBorderFlats(params=params, processes=core) 
BorderFlats.LabelBorderFlats(params=params, processes=core, tileset='10kbis') 
    
BorderFlats.ResolveFlatSpillover(params=params)
BorderFlats.ResolveFlatSpillover(params=params, tileset='10kbis')

BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=core)
BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=core, tileset='10kbis')
    
# FlatMap.DepressionDepthMap is useful if you want to check which flat areas have been resolved

# Flow direction
from fct.drainage import FlowDirection
params = FlowDirection.Parameters()
# params.exterior = 'off'
params.elevations = 'dem-drainage-resolved'

FlowDirection.exterior_mask(params)
FlowDirection.exterior_mask(params, tileset='10kbis')
FlowDirection.FlowDirection(params=params, overwrite=True, processes=core)
FlowDirection.FlowDirection(params=params, overwrite=True, processes=core, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'flow')
buildvrt('10kbis', 'flow')
buildvrt('10k', 'dem-drainage-resolved')
buildvrt('10kbis', 'dem-drainage-resolved')

# Flow tiles inlets/outlets graph
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=core)
Accumulate.Outlets(params=params, processes=core, tileset='10kbis')

Accumulate.AggregateOutlets(params)
Accumulate.AggregateOutlets(params, tileset='10kbis')

# Resolve inlets/outlets graph
Accumulate.InletAreas(params=params)
Accumulate.InletAreas(params=params, tileset='10kbis')

# Flow accumulation
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=core) 
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=core, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'acc')
buildvrt('10kbis', 'acc')

# Stream Network from sources
from fct.drainage import StreamSources
params = StreamSources.Parameters()

StreamSources.InletSources(params)
StreamSources.InletSources(params, tileset='10kbis')

StreamSources.StreamToFeatureFromSources(params, min_drainage=500, processes=core)
StreamSources.StreamToFeatureFromSources(params, min_drainage=500, processes=core, tileset='10kbis')

StreamSources.AggregateStreamsFromSources(params)
StreamSources.AggregateStreamsFromSources(params, tileset='10kbis')

# Find NoFlow pixels from RHTS
from fct.drainage import FixNoFlow
params = FixNoFlow.Parameters()

params.noflow = 'noflow'
params.fixed = 'noflow-targets'

FixNoFlow.DrainageRaster(params=params, processes=core)
FixNoFlow.DrainageRaster(params=params, processes=core, tileset='10kbis')
   
from fct.tileio import buildvrt 
buildvrt('10k', 'drainage-raster-from-sources')
buildvrt('10kbis', 'drainage-raster-from-sources')

FixNoFlow.NoFlowPixels(params=params, processes=core)
FixNoFlow.NoFlowPixels(params=params, processes=core, tileset='10kbis')

FixNoFlow.AggregateNoFlowPixels(params)
FixNoFlow.AggregateNoFlowPixels(params, tileset='10kbis')

# Fix NoFlow (create TARGETS shapefile and fix Flow Direction data)
FixNoFlow.FixNoFlow(params, tileset1='10k', tileset2='10kbis', fix=True)

# Remake FlowAccumulation with NoFlow pixels fixed
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=core)
Accumulate.AggregateOutlets(params)
Accumulate.InletAreas(params=params)
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=core) 

####### RHTS #######

# Remake stream Network from sources with NoFlow pixels fixed
from fct.drainage import StreamSources
params = StreamSources.Parameters()

StreamSources.InletSources(params)
StreamSources.StreamToFeatureFromSources(params, min_drainage=500, processes=core)
StreamSources.AggregateStreamsFromSources(params)

# Identify network nodes
from fct.drainage import IdentifyNetworkNodes
params = IdentifyNetworkNodes.Parameters()

IdentifyNetworkNodes.IdentifyNetworkNodes(params)

# compute Strahler order on RHTS
from fct.drainage import PrepareNetwork
params = PrepareNetwork.Parameters()
params.hydro_network = 'network-identified'
params.hydrography_strahler= 'network-identified-strahler'
# Compute Strahler order on input network
PrepareNetwork.StrahlerOrder(params, tileset='default', overwrite=True)

#7 TODO: Update JoinNetworkAttributes
# yml global measure
# preparenetwork source and confluence
# JoinNetworkAttributes get attribut from sourceconfluence get axis from cours d'eau id
# AggregateByAxis 
import os
if not os.path.isdir('../outputs/GLOBAL/MEASURE'):
    os.mkdir('../outputs/GLOBAL/MEASURE')

from fct.drainage import JoinNetworkAttributes_lm
params = JoinNetworkAttributes_lm.Parameters()
JoinNetworkAttributes_lm.JoinNetworkAttributes(params)

JoinNetworkAttributes.JoinNetworkAttributes('../inputs/sources.gpkg', '../outputs/GLOBAL/DEM/NETWORK_IDENTIFIED_10K.shp', '../outputs/GLOBAL/DEM/RHTS.shp')
# JoinNetworkAttributes.UpdateLengthOrder('../outputs/GLOBAL/DEM/RHTS.shp', '../outputs/GLOBAL/DEM/RHTS.shp')
JoinNetworkAttributes.AggregateByAxis('../outputs/GLOBAL/DEM/RHTS.shp', '../outputs/GLOBAL/MEASURE/REFAXIS.shp')
