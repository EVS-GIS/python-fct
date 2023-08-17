# coding: utf-8

'''
Goals of step 01-drainage : 
    - Get a clean Flow Accumulation raster in km2
    - 
'''

# Create your tileset

from fct.cli import Tiles
Tiles.CreateTileset('rgealti', 10000.0, 
                    tileset1 = '/data/sdunesme/fct/tests_1m/fct_workdir/10k_tileset.gpkg',
                    tileset2 = '/data/sdunesme/fct/tests_1m/fct_workdir/10kbis_tileset.gpkg')

# Prepare the DEM tiles and VRT

from fct.cli import Tiles
Tiles.DatasourceToTiles('rgealti', '10k', 'dem', processes=64)
Tiles.DatasourceToTiles('rgealti', '10kbis', 'dem', processes=64)

from fct.tileio import buildvrt
buildvrt('10k', 'dem')
buildvrt('10kbis', 'dem')

# If you have two different scales DEM, you can fill the precise one with the less precise
# First step when you have only one DEM : Smoothing
from fct.drainage import PrepareDEM
params = PrepareDEM.SmoothingParameters()
params.window=15

PrepareDEM.MeanFilter(params, overwrite=True, processes=64, tileset='10k')
PrepareDEM.MeanFilter(params, overwrite=True, processes=64, tileset='10kbis')

# Fill sinks
from fct.drainage import DepressionFill
params = DepressionFill.Parameters()
params.elevations = 'smoothed'
params.offset = -1.0 # burn offset in meters
params.exterior_data = 9000.0 # exterior value

DepressionFill.LabelWatersheds(params, overwrite=True, processes=64)
DepressionFill.LabelWatersheds(params, overwrite=True, processes=64, tileset='10kbis')

DepressionFill.ResolveWatershedSpillover(params, overwrite=True)
DepressionFill.ResolveWatershedSpillover(params, overwrite=True, tileset='10kbis')

DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=64)
DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=64, tileset='10kbis')
 
# Resolve flats
from fct.drainage import BorderFlats
params = BorderFlats.Parameters()
BorderFlats.LabelBorderFlats(params=params, processes=64) 
BorderFlats.LabelBorderFlats(params=params, processes=64, tileset='10kbis') 
    
BorderFlats.ResolveFlatSpillover(params=params)
BorderFlats.ResolveFlatSpillover(params=params, tileset='10kbis')

BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=64)
BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=64, tileset='10kbis')
    
# FlatMap.DepressionDepthMap is useful if you want to check which flat areas have been resolved

# Flow direction
from fct.drainage import FlowDirection
params = FlowDirection.Parameters()
params.exterior = 'off'
FlowDirection.FlowDirection(params=params, overwrite=True, processes=64)
FlowDirection.FlowDirection(params=params, overwrite=True, processes=64, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'flow')
buildvrt('10kbis', 'flow')
buildvrt('10k', 'dem-drainage-resolved')
buildvrt('10kbis', 'dem-drainage-resolved')

# Flow tiles inlets/outlets graph
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=64)
Accumulate.Outlets(params=params, processes=64, tileset='10kbis')

Accumulate.AggregateOutlets(params)
Accumulate.AggregateOutlets(params, tileset='10kbis')

# Resolve inlets/outlets graph
Accumulate.InletAreas(params=params)
Accumulate.InletAreas(params=params, tileset='10kbis')

# Flow accumulation
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=64) 
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=64, tileset='10kbis')

from fct.tileio import buildvrt
buildvrt('10k', 'acc')
buildvrt('10kbis', 'acc')

# Stream Network from sources
from fct.drainage import StreamSources

StreamSources.InletSources(params)
StreamSources.InletSources(params, tileset='10kbis')

StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=64)
StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=64, tileset='10kbis')

StreamSources.AggregateStreamsFromSources()
StreamSources.AggregateStreamsFromSources(tileset='10kbis')

# Find NoFlow pixels from RHTS
from fct.drainage import FixNoFlow
params = FixNoFlow.Parameters()

params.noflow = 'noflow'
params.fixed = 'noflow-targets'

FixNoFlow.DrainageRaster(params=params, processes=64)
FixNoFlow.DrainageRaster(params=params, processes=64, tileset='10kbis')
   
from fct.tileio import buildvrt 
buildvrt('10k', 'drainage-raster-from-sources')
buildvrt('10kbis', 'drainage-raster-from-sources')

FixNoFlow.NoFlowPixels(params=params, processes=64)
FixNoFlow.NoFlowPixels(params=params, processes=64, tileset='10kbis')

FixNoFlow.AggregateNoFlowPixels(params)
FixNoFlow.AggregateNoFlowPixels(params, tileset='10kbis')

# Fix NoFlow (create TARGETS shapefile and fix Flow Direction data)
FixNoFlow.FixNoFlow(params, tileset1='10k', tileset2='10kbis', fix=True)

# Remake FlowAccumulation with NoFlow pixels fixed
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=64)
Accumulate.AggregateOutlets(params)
Accumulate.InletAreas(params=params)
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=64) 

# Remake stream Network from sources with NoFlow pixels fixed
from fct.drainage import StreamSources

StreamSources.InletSources(params)
StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=64)
StreamSources.AggregateStreamsFromSources()

# Identify network nodes
from fct.drainage import IdentifyNetworkNodes
params = IdentifyNetworkNodes.Parameters()

IdentifyNetworkNodes.IdentifyNetworkNodes(params)

#7 TODO: Update JoinNetworkAttributes
import os
if not os.path.isdir('/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/MEASURE'):
    os.mkdir('/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/MEASURE')

from fct.drainage import JoinNetworkAttributes
JoinNetworkAttributes.JoinNetworkAttributes('/data/sdunesme/fct/tests_1m/inputs/sources.gpkg', '/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/DEM/NETWORK_IDENTIFIED_10K.shp', '/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/DEM/RHTS.shp')
# JoinNetworkAttributes.UpdateLengthOrder('/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/DEM/RHTS.shp', '/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/DEM/RHTS.shp')
JoinNetworkAttributes.AggregateByAxis('/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/DEM/RHTS.shp', '/data/sdunesme/fct/tests_1m/fct_workdir/GLOBAL/MEASURE/REFAXIS.shp')
