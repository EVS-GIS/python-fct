import os
from fct.config import config
from multiprocessing import cpu_count
p = cpu_count()/2

# Create your tileset
from fct.cli import Tiles
Tiles.CreateTileset('dem5m', 30000.0, 
                    tileset1 = os.path.join(config.workdir, 'default_tileset.gpkg'),
                    tileset2 = os.path.join(config.workdir, 'aux_tileset.gpkg'))

# ATTENTION : Bien vérifier que les tilesets couvrent bien tout le DEM avant de continuer

# Copy the Hydrographic Reference to outputs/GLOBAL/REFHYDRO
if not os.path.isdir(os.path.join(config.workdir, 'GLOBAL/REFHYDRO/')):
    os.makedirs(os.path.join(config.workdir, 'GLOBAL/REFHYDRO/'))

# COPY input REFHYDRO to /local/sdunesme/run_rmc_5m/GLOBAL/REFHYDRO/REFERENTIEL_HYDRO.shp

# Prepare the DEM tiles and VRT
from fct.cli import Tiles
Tiles.DatasourceToTiles('dem5m', 'default', 'dem', processes=p) # Environ 120G pour 32*20km
Tiles.DatasourceToTiles('dem5m', 'aux', 'dem', processes=p)

from fct.tileio import buildvrt
buildvrt('default', 'dem')
buildvrt('aux', 'dem')

# First step when you have only one DEM : Smoothing
from fct.drainage import PrepareDEM
params = PrepareDEM.SmoothingParameters()
#params.window=15 # Uncomment for 1m resolution DEM

PrepareDEM.MeanFilter(params, overwrite=True, processes=p, tileset='default')
PrepareDEM.MeanFilter(params, overwrite=True, processes=p, tileset='aux')

from fct.tileio import buildvrt
buildvrt('default', 'smoothed')
buildvrt('aux', 'smoothed')

# Drape hydrography network
from fct.drainage import Drape
params = Drape.Parameters()
#params.elevations = 'smoothed'
params.stream_network = 'network-cartography-ready'
params.draped = 'stream-network-draped'
Drape.DrapeNetworkAndAdjustElevations(params)
Drape.SplitStreamNetworkIntoTiles(params, tileset='default')
Drape.SplitStreamNetworkIntoTiles(params, tileset='aux')

# Fill sinks
from fct.drainage import DepressionFill
params = DepressionFill.Parameters()
#params.elevations = 'smoothed'
params.offset = 1.0 # burn offset in meters
params.exterior_data = 9000.0 # exterior value

DepressionFill.LabelWatersheds(params, overwrite=True, processes=p)
DepressionFill.LabelWatersheds(params, overwrite=True, processes=p, tileset='aux')

from fct.tileio import buildvrt
buildvrt('default', 'dem-watershed-labels')
buildvrt('aux', 'dem-watershed-labels')

DepressionFill.ResolveWatershedSpillover(params, overwrite=True)
DepressionFill.ResolveWatershedSpillover(params, overwrite=True, tileset='aux')

DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=p)
DepressionFill.DispatchWatershedMinimumZ(params, overwrite=True, processes=p, tileset='aux')

from fct.tileio import buildvrt
buildvrt('default', 'dem-filled-resolved')
buildvrt('aux', 'dem-filled-resolved')

# Resolve flats
from fct.drainage import BorderFlats
params = BorderFlats.Parameters()
BorderFlats.LabelBorderFlats(params=params, processes=p) 
BorderFlats.LabelBorderFlats(params=params, processes=p, tileset='aux') 

from fct.tileio import buildvrt
buildvrt('default', 'dem-flat-labels')
buildvrt('aux', 'dem-flat-labels')
    
BorderFlats.ResolveFlatSpillover(params=params)
BorderFlats.ResolveFlatSpillover(params=params, tileset='aux')

BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=p)
BorderFlats.DispatchFlatMinimumZ(params=params, overwrite=True, processes=p, tileset='aux')

from fct.tileio import buildvrt
buildvrt('default', 'dem-drainage-resolved')
buildvrt('aux', 'dem-drainage-resolved')

# FlatMap.DepressionDepthMap is useful if you want to check which flat areas have been resolved

# Flow direction
from fct.drainage import FlowDirection
params = FlowDirection.Parameters()
params.exterior = 'off'
FlowDirection.FlowDirection(params=params, overwrite=True, processes=p)
FlowDirection.FlowDirection(params=params, overwrite=True, processes=p, tileset='aux')

from fct.tileio import buildvrt
buildvrt('default', 'flow')
buildvrt('aux', 'flow')

# Flow tiles inlets/outlets graph
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

Accumulate.Outlets(params=params, processes=p)
Accumulate.Outlets(params=params, processes=p, tileset='aux')

Accumulate.AggregateOutlets(params)
Accumulate.AggregateOutlets(params, tileset='aux')

# Resolve inlets/outlets graph
Accumulate.InletAreas(params=params)
Accumulate.InletAreas(params=params, tileset='aux')

# Flow accumulation
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=p) 
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=p, tileset='aux')

from fct.tileio import buildvrt
buildvrt('default', 'acc')
buildvrt('aux', 'acc')

# Stream Network from sources
from fct.drainage import StreamSources

StreamSources.InletSources(params)
StreamSources.InletSources(params, tileset='aux')

StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=p)
StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=p, tileset='aux')

StreamSources.AggregateStreamsFromSources()
StreamSources.AggregateStreamsFromSources(tileset='aux')

# Find NoFlow pixels from RHTS
from fct.drainage import FixNoFlow
params = FixNoFlow.Parameters()

params.noflow = 'noflow'
params.fixed = 'noflow-targets'

FixNoFlow.DrainageRaster(params=params, processes=p)
FixNoFlow.DrainageRaster(params=params, processes=p, tileset='aux')
   
from fct.tileio import buildvrt 
buildvrt('default', 'drainage-raster-from-sources')
buildvrt('aux', 'drainage-raster-from-sources')

FixNoFlow.NoFlowPixels(params=params, processes=p)
FixNoFlow.NoFlowPixels(params=params, processes=p, tileset='aux')

FixNoFlow.AggregateNoFlowPixels(params)
FixNoFlow.AggregateNoFlowPixels(params, tileset='aux')

# Fix NoFlow (create TARGETS shapefile and fix Flow Direction data)
FixNoFlow.FixNoFlow(params, tileset1='default', tileset2='aux', fix=True)
FixNoFlow.FixNoFlow(params, tileset1='aux', tileset2='default', fix=True)

# Remake FlowAccumulation with NoFlow pixels fixed
# Il est possible de déplacer un peu les sources à la main et relancer à partir d'ici pour recalculer le RHTS
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'
# params.exterior_flow = 'exterior-inlet' # Uncomment this if you have exterior EXTERIOR INLET points 

Accumulate.Outlets(params=params, processes=p, tileset='default')
Accumulate.AggregateOutlets(params, tileset='default')
Accumulate.InletAreas(params=params, tileset='default')
Accumulate.FlowAccumulation(params=params, overwrite=True, processes=p, tileset='default') 

# Remake stream Network from sources with NoFlow pixels fixed
from fct.drainage import Accumulate
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'
# params.exterior_flow = 'exterior-inlet' # Uncomment this if you have exterior EXTERIOR INLET points 
from fct.drainage import StreamSources

StreamSources.InletSources(params, tileset='default')
StreamSources.StreamToFeatureFromSources(min_drainage=500, processes=p, tileset='default')
StreamSources.AggregateStreamsFromSources(tileset='default')

# IF NETWORK IS DISCONNECTED, TRY TO RESTART THE WORKFLOW FROM "Remake flow accumulation" WITH THE OTHER TILESET

# Identify network nodes
from fct.drainage import IdentifyNetworkNodes
params = IdentifyNetworkNodes.Parameters()

IdentifyNetworkNodes.IdentifyNetworkNodes(params, tileset='default')
IdentifyNetworkNodes.JoinSourcesAttributes(params, tileset='default')

# Join network attributes and aggregate by axis
import os
if not os.path.isdir(f'{config.workdir}/GLOBAL/MEASURE'):
    os.mkdir(f'{config.workdir}/GLOBAL/MEASURE')

from fct.drainage import JoinNetworkAttributes
JoinNetworkAttributes.JoinNetworkAttributes(f'{config.workdir}/GLOBAL/DEM/SOURCES_IDENTIFIED_DEFAULT.shp', f'{config.workdir}/GLOBAL/DEM/NETWORK_IDENTIFIED_DEFAULT.shp', f'{config.workdir}/GLOBAL/DEM/RHTS.shp')
# JoinNetworkAttributes.UpdateLengthOrder(f'{config.workdir}/GLOBAL/DEM/RHTS.shp', f'{config.workdir}/GLOBAL/DEM/RHTS.shp')
JoinNetworkAttributes.AggregateByAxis(f'{config.workdir}/GLOBAL/DEM/RHTS.shp', f'{config.workdir}/GLOBAL/MEASURE/REFAXIS.shp')
