# fct-tiles -c ./tutorials/dem_to_dgo/config.ini extract bdalti 10k dem
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini extract bdalti 10kbis dem

# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k dem
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10kbis dem

# If you have two different scales DEM, you can fill the precise one with the less precise
# First step when you have only one DEM : Smoothing
from fct.drainage import PrepareDEM
PrepareDEM.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = PrepareDEM.SmoothingParameters()
params.windows=25
for tile in PrepareDEM.config.tileset().tiles():
    PrepareDEM.MeanFilter(row=tile.row, col=tile.col, params=params)

# Fill sinks
from fct.drainage import DepressionFill
DepressionFill.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = DepressionFill.Parameters()
params.elevations = 'smoothed'
params.exterior_data = 0.0
for tile in DepressionFill.config.tileset().tiles():
    DepressionFill.LabelWatersheds(row=tile.row, col=tile.col, params=params, overwrite=True)
    
DepressionFill.ResolveWatershedSpillover(params, overwrite=True)

for tile in DepressionFill.config.tileset().tiles():
    DepressionFill.DispatchWatershedMinimumZ(row=tile.row, col=tile.col, params=params)

# Resolve flats
from fct.drainage import BorderFlats
BorderFlats.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = BorderFlats.Parameters()
for tile in BorderFlats.config.tileset().tiles():
    BorderFlats.LabelBorderFlats(row=tile.row, col=tile.col, params=params)
    
BorderFlats.ResolveFlatSpillover(params=params)

for tile in BorderFlats.config.tileset().tiles():
    BorderFlats.DispatchFlatMinimumZ(row=tile.row, col=tile.col, params=params, overwrite=True)
    
#FlatMap.DepressionDepthMap ?

# Flow direction
from fct.drainage import FlowDirection
FlowDirection.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = FlowDirection.Parameters()
params.exterior = 'off'
for tile in FlowDirection.config.tileset().tiles():
    FlowDirection.FlowDirectionTile(row=tile.row, col=tile.col, params=params, overwrite=True)

# Build VRT for flow direction tiles: fct-tiles -c tutorials/dem_to_dgo/config.ini buildvrt 10k dem-drainage-resolved

# Flow tiles inlets/outlets graph
from fct.drainage import Accumulate
Accumulate.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'

for tile in Accumulate.config.tileset().tiles():
    Accumulate.TileOutlets(row=tile.row, col=tile.col, params=params)

Accumulate.AggregateOutlets(params)

# Resolve inlets/outlets graph
Accumulate.InletAreas(params=params)

# Flow accumulation
for tile in Accumulate.config.tileset().tiles():
    Accumulate.FlowAccumulationTile(row=tile.row, col=tile.col, params=params, overwrite=True) 

################
# Stream Network from sources
from fct.drainage import StreamSources
StreamSources.config.from_file('./tutorials/dem_to_dgo/config.ini')

StreamSources.InletSources()

for tile in StreamSources.config.tileset().tiles():
    StreamSources.StreamToFeatureFromSources(row=tile.row, col=tile.col, min_drainage=500)

StreamSources.AggregateStreamsFromSources()

# Find NoFlow pixels
for tile in StreamSources.config.tileset().tiles():
    StreamSources.NoFlowPixels(row=tile.row, col=tile.col, min_drainage=500)
    
StreamSources.AggregateNoFlowPixels()

################
# Stream Network from DEM
# from fct.drainage import StreamNetwork
# StreamNetwork.config.from_file('./tutorials/dem_to_dgo/config.ini')
# params = StreamNetwork.Parameters()

# for tile in StreamNetwork.config.tileset().tiles():
#     StreamNetwork.StreamToFeatureTile(row=tile.row, col=tile.col, params=params)

# StreamNetwork.AggregateStreams(params)

##################################
# Fix NoFlow pixels (needs a second tileset)
# Build VRT for flow and acc rasters
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k flow
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k acc

from fct.drainage import FixNoFlow
FixNoFlow.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = FixNoFlow.Parameters()

for tile in FixNoFlow.config.tileset().tiles():
    FixNoFlow.DrainageRasterTile(row=tile.row, col=tile.col, params=params)
    
# Build VRT 
# fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k drainage-raster-from-sources

##############################
# Re-run all the beggining with the second tileset by modifing the default tileset in the config.ini file
##############################


for tile in FixNoFlow.config.tileset().tiles():
    FixNoFlow.NoFlowPixels(row=tile.row, col=tile.col, params=params)
    
FixNoFlow.AggregateNoFlowPixels(params)

from fct.drainage import FixNoFlow
FixNoFlow.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = FixNoFlow.Parameters()

import fiona
with fiona.open(params.noflow.filename(), 'r') as src:
    
    options = dict(driver=src.driver, crs=src.crs, schema=src.schema)
    
    with fiona.open(params.fixed.filename(), 'w', **options) as dst:
            for f in src:

                x, y = f['geometry']['coordinates']

                try:

                    tox, toy = FixNoFlow.FixNoFlow(x, y, '10k', '10kbis', params, fix=True)
                    f['geometry']['coordinates'] = [tox, toy]
                    dst.write(f)

                except ValueError as error:

                    print(f['properties']['GID'], error)
                    continue
        
################
# Restart the process from flow accumulation and skip FixNoFlow part
################

# Run identify Network Nodes with the QGIS Toolbox on RHTS_10K_NOATTR.shp and save the result in outputs/RHTS_Network.gpkg

from fct.drainage import JoinNetworkAttributes
JoinNetworkAttributes.JoinNetworkAttributes('./tutorials/dem_to_dgo/inputs/sources.gpkg', './tutorials/dem_to_dgo/outputs/RHTS_Network.gpkg', './tutorials/dem_to_dgo/outputs/RHTS.shp')
JoinNetworkAttributes.AggregateByAxis('./tutorials/dem_to_dgo/outputs/RHTS.shp', './tutorials/dem_to_dgo/outputs/GLOBAL/MEASURE/REFAXIS.shp')

################
# Multiprocessing example
from multiprocessing import Pool
import click

def starcall(args):
    """
    Invoke first arg function with all other arguments.
    """

    fun = args[0]
    return fun(*args[1:-1], **args[-1])

tile_index = [(tile.row, tile.col) for tile in Accumulate.config.tileset().tiles()]

kwargs = {'params': params}
arguments = ([Accumulate.TileOutlets, row, col, kwargs] for row, col in tile_index)

with Pool(processes=8) as pool:

    pooled = pool.imap_unordered(starcall, arguments)

    with click.progressbar(pooled, length=len(tile_index)) as bar:
        for _ in bar:
            pass
