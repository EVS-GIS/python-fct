from fct.drainage import PrepareDEM, DepressionFill, BorderFlats, FlowDirection, StreamNetwork, Accumulate


# If you have two different scales DEM, you can fill the precise one with the less precise
# First step when you have only one DEM : Smoothing
PrepareDEM.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = PrepareDEM.SmoothingParameters()
params.windows=25
for tile in PrepareDEM.config.tileset().tiles():
    PrepareDEM.MeanFilter(row=tile.row, col=tile.col, params=params)

# Fill sinks
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
BorderFlats.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = BorderFlats.Parameters()
for tile in BorderFlats.config.tileset().tiles():
    BorderFlats.LabelBorderFlats(row=tile.row, col=tile.col, params=params)
    
BorderFlats.ResolveFlatSpillover(params=params)

for tile in BorderFlats.config.tileset().tiles():
    BorderFlats.DispatchFlatMinimumZ(row=tile.row, col=tile.col, params=params, overwrite=True)
    
#FlatMap.DepressionDepthMap ?

# Flow direction
FlowDirection.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = FlowDirection.Parameters()
params.exterior = 'off'
for tile in FlowDirection.config.tileset().tiles():
    FlowDirection.FlowDirectionTile(row=tile.row, col=tile.col, params=params, overwrite=True)

#TODO: Check if this part is facultative
# Flow tiles outlets
Accumulate.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = Accumulate.Parameters()
params.elevations = 'dem-drainage-resolved'
for tile in Accumulate.config.tileset().tiles():
    Accumulate.TileOutlets(row=tile.row, col=tile.col, params=params)

Accumulate.AggregateOutlets(params)

# Flow accumulation
for tile in Accumulate.config.tileset().tiles():
    Accumulate.FlowAccumulationTile(row=tile.row, col=tile.col, params=params, overwrite=True) 
    
# Stream Network
#TODO: This part return an empty shapefile, check what appends
StreamNetwork.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = StreamNetwork.Parameters()

for tile in StreamNetwork.config.tileset().tiles():
    StreamNetwork.StreamToFeatureTile(row=tile.row, col=tile.col, params=params)

StreamNetwork.AggregateStreams(params)

# Label and fix NoFlow if needed
# Stream network from DEM

# End of facultative

# Beginning of the by-axis processing

