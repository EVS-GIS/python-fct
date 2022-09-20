from fct.drainage import PrepareDEM, FlowDirection, Accumulate


# First step when you have only one DEM : Smoothing
PrepareDEM.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = PrepareDEM.SmoothingParameters()
params.windows=25
for tile in PrepareDEM.config.tileset().tiles():
    PrepareDEM.MeanFilter(row=tile.row, col=tile.col, params=params)

# Resolve flat if needed

# Flow direction
FlowDirection.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = FlowDirection.Parameters()
params.elevations = 'smoothed'
for tile in Accumulate.config.tileset().tiles():
    FlowDirection.FlowDirectionTile(row=tile.row, col=tile.col, params=params, overwrite=True)
    
# Flow accumulation
Accumulate.config.from_file('./tutorials/dem_to_dgo/config.ini')
params = Accumulate.Parameters()
for tile in Accumulate.config.tileset().tiles():
    Accumulate.FlowAccumulationTile(row=tile.row, col=tile.col, params=params, overwrite=True)