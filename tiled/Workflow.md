# Tiled DEM Processing Workflow

Before:

- Generate Tile Index
- Setup path in `config.ini`

```bash
# Fill Depressions and Label Watersheds
python PreProcessing.py batch -j 4
python PreProcessing.py boxes
python PreProcessing.py spillover
python PreProcessing.py finalize -j 2
python FlowDirection.py batch -j 4 -w
python FlowDirection.py aggregate
python StreamNetwork.py areas
python StreamNetwork.py batch -j 4 -w
python StreamNetwork.py aggregate
```