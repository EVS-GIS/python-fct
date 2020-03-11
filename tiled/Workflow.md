# Tiled DEM Processing Workflow

Before:

- Generate Tile Index
- Setup path in `config.ini`

```bash
# Fill Depressions and Label Watersheds
python PreProcessing.py batch -j 4
python PreProcessing.py boxes
python PreProcessing.py spillover -w
python PreProcessing.py finalize -j 2 -w
python FlowDirection.py flow -j 4 -w
python FlowDirection.py aggregate -w
python StreamNetwork.py areas -w
python StreamNetwork.py accumulate -j 4 -w
python StreamNetwork.py aggregate -w
```