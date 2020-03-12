# Tiled DEM Processing Workflow

Before:

- Generate Tile Index
- Setup path in `config.ini`

```bash
# Fill Depressions and Label Watersheds
python PreProcessing.py patch -j 4
# burn
python PreProcessing.py boxes
python PreProcessing.py spillover -w
python PreProcessing.py finalize -j 2 -w
python Command.py flats labelflats -j 4 -p
python Command.py flats spillover
python Command.py flats applyminz -j 4 -p
python Command.py flow calculate -j 4 -p
python Command.py flow aggregate
python Command.py drainage inletareas
python Command.py drainage accumulate -j 4 -p
python Command.py drainage vectorize -j 4 -p -a 5.0
python Command.py drainage aggregate
# watersheds
```