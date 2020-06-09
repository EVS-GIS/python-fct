# Tiled DEM Processing Workflow

Before:

- Generate Tile Index
- Setup path in `config.ini`

```bash
# Fill Depressions and Label Watersheds
python Command.py prepare mktiles -j 4 -p --smooth 5
# python Command.py prepare boxes
python Command.py prepare fill -j 4 -p
python Command.py prepare spillover
python Command.py prepare applyz -j 6 -p
# burn
python Command.py flats labelflats -j 6 -p
python Command.py flats spillover
python Command.py flats applyminz -j 6 -p
python Command.py flats depthmap -j 6 -p
python Command.py flow calculate -j 4 -p
python Command.py flow outlets -j 6 -p
python Command.py flow aggregate
python Command.py drainage inletareas
python Command.py drainage accumulate -j 6 -p
python Command.py drainage vectorize -j 6 -p -a 5.0
python Command.py drainage aggregate
# watersheds
```