# From DEM to DGO

Frist, create a tileset with qgiscreategrid. Create GID, ROW, COL, X0 and Y0 fields 
<!-- Modifier le nom des champs dans le code -->

Create config.ini
Tile DEM
fct-tiles -c ./tutorials/dem_to_dgo/config.ini extract bdalti 10k dem

Build VRT
fct-tiles -c ./tutorials/dem_to_dgo/config.ini buildvrt 10k dem
<!-- Non fonctionnel sous windows (commande find + xargs) -->
 