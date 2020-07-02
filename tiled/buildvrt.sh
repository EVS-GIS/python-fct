# Global Datasets

find LANDCOVER -name "CESBIO_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 LANDCOVER_2018.vrt
find POPULATION -name "POP_INSEE_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 POP_2015.vrt
find ACC -name "POP_INSEE_ACC_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 POP_2015_ACC.vrt
find ACC -name "CESBIO_ACC_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 LANDCOVER_2018_ACC.vrt

# Per axis Datasets


find TILES -name "FLOW_DIST_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 FLOW_DIST.vrt
find TILES -name "FLOW_RELZ_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 FLOW_RELZ.vrt
find TILES -name "AXIS_MEASURE_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 AXIS_MEASURE.vrt
find TILES -name "AXIS_DISTANCE_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 AXIS_DISTANCE.vrt
find TILES -name "DGO_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 DGO.vrt
find TILES -name "NEAREST_RELZ_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 NEAREST_RELZ.vrt
find TILES -name "NEAREST_DISTANCE_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 NEAREST_DISTANCE.vrt

find TILES -name "CONTINUITY_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 CONTINUITY.vrt

find TILES -name "BUFFER30_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 BUFFER30.vrt
find TILES -name "BUFFER100_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 BUFFER100.vrt
find TILES -name "BUFFER200_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 BUFFER200.vrt
find TILES -name "BUFFER1000_*.tif" | xargs gdalbuildvrt -a_srs EPSG:2154 BUFFER1000.vrt
