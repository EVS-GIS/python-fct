# Rééchantillonne la BD Alti 25 m à la résolution de 5 m,
# pour produire un MNT de même dimension que le RGE Alti 5 m
# extrait à l'étape précédente.
#
# Date: 2020-02-05

# cat ZONEHYDR/ZoneHydroAlti25m.list | while read BASSIN ZONE;
# do
#     if [ -f ZONEHYDR/$BASSIN/$ZONE/BDALTI25M.tif ]; then
#         rm ZONEHYDR/$BASSIN/$ZONE/BDALTI25M.tif
#     fi
# done

# cat ZONEHYDR/ZoneHydroRhoneAlpes.list | while read BASSIN ZONE;
# do
#     if ! [ -f ZONEHYDR/$BASSIN/$ZONE/DEM5M.tif ]; then
#         mv ZONEHYDR/$BASSIN/$ZONE/RGEALTI5M.tif ZONEHYDR/$BASSIN/$ZONE/DEM5M.tif
#     fi
# done

function extractdem {
    BASSIN=$1
    ZONE=$2
    SOURCE=/media/crousson/Backup/REFERENTIELS/IGN/BDALTI_25M/BDALTI25M.tif
    DESTINATION=ZONEHYDR/$BASSIN/$ZONE/BDALTI25M.tif
    gdalwarp -of GTiff -tr 25.0 -25.0 -tap -cutline \
        ZONEHYDR/$BASSIN/$ZONE/EMPRISE.shp -cl EMPRISE \
        -crop_to_cutline -co COMPRESS=DEFLATE \
        $SOURCE $DESTINATION
}

cat ZONEHYDR/ZoneHydroAlti25m.list | while read BASSIN ZONE;
do
    if ! [ -f ZONEHYDR/$BASSIN/$ZONE/BDALTI25M.tif ]; then
        extractdem $BASSIN $ZONE
    else
        echo 'Already exists !'
    fi
done

function upscaledem {
    BASSIN=$1
    ZONE=$2
    SOURCE=/media/crousson/Backup/REFERENTIELS/IGN/BDALTI_25M/BDALTI25M.tif
    DESTINATION=ZONEHYDR/$BASSIN/$ZONE/BDALTI_UPSCALED5M.tif
    gdalwarp -of GTiff -tr 5.0 -5.0 -r bilinear -tap -cutline \
        ZONEHYDR/$BASSIN/$ZONE/EMPRISE.shp -cl EMPRISE \
        -crop_to_cutline -co COMPRESS=DEFLATE \
        $SOURCE $DESTINATION
}

cat ZONEHYDR/ZoneHydroAlti25m.list | while read BASSIN ZONE;
do
    if ! [ -f ZONEHYDR/$BASSIN/$ZONE/BDALTI_UPSCALED5M.tif ]; then
        upscaledem $BASSIN $ZONE
    else
        # echo 'Already exists !'
        rm ZONEHYDR/$BASSIN/$ZONE/BDALTI_UPSCALED5M.tif
        rm ZONEHYDR/$BASSIN/$ZONE/DEM5M.tif
        upscaledem $BASSIN $ZONE
    fi
done