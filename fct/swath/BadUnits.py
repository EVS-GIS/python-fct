from fct.config import config
from fct.swath.SwathMeasurement import VectorizeOneSwathPolygon, ValleyBottomParameters, SwathMeasurementParams, ReadSwathsBounds
import fiona
import fiona.crs
from shapely.geometry import asShape, Polygon, LinearRing

config.auto()

def BadUnits(axis, *gids):

    parameters = ValleyBottomParameters()
    params = SwathMeasurementParams(**parameters)

    defs = ReadSwathsBounds(axis, params)
    output = '/media/crousson/Backup2/RhoneMediterranee/AXES/AX%04d/WORK/UNITS.shp' % axis

    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Polygon',
        'properties': [('GID', 'int'), ('AXIS', 'int'), ('VALUE', 'int'), ('M', 'float')]}
    options = dict(driver=driver, crs=crs, schema=schema)

    for k, gid in enumerate(gids):

        measure, bounds = defs[gid]
        _, _, polygons = VectorizeOneSwathPolygon(axis, gid, measure, bounds, params)
        not_fixed = asShape(polygons[0][0])


        mode = 'w' if k == 0 else 'a'

        with fiona.open(output, mode, **options) as fst:
            
            fst.write({
                'geometry': Polygon(not_fixed.exterior).__geo_interface__,
                'properties': dict(AXIS=12, M=float(measure), GID=int(gid), VALUE=2)
            })

            for ring in not_fixed.interiors:
                
                fst.write({
                    'geometry': Polygon(ring).__geo_interface__,
                    'properties': dict(AXIS=12, M=float(measure), GID=int(gid), VALUE=2)
                })
