import fiona
import fiona.crs
import click
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

from ..config import config, DatasetParameter, LiteralParameter

class Parameters:
    """
    Identify Network Nodes parameters
    """
    
    network = DatasetParameter('input network features', type='input')
    quantization = LiteralParameter('Quantization factor for node coordinates')
    network_identified = DatasetParameter('Output network with identified nodes', type='output')
    nodes = DatasetParameter('Output nodes', type='output')
    
    def __init__(self):
        """
        Default parameter values
        """

        self.network = 'streams-from-sources'
        self.quantization = 1e8
        self.network_identified = 'network-identified'
        self.nodes = 'network-nodes'
        
        
def IdentifyNetworkNodes(params, tileset='default'):
    """
    Identifies network nodes by finding the endpoints of lines in a given network dataset and 
    quantizing their coordinates. The nodes are output as a separate dataset and their 
    attributes are added to the input network dataset. 
    
    Parameters
    ----------
    params : Parameters
        Input parameters.
    tileset : str, optional
        The tileset to use for the input and output datasets. Default is 'default'.
        
    Returns
    -------
    None
    
    Raises
    ------
    None
    """
    # Step 1
    click.secho('Get lines endpoints', fg='yellow')
    coordinates = list()
    
    def extract_coordinates(polyline):
        """ Extract endpoints coordinates
        """

        a = polyline['coordinates'][0]
        b = polyline['coordinates'][-1]
        coordinates.append(tuple(a))
        coordinates.append(tuple(b))
        
    with fiona.open(params.network.filename(tileset=tileset)) as fs:

        with click.progressbar(fs) as processing:
            for feature in processing:
                
                extract_coordinates(feature['geometry'])
                
        # Step 2
        click.secho('Quantize coordinates', fg='yellow')
        
        coordinates = np.array(coordinates)
        minx = np.min(coordinates[:, 0])
        miny = np.min(coordinates[:, 1])
        maxx = np.max(coordinates[:, 0])
        maxy = np.max(coordinates[:, 1])

        quantization = 1e8
        kx = (minx == maxx) and 1 or (maxx - minx)
        ky = (miny == maxy) and 1 or (maxy - miny)
        sx = kx / quantization
        sy = ky / quantization

        coordinates = np.int32(np.round((coordinates - (minx, miny)) / (sx, sy)))    
        
        # Step 3
        click.secho('Build endpoints index', fg='yellow')
        
        driver = 'ESRI Shapefile'
        schema = {
            'geometry': 'Point',
            'properties': [
                ('GID', 'int')
            ]
        }
        crs = fiona.crs.from_epsg(config.srid)
        options = dict(driver=driver, crs=crs, schema=schema)
        
        with fiona.open(params.nodes.filename(tileset=tileset), 'w', **options) as dst:
            
            coordinates_map = dict()
            gid = 0
            
            point_index = dict()
            
            with click.progressbar(enumerate(coordinates), length=len(coordinates)) as processing:
                for i, coordinate in processing:
                    
                    c = tuple(coordinate)
                    
                    if c not in coordinates_map:
                        
                        coordinates_map[c] = i
                        node_coords = (c[0]*sx + minx, c[1]*sy + miny)
                         
                        point_index[gid] = Point(node_coords[0], node_coords[1])
                        
                        dst.write({
                            'type': 'Feature',
                            'geometry': {'type': 'Point', 'coordinates': node_coords},
                            'properties': {'GID': gid}
                        }) 

                        gid = gid + 1

            del coordinates
            del coordinates_map
            
        # Step 4
        click.secho('Output lines with nodes attributes', fg='yellow')
        
        nodes_list = MultiPoint(list(point_index.values()))

        def nearest(point):
            """ Return the nearest point in the point index
            """
            nearest_node = nearest_points(point, nodes_list)[1]
            
            for candidate in point_index:
                if point_index[candidate].equals(nearest_node):
                    return candidate
                
            return None

        schema = fs.schema
        schema['properties']['NODEA'] = 'int:10'
        schema['properties']['NODEB'] = 'int:10'

        options = dict(driver=driver, crs=crs, schema=schema)

        with fiona.open(params.network_identified.filename(tileset=tileset), 'w', **options) as dst:
            with click.progressbar(fs) as processing:
                for feature in processing:
                    
                    output_feature = feature
                    
                    a = Point(output_feature['geometry']['coordinates'][0])
                    b = Point(output_feature['geometry']['coordinates'][-1])
                    
                    output_feature['properties']['NODEA'] = nearest(a)
                    output_feature['properties']['NODEB'] = nearest(b)
                    
                    dst.write(output_feature)
                    