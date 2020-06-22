import os
from collections import Counter, defaultdict
import click
import fiona
import fiona.crs

def JoinNetworkAttributes():

    sources_shapefile = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS_SOURCES.shp'
    network_shapefile = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS.shp'
    output = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS_ATTR.shp'

    graph = dict()
    rgraph = defaultdict(list)
    indegree = Counter()

    with fiona.open(network_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                fid = feature['id']
                properties = feature['properties']
                a = properties['NODEA']
                b = properties['NODEB']

                graph[a] = (b, fid)
                indegree[b] += 1

    sources = dict()

    with fiona.open(sources_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                node = properties['NODE']
                sources[node] = properties

    def greater(c1, h1, c2, h2):

        if c2 is None or h2 is None:
            return True

        if c1.count('-') > c2.count('-'):
            return True

        if c1.count('-') == c2.count('-'):

            rank1 = int(c1[4:7])
            rank2 = int(c2[4:7])

            if rank1 < rank2:
                return True

            if rank1 == rank2:
                return h1 < h2

    def resolve_properties(node):

        if node in sources:
            return sources[node]

        if len(rgraph[node]) == 1:
            return rgraph[node][0][1]

        _cdentite = None
        _hack = None
        _properties = None

        for _, properties in rgraph[node]:

            if properties is None:
                continue

            cdentite = properties['CDENTITEHY']
            hack = properties['HACK']

            if greater(cdentite, hack, _cdentite, _hack):

                _cdentite, _hack = cdentite, hack
                _properties = properties

        return _properties

    queue = [node for node in graph if indegree[node] == 0]
    features = dict()

    while queue:

        node = queue.pop()
        properties = resolve_properties(node)

        if node in graph:

            next_node, fid = graph[node]
            features[fid] = properties
            rgraph[next_node].append((node, properties))

            indegree[next_node] -= 1

            if indegree[next_node] == 0:
                queue.append(next_node)


    with fiona.open(network_shapefile) as fs:

        driver = fs.driver
        schema = fs.schema
        crs = fs.crs

        schema['properties'].update({
            'CDENTITEHY': 'str:8',
            'TOPONYME': 'str:254',
            'AXIS': 'int'
        })

        options = dict(driver=driver, crs=crs, schema=schema)

        with fiona.open(output, 'w', **options) as dst:
            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    fid = feature['id']

                    if fid not in features:
                        continue

                    properties = features[fid]

                    if properties is None:
                        feature['properties'].update({
                            'CDENTITEHY': None,
                            'TOPONYME': None,
                            'AXIS': None
                        })
                    else:
                        feature['properties'].update({
                            k: properties[k] for k in ('CDENTITEHY', 'TOPONYME', 'AXIS')
                        })

                    dst.write(feature)
