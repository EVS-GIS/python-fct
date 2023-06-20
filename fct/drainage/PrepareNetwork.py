# coding: utf-8

"""
Hydrologic network preparation and stream sources creation

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os

import fiona

import click

from shapely.geometry import (
    shape,
    Point,
    mapping
)

from ..config import (
    config,
    DatasetParameter,
    DatasourceParameter
)

from ..config import config

class Parameters:
    """
    Prepare hydrologic network
    """
    hydro_network = DatasourceParameter('reference hydrologic network')
    hydro_network_strahler = DatasetParameter('reference stream network with Strahler order', type='input')
    hydrography_strahler_fieldbuf = DatasetParameter('reference stream network with Strahler order and buffer field to compute buffer before burn DEM', type='output')
    sources = DatasetParameter('stream sources from the reference hydrologic network', type='output')
    sources_confluences = DatasetParameter('sources and confluences extracted from hydrologic network input', type='output')

    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.hydro_network = 'hydrography'
        self.hydro_network_strahler = 'hydrography-strahler'
        self.hydrography_strahler_fieldbuf = 'hydrography-strahler-fieldbuf'
        self.sources = 'river-network-sources'
        self.sources_confluences = 'sources-confluences'

# source code https://here.isnew.info/strahler-stream-order-in-python.html
def StrahlerOrder(params, tileset=None, overwrite=True):
    """
    Calculate Strahler stream order
    Parameters:
    - params (object): An object containing the parameters.
        - hydro_network (str): The filename of the hydro network.
        - hydrography_strahler (str): The filename for hydro network with Strahler order.
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.

    Returns:
    - None

    """
    click.secho('Compute Strahler order', fg='yellow')
    # file path definition
    hydro_network = params.hydro_network.filename()
    hydrography_strahler = params.hydrography_strahler.filename(tileset=tileset)

    # check overwrite
    if os.path.exists(hydrography_strahler) and not overwrite:
        click.secho('Output already exists: %s' % hydrography_strahler, fg='yellow')
        return

    # function to find head line in network (top upstream)
    def find_head_lines(lines):
        head_idx = []

        num_lines = len(lines)
        for i in range(num_lines):
            line = lines[i]
            first_point = line[0]

            has_upstream = False

            for j in range(num_lines):
                if j == i:
                    continue
                line = lines[j]
                last_point = line[len(line)-1]

                if first_point == last_point:
                    has_upstream = True

            if not has_upstream:
                head_idx.append(i)

        return head_idx

    # function to find next line downstream
    def find_next_line(curr_idx, lines):
        num_lines = len(lines)

        line = lines[curr_idx]
        last_point = line[len(line)-1]

        next_idx = None

        for i in range(num_lines):
            if i == curr_idx:
                continue
            line = lines[i]
            first_point = line[0]

            if last_point == first_point:
                next_idx = i
                break

        return next_idx

    # function to find sibling line (confluence line)
    def find_sibling_line(curr_idx, lines):
        num_lines = len(lines)

        line = lines[curr_idx]
        last_point = line[len(line)-1]

        sibling_idx = None

        for i in range(num_lines):
            if i == curr_idx:
                continue
            line = lines[i]
            last_point2 = line[len(line)-1]

            if last_point == last_point2:
                sibling_idx = i
                break

        return sibling_idx

    # read reference network
    with fiona.open(hydro_network, 'r') as source:

        schema = source.schema.copy()
        driver=source.driver
        crs=source.crs

        # define new fields
        strahler_field_name = "strahler"
        strahler_field_type = 'int'
        # Add the new field to the schema
        schema['properties'][strahler_field_name] = strahler_field_type

        lines = []
        source_copy = []

        # copy feature with strahler field in source_copy and the the line coordinates in lines
        for feature in source:
                # Create a new feature with the new field
                new_properties = feature['properties']
                new_properties[strahler_field_name] = 0  # Set the strahler field value to 0
                geom = shape(feature['geometry'])
                # copy line coordinates to find head line
                line = geom.coords
                lines.append(line)
                # copy features in new list to update the data before write all
                source_copy.append(feature)

        # save head lines index
        head_idx = find_head_lines(lines)

        with click.progressbar(head_idx) as processing:
            for idx in processing:
                curr_idx = idx
                curr_ord = 1
                # head lines order = 1
                source_copy[curr_idx]['properties'][strahler_field_name] = curr_ord
                # go downstream from each head lines
                while True:
                    # find next line downstream
                    next_idx = find_next_line(curr_idx, lines)
                    # stop iteration if no next line
                    if not next_idx:
                        break
                    # copy next line feature and order
                    next_feat = source_copy[next_idx]
                    next_ord = next_feat['properties'][strahler_field_name]
                    # find sibling line
                    sibl_idx = find_sibling_line(curr_idx, lines)
                    # if sibling line exist
                    if sibl_idx is not None:
                        # copy sibling line feature and order
                        sibl_feat = source_copy[sibl_idx]
                        sibl_ord = sibl_feat['properties'][strahler_field_name]
                        # determinate order base on sibling, next and current line
                        if sibl_ord > curr_ord:
                            break
                        elif sibl_ord < curr_ord:
                            if next_ord == curr_ord:
                                break
                        else:
                            curr_ord += 1
                    # update order in feature copy dict
                    source_copy[next_idx]['properties'][strahler_field_name] = curr_ord
                    # go further downstream
                    curr_idx = next_idx

                # write final features from updated features copy
                with fiona.open(hydrography_strahler, 'w', driver=driver, crs=crs, schema=schema) as modif:
                    for feature in source_copy:
                        modified_feature = {
                                'type': 'Feature',
                                'properties': feature['properties'],
                                'geometry': feature['geometry'],
                            }

                        modif.write(modified_feature)


def BufferFieldOnStrahler(params, buffer_factor=5, overwrite=True):
    """
    Prepare hydrologic network before burn : 
        - calculate buffer used to burn DEM based on Strahler order (order 1 = buffer_factor meters, 2 = 2*buffer_factor meters, 3*buffer_factor = 4x meters ...)

    Parameters:
    - params (object): An object containing the parameters for buffering.
        - hydro_network_strahler (str): The filename of the hydro network with strahler order.
        - hydrography_strahler_fieldbuf (str): The filename of the hydro network with buffer field based on strahler order.
    - buffer_factor(float): default=5
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.

    Returns:
    - None

    """
    click.secho('Calculate buffer field on Strahler order', fg='yellow')
    hydro_network_strahler = params.hydro_network_strahler.filename()
    hydrography_strahler_fieldbuf = params.hydrography_strahler_fieldbuf.filename(tileset=None)

    # check overwrite
    if os.path.exists(hydrography_strahler_fieldbuf) and not overwrite:
        click.secho('Output already exists: %s' % hydrography_strahler_fieldbuf, fg='yellow')
        return
    
     # read reference network and create new buffer field
    with fiona.open(hydro_network_strahler, 'r') as strahler:

        schema = strahler.schema.copy()

        # define new fields
        buffer_field_name = 'buffer'
        buffer_field_type = 'float'
        strahler_field_name = 'strahler'

        # Add the new field to the schema
        schema['properties'][buffer_field_name] = buffer_field_type

        options = dict(
            schema = schema,
            driver=strahler.driver,
            crs=strahler.crs
        )

        # edit buffer field and save in new file
        with fiona.open(hydrography_strahler_fieldbuf, 'w', **options) as dst:
            with click.progressbar(dst) as processing:
                for feature in processing:
                        # Create a new feature with the new field
                    feature['properties'][buffer_field_name] = buffer_factor * (2 ** (feature['properties'][strahler_field_name]-1))
                    modified_feature = {
                            'type': 'Feature',
                            'properties': feature['properties'],
                            'geometry': feature['geometry'],
                        }

                    dst.write(modified_feature)

def CreateSources(params, overwrite=True):
    """
    Create stream sources from reference hydrologic network : 

    Parameters:
    - params (object): An object containing the parameters for buffering.
        - hydrography_strahler_fieldbuf (str): The filename for hydro network pepared.
        - sources (str) : stream sources filename path output.
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.

    Returns:
    - None

    """
    click.secho('Create stream sources from reference hydrologic network', fg='yellow')
    # paths to files
    hydrography_strahler_fieldbuf = params.hydrography_strahler_fieldbuf.filename(tileset=None)
    sources = params.sources.filename(tileset=None)

    # check overwrite
    if os.path.exists(sources) and not overwrite:
        click.secho('Output already exists: %s' % sources, fg='yellow')
        return

    with fiona.open(hydrography_strahler_fieldbuf, 'r') as hydro:

        # Create output schema
        schema = hydro.schema.copy()
        schema['geometry'] = 'Point'

        options = dict(
            driver=hydro.driver,
            schema=schema,
            crs=hydro.crs)
        
        with fiona.open(sources, 'w', **options) as output:

            # extract network line with strahler = 1 and create point with first line point coordinates
            for feature in hydro:
                if feature['properties']['strahler'] == 1:
                    properties = feature['properties']
                    geom = shape(feature['geometry'])
                    head_point = Point(geom.coords[0][:2])

                    output.write({
                        'geometry': mapping(head_point),
                        'properties': properties,
    })

def CreateSourcesAndConfluences(params, node_id_field, axis_field, hydro_id_field=None, toponym_field=None, hack_field=None, overwrite=True):
    """
    Create stream sources and confluences from a reference hydrologic network.

    Parameters:
    - params (object): An object containing the parameters for buffering.
        - hydrography_strahler_fieldbuf (str): The filename for the prepared hydro network.
        - sources_confluences (str): The filename path for the output stream sources.
    - node_id_field (str): The field name for the node ID.
    - axis_field (str): The field name for the axis.
    - hydro_id_field (str): Optional. The field name for the hydro ID.
    - toponym_field (str): Optional. The field name for the toponym.
    - hack_field (str): Optional. The field name for the hack. Pass None if not applicable.
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.

    Returns:
    - None
    """
    click.secho('Create stream sources and confluences from reference hydrologic network', fg='yellow')
    # paths to files
    hydrography_strahler_fieldbuf = params.hydrography_strahler_fieldbuf.filename(tileset=None)
    sources_confluences = params.sources_confluences.filename(tileset=None)

    # check overwrite
    if os.path.exists(sources_confluences) and not overwrite:
        click.secho('Output already exists: %s' % sources_confluences, fg='yellow')
        return
    
    def remove_chars_and_zeros(input_string):
        numeric_part = ""
        found_non_zero = False

        for char in input_string:
            if char.isdigit():
                if char != '0' or found_non_zero:
                    numeric_part += char
                    found_non_zero = True

        return int(numeric_part)

    with fiona.open(hydrography_strahler_fieldbuf, 'r') as hydro:

        # Create output schema
        schema = hydro.schema.copy()

        schema['geometry'] = 'Point'

        node_id_name = 'NODE'
        hydro_id_name = 'CDENTITEHY'
        toponym_name = 'TOPONYME'
        axis_name = 'AXIS'
        hack_name = 'HACK'
        nodea = 'NODEA'
        nodeb = 'NODEB'


        # Add the new field to the schema
        if not node_id_name in schema :
            schema['properties'][node_id_name] = 'int'
        if not axis_name in schema :
            schema['properties'][axis_name] = 'int' 
        if not hydro_id_name in schema :
                schema['properties'][hydro_id_name] = 'str' 
        if not toponym_name in schema :
            schema['properties'][toponym_name] = 'str' 
        if not hack_name in schema :
            schema['properties'][hack_name] = 'int'
    
        if schema['properties'][nodea]:
            click.secho('NODEA field identified, auto-remove', fg='blue')
            del schema['properties'][nodea]
        if schema['properties'][nodeb]:
            click.secho('NODEB field identified, auto-remove', fg='blue')
            del schema['properties'][nodeb]

        options = dict(
            driver=hydro.driver,
            schema=schema,
            crs=hydro.crs)
        
        with fiona.open(sources_confluences, 'w', **options) as output:

            # extract first point for each line in hydrologic network
            for feature in hydro:
                properties = feature['properties']
                geom = shape(feature['geometry'])
                head_point = Point(geom.coords[0][:2])

                # update properties with FCT names
                if type(properties[node_id_field]) == str:
                    properties[node_id_name] = remove_chars_and_zeros(properties[node_id_field])
                else:
                    properties[node_id_name] = properties[node_id_field]

                if type(properties[axis_field]) == str:
                    properties[axis_name] = remove_chars_and_zeros(properties[axis_field])
                else:
                    properties[axis_name] = properties[axis_field]

                if hydro_id_field:
                    properties[hydro_id_name] = str(properties[hydro_id_field])
                else:
                    properties[hydro_id_name] = None

                if toponym_field:
                    properties[toponym_name] = str(properties[toponym_field])
                else:
                    properties[toponym_name] = None

                if hack_field:
                    properties[hack_name] = int(properties[hydro_id_field])
                else:
                    properties[hack_name] = None

                if nodea in properties:
                    del properties[nodea]
                if nodeb in properties:
                    del properties[nodeb]

                output.write({
                    'geometry': mapping(head_point),
                    'properties': properties,
    })

