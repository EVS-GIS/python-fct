# coding: utf-8

"""
DEM Burning
Match mapped stream network and DEM by adjusting stream's elevation

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
    Burns DEM from hydrologic network parameters
    """
    hydro_network = DatasourceParameter('reference hydrologic network')
    hydrography_strahler_fieldbuf = DatasetParameter('reference stream network with strahler order and buffer field to compute buffer before burn DEM', type='input')
    sources = DatasetParameter('Stream sources from the reference hydrologic network', type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.hydro_network = 'hydrography'
        self.hydrography_strahler_fieldbuf = 'hydrography-strahler-fieldbuf'
        self.sources = 'river-network-sources'

# source code https://here.isnew.info/strahler-stream-order-in-python.html
def PrepareStrahlerAndBuffer(params, buffer_factor=5, overwrite=True):
    """
    Prepare hydrologic network before burn : 
        - calculate Strahler stream order
        - calculate buffer used to burn DEM based on Strahler order (order 1 = buffer_factor meters, 2 = 2*buffer_factor meters, 3*buffer_factor = 4x meters ...)

    Parameters:
    - params (object): An object containing the parameters for buffering.
        - hydro_network (str): The filename of the hydro network.
        - hydrography_strahler_fieldbuf (str): The filename for hydro network pepared.
    - buffer_factor(float): defaut=5
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.

    Returns:
    - None

    """
    # file path definition
    hydro_network = params.hydro_network.filename()
    hydrography_strahler_fieldbuf = params.hydrography_strahler_fieldbuf.filename(tileset=None)

    # check overwrite
    if os.path.exists(hydrography_strahler_fieldbuf) and not overwrite:
        click.secho('Output already exists: %s' % hydrography_strahler_fieldbuf, fg='yellow')
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

    # reak reference network
    with fiona.open(hydro_network, 'r') as source:

        schema = source.schema.copy()
        driver=source.driver
        crs=source.crs

        # define new fields
        lines = []
        strahler_field_name = "strahler"
        strahler_field_type = 'int'
        buffer_field_name = 'buffer'
        buffer_field_type = 'float'

        # Add the new field to the schema
        schema['properties'][strahler_field_name] = strahler_field_type
        schema['properties'][buffer_field_name] = buffer_field_type

        source_buff_copy = []
        for feature in source:
                # Create a new feature with the new field
                new_properties = feature['properties']
                new_properties[strahler_field_name] = 0  # Set the strahler field value to 0
                new_properties[buffer_field_name] = 0 # Set the buffer field value to 0
                geom = shape(feature['geometry'])
                # copy line coordinates to find head line
                line = geom.coords
                lines.append(line)
                # copy features in new list to update the data before write all
                source_buff_copy.append(feature)

        # save head lines index
        head_idx = find_head_lines(lines)

        for idx in head_idx:
            curr_idx = idx
            curr_ord = 1
            # head lines order = 1
            source_buff_copy[curr_idx]['properties'][strahler_field_name] = curr_ord
            # go downstream from each head lines
            while True:
                # find next line downstream
                next_idx = find_next_line(curr_idx, lines)
                # stop iteration if no next line
                if not next_idx:
                    break
                # copy next line feature and order
                next_feat = source_buff_copy[next_idx]
                next_ord = next_feat['properties'][strahler_field_name]
                # find sibling line
                sibl_idx = find_sibling_line(curr_idx, lines)
                # if sibling line exist
                if sibl_idx is not None:
                    # copy sibling line feature and order
                    sibl_feat = source_buff_copy[sibl_idx]
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
                source_buff_copy[next_idx]['properties'][strahler_field_name] = curr_ord
                # go further downstream
                curr_idx = next_idx

            # write final features from updated features copy
            with fiona.open(hydrography_strahler_fieldbuf, 'w', driver=driver, crs=crs, schema=schema) as modif:
                for feature in source_buff_copy:
                    # calculate buffer based on strahler order
                    feature['properties'][buffer_field_name] = buffer_factor * (2 ** (feature['properties'][strahler_field_name]-1))
                    modified_feature = {
                            'type': 'Feature',
                            'properties': feature['properties'],
                            'geometry': feature['geometry'],
                        }

                    modif.write(modified_feature)

def CreateSources(params, overwrite=True):
    """
    Create sources from reference hydrologic network : 

    Parameters:
    - params (object): An object containing the parameters for buffering.
        - hydrography_strahler_fieldbuf (str): The filename for hydro network pepared.
        - sources (str) : sources filename path output.
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.

    Returns:
    - None

    """
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