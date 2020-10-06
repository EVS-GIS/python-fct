# coding: utf-8

"""
NetCDF Metadata Helper

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import time
from datetime import datetime
import yaml

from .. import __version__

def set_metadata(dataset, metafile):
    """
    Set metadata on xarray/netcdf dataset
    from YAML descriptor
    """

    filename = os.path.join(
        os.path.dirname(__file__),
        'global.yml'
    )

    if os.path.exists(filename):

        with open(filename) as fp:
            metadata = yaml.safe_load(fp)

        for attr, value in metadata.items():
            dataset.attrs[attr] = value

    filename = os.path.join(
        os.path.dirname(__file__),
        metafile + '.yml'
    )

    # print(filename, os.path.exists(filename))

    if os.path.exists(filename):

        with open(filename) as fp:
            metadata = yaml.safe_load(fp)

        for attr, value in metadata['global'].items():
            dataset.attrs[attr] = value

        for variable in metadata['coordinates']:
            meta = metadata['coordinates'][variable]
            for attr, value in meta.items():
                dataset[variable].attrs[attr] = value

        for variable in metadata['variables']:
            meta = metadata['variables'][variable]
            for attr, value in meta.items():
                dataset[variable].attrs[attr] = value

    else:

        raise ValueError('File does not exist: %s' % (metafile + '.yml'))

    generated_time = time.time()
    dataset.attrs['version'] = __version__
    dataset.attrs['generated'] = str(datetime.fromtimestamp(generated_time))
