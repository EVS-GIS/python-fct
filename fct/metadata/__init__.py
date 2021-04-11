"""
NetCDF Metadata Helper
"""

import os
from itertools import chain
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

        for section in ('dims', 'coordinates', 'variables'):

            if section not in metadata:
                continue

            for variable in metadata[section]:

                meta = metadata[section][variable]

                for attr, value in meta.items():
                    if variable in dataset:

                        dataset[variable].attrs[attr] = value

    else:

        raise ValueError('File does not exist: %s' % (metafile + '.yml'))

    generated_time = time.time()
    dataset.attrs['version'] = __version__
    dataset.attrs['generated'] = str(datetime.fromtimestamp(generated_time))
