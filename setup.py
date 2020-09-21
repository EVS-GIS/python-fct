# coding: utf-8

"""
Fluvial Corridor Toolbox Setup

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from setuptools import setup, find_packages
from distutils.extension import Extension
import numpy
from Cython.Build import cythonize

# Parse the version from the main module.
with open('fct/__init__.py', 'r') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

open_kwds = {'encoding': 'utf-8'}

with open('VERSION.txt', 'w', **open_kwds) as f:
    f.write(version)

extensions = [

    Extension(
        'fct.transform',
        ['cython/transform.pyx'],
        language='c',
        include_dirs=[numpy.get_include()]
    ),

    Extension(
        'fct.speedup',
        ['cython/speedup.pyx'],
        language='c++',
        include_dirs=[numpy.get_include()]
    ),

    Extension(
        'fct.terrain_analysis',
        ['terrain/cython/terrain_analysis.pyx'],
        language='c++',
        include_dirs=[numpy.get_include()]
    )

]

setup(
    name='fct',
    version=version,
    packages=['fct'],
    ext_modules=cythonize(extensions),
    include_package_data=True,
    install_requires=[
        'numpy>=1.18',
        'xarray>=0.15',
        'scipy>=1.4',
        'rasterio>=1.1',
        'fiona>=1.8.6',
        'shapely>=1.7',
        'pyyaml>=5.3',
        'Click>=7.0',
        'pydotenv'
    ],
    entry_points='''
        [console_scripts]
            fct=fct.cli.InfoCommand:cli
            fct-files=fct.cli.FileCommand:cli
            fct-tiles=fct.cli.TileCommand:cli
            fct-drainage=fct.drainage.Command:cli
            fct-corridor=fct.corridor.Command:cli
            fct-metrics=fct.metrics.Command:cli
            fct-plot=fct.plotting.Command:cli
    ''',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: GIS']
)
