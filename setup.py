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

def get_version():
    """Get version and version_info without importing the entire module."""

    # Parse the version from the main module.
    with open('fct/__init__.py', 'r') as f:
        for line in f:
            if line.find("__version__") >= 0:
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

    open_kwds = {'encoding': 'utf-8'}

    with open('VERSION.txt', 'w', **open_kwds) as f:
        f.write(version)

    return version

def get_requirements(req):
    """Load list of dependencies."""

    with open(req) as f:

        install_requires = [
            line.strip()
            for line in f
            if not line.startswith("#")
        ]

    return install_requires


def get_description():
    """Get long description."""

    with open("README.md", encoding="utf-8") as f:
        desc = f.read()
    return desc

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
    version=get_version(),
    long_description=get_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tools', 'scripts', 'test*']),
    ext_modules=cythonize(extensions),
    include_package_data=True,
    install_requires=get_requirements('requirements/fct.txt'),
    entry_points='''
        [console_scripts]
            fct=fct.cli.InfoCommand:cli
            fct-files=fct.cli.FileCommand:cli
            fct-tiles=fct.cli.TileCommand:cli
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
