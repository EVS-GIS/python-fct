# coding: utf-8

"""
Configuration Classes

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
from collections import namedtuple
from configparser import ConfigParser
import yaml
from shapely.geometry import asShape
import fiona

Tile = namedtuple('Tile', ('gid', 'row', 'col', 'x0', 'y0', 'bounds', 'tileset'))
DataSource = namedtuple('DataSource', ('name', 'filename', 'resolution'))

def from_srs(srs):
    """
    Return SRID from EPSG Identifier
    """

    if srs.startswith('EPSG:'):
        return int(srs[5:])

    return 0

class Configuration():
    """
    Configuration singleton,
    which can be read from a .ini file.

    Configuration defines:

    - datasources
    - tilesets
    - datasets
    - shared parameters: workdir, srid
    """

    def __init__(self):

        self._tilesets = dict()
        self._datasources = dict()
        self._workspace = None

    def default(self):
        """
        Populate configuration from default `config.ini`
        """
        filename = os.path.join(os.path.dirname(__file__), 'config.ini')
        self.configure(*FileParser.parse(filename))

    def from_file(self, filename):
        """
        Populate configuration from .ini file
        """
        self.configure(*FileParser.parse(filename))

    def dataset(self, name):
        """
        Return Dataset definition
        """
        return self._workspace.dataset(name)

    def fileset(self, names, **kwargs):
        """
        Return list of Dataset filename instances
        """
        return [
            self.filename(name, **kwargs)
            for name in names
        ]

    def tileset(self, name='default'):
        """
        Return Tileset definition
        """
        return self._tilesets[name]

    def datasource(self, name):
        """
        Return DataSource definition
        """
        return self._datasources[name]

    def filename(self, name, **kwargs):
        """
        Return Dataset filename instance
        """
        dst = self.dataset(name)

        folder = os.path.join(
            self.workdir,
            dst.subdir(**kwargs))

        if not os.path.exists(folder):
            os.makedirs(folder)

        return os.path.join(
            folder,
            dst.filename(**kwargs))

    @property
    def workdir(self):
        """
        Return working directory
        """
        return self._workspace.workdir

    @property
    def srid(self):
        """
        Return SRID
        """
        return self._workspace.srid

    def configure(self, workspace, datasources, tilesets):
        """
        Populate configuration
        """

        for tileset in tilesets.values():
            tileset.workspace = workspace

        self._workspace = workspace
        self._datasources = datasources
        self._tilesets = tilesets

class Dataset():
    """
    Describes an output dataset
    """

    def __init__(self, name, filename='', tilename='', subdir='', ext='.tif'):

        self._name = name
        self._filename = filename
        self._tilename = tilename
        self._subdir = subdir
        self._ext = ext

        if filename == '':
            self._filename = name.upper() + ext

        if tilename == '':
            self._tilename = name.upper() + '_%(row)02d_%(col)02d'

    @property
    def name(self):
        """
        Return dataset's name
        """
        return self._name

    def subdir(self, **kwargs):
        """
        Return storage subdirectory
        """
        return self._subdir % kwargs

    @property
    def ext(self):
        """
        Return file extension
        """
        return self._ext

    def filename(self, **kwargs):
        """
        Return filename instance
        """

        if self._filename is None:
            raise ValueError('Dataset %s has only tiles' % self.name)

        if kwargs:
            return self._filename % kwargs

        return self._filename

    def tilename(self, **kwargs):
        """
        Return tilename instance
        """

        if self._tilename is None:
            raise ValueError('Dataset %s does not have tiles' % self.name)

        if kwargs:
            return (self._tilename + self.ext) % kwargs

        return self._tilename


class Workspace():
    """
    Shared parameters and output dataset definitions
    """

    def __init__(self, workspace, datasets):

        self._workdir = ''
        self._datasets = datasets
        self._srs = ''
        self._srid = 0

        if 'workdir' in workspace:
            self._workdir = workspace['workdir']

        if 'srs' in workspace:
            self._srs = workspace['srs']
            self._srid = from_srs(self._srs)

    def dataset(self, name):
        """
        Return named Dataset
        """
        
        if not name in self._datasets:
            self._datasets[name] = Dataset(name)

        return self._datasets[name]

    @property
    def workdir(self):
        """
        Return working directory
        """
        return self._workdir

    @property
    def srid(self):
        """
        Return SRID
        """
        return self._srid

    @property
    def srs(self):
        """
        Return SRS Identifier
        """
        return self._srs

class Tileset():
    """
    Describes a tileset from a shapefile index
    """

    def __init__(self, name, index, height, width, tiledir, resolution):
        
        self._name = name
        self._index = index
        self._height = height
        self._width = width
        self._bounds = None
        self._length = 0
        self._tileindex = None
        self._tiledir = tiledir
        self._resolution = resolution
        self._is_configured = False
        self.workspace = None

    def configure(self):

        if self._is_configured:
            return

        with fiona.open(self._index) as fs:
            self._bounds = fs.bounds
            self._length = len(fs)

        self._is_configured = True

    @property
    def name(self):
        """
        Name of this tileset
        """
        return self._name

    @property
    def height(self):
        """
        Height in pixels of one tile
        """
        return self._height

    @property
    def width(self):
        """
        Width in pixels of one tile
        """
        return self._width

    @property
    def resolution(self):
        return self._resolution
    

    @property
    def bounds(self):
        """
        (minx, miny, maxx, maxy) bounds of this tileset
        """
        self.configure()
        return self._bounds

    @property
    def tileindex(self):
        """
        Index of tiles belonging to this tileset
        """

        if self._tileindex is None:

            index = dict()

            with fiona.open(self._index) as fs:

                for feature in fs:
                    props = feature['properties']
                    values = [props[k] for k in ('GID', 'ROW', 'COL', 'X0', 'Y0')]
                    values.append(asShape(feature['geometry']).bounds)
                    values.append(self.name)
                    tile = Tile(*values)
                    index[(tile.row, tile.col)] = tile

            self._tileindex = index

        return self._tileindex

    def __len__(self):
        """
        Return number of tiles in tileindex
        """

        self.configure()
        return self._length

    def tiles(self):
        """
        Generator of tiles
        """

        with fiona.open(self._index) as fs:
            for feature in fs:
                
                props = feature['properties']
                values = [props[k] for k in ('GID', 'ROW', 'COL', 'X0', 'Y0')]
                values.append(asShape(feature['geometry']).bounds)
                values.append(self.name)
                yield Tile(*values)


    def index(self, x, y):
        """
        Return tile coordinates of real world point (x, y)
        """

        self.configure()
        minx, _, _, maxy = self._bounds
        row = int((maxy - y) // self._resolution)
        col = int((x - minx) // self._resolution)
        return row, col

    def tilename(self, dataset, row, col, **kwargs):
        """
        Return full-path filename instance for specific tile
        """

        dst = self.workspace.dataset(dataset)

        folder = os.path.join(
            self.workspace.workdir,
            dst.subdir(**kwargs),
            self._tiledir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        return os.path.join(
            folder,
            dst.tilename(row=row, col=col, **kwargs))

class FileParser():
    """
    Read configuration components form a .ini file
    """

    @staticmethod
    def parse(configfile):
        """
        Main parsing method
        """

        parser = ConfigParser()
        parser.read(configfile)

        sections = list()
        datasources = dict()
        tilesets = dict()
        workspace = dict()

        for section in parser.sections():

            if section in ('Workspace', 'DataSources', 'Tilesets'):
                sections.append(section)
                continue

            items = dict(parser.items(section))

            if 'type' in items:
                item_type = items['type']
                if item_type == 'datasource':
                    datasources[section] = FileParser.datasource(section, items)
                elif item_type == 'tileset':
                    tilesets[section] = FileParser.tileset(section, items)

        while sections:

            section = sections.pop()

            if section == 'Workspace':
                for key, value in parser.items(section):
                    workspace[key] = value
            elif section == 'DataSources':
                for key, value in parser.items(section):
                    if value in datasources:
                        datasources[key] = datasources[value]
                    else:
                        raise KeyError(value)
            elif section == 'Tilesets':
                for key, value in parser.items(section):
                    if value in tilesets:
                        tilesets[key] = tilesets[value]
                    else:
                        raise KeyError(value)

        datasets = FileParser.datasets(
            os.path.join(
                os.path.dirname(configfile),
                'datasets.yml'))

        return Workspace(workspace, datasets), datasources, tilesets

    @staticmethod
    def datasource(name, items):
        """
        Populate a DataSource object
        """

        filename = items.get('filename', None)
        resolution = float(items.get('resolution', 0.0))
        return DataSource(name, filename, resolution)

    @staticmethod
    def tileset(name, items):
        """
        Populate a Tileset object
        """

        index = items['index']
        tiledir = items['tiledir']
        height = int(items['height'])
        width = int(items['width'])
        resolution = float(items['resolution'])
        return Tileset(name, index, height, width, tiledir, resolution)

    @staticmethod
    def datasets(dstfile):
        """
        Read YAML Dataset definitions
        and return a dict of datasets
        """

        datasets = dict()

        with open(dstfile) as fp:
            data = yaml.safe_load(fp)

        for name in data:

            filename = data[name].get('filename', None)
            subdir = data[name]['subdir']

            if 'tiles' in data[name]:

                tilename = data[name]['tiles']['template']
                ext = data[name]['tiles']['extension']

            else:

                tilename = None
                ext = None

            dst = Dataset(name, filename, tilename, subdir, ext)
            datasets[name] = dst

        return datasets
