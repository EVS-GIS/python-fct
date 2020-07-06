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
import fiona

Tile = namedtuple('Tile', ('gid', 'row', 'col', 'x0', 'y0'))
DataSource = namedtuple('DataSource', ('name', 'filename', 'resolution'))

class Configuration():
    """
    DOCME
    """

    def __init__(self):
        
        self._tilesets = dict()
        self._datasources = dict()
        self._workspace = None

    def default(self):
        
        filename = os.path.join(os.path.dirname(__file__), 'config.ini')
        self.set(*FileParser.parse(filename))

    def from_file(self, filename):

        self.set(*FileParser.parse(filename))

    def tileset(self, name):
        
        if name in self._tilesets:
            return self._tilesets[name]

        raise KeyError('No such tileset %s' % name)

    def datasource(self, name):
        
        if name in self._datasources:
            return self._datasources[name]

        raise KeyError('No such datasource %s' % name)

    def filename(self, name, **kwargs):
        pass

    @property
    def workdir(self):
        return self._workspace.workdir
    
    @property    
    def srid(self):
        return self._workspace.srid

    def set(self, workspace, datasources, tilesets):

        for tileset in tilesets.values():
            tileset.workspace = workspace

        self._workspace = workspace
        self._datasources = datasources
        self._tilesets = tilesets

class Workspace():

    def __init__(self, workspace):

        self._workdir = ''
        self._srs = ''
        self._srid = 0

        if 'workdir' in workspace:
            self._workdir = workspace['workdir']

        if 'srs' in workspace:
            self._srs = workspace['srs']
            self._srid = self.from_srs(self._srs)

    @property
    def workdir(self):
        return self._workdir

    @property
    def srid(self):
        return self._srid

    @property
    def srs(self):
        return self._srs
    
    def from_srs(self, srs):
        
        if srs.startswith('EPSG:'):
            return int(srs[5:])
        
        return 0

class Tileset():
    """
    DOCME
    """

    def __init__(self, name, index, height, width, subdir='', template=None):
        self._name = name
        self._index = index
        self._height = height
        self._width = width
        self._bounds = None
        self._tileindex = None
        self.workspace = None
        self.subdir = subdir
        self.template = template or '%(dataset)s_%(row)02d_%(col)02d'

    @property
    def name(self):
        return self._name

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def bounds(self):
        if self._tileindex is None:
            assert self.tileindex is not None
        return self._bounds
    
    @property
    def tileindex(self):

        if self._tileindex is None:

            index = dict()
            
            with fiona.open(self._index) as fs:

                self._bounds = fs.bounds

                for feature in fs:
                    props = feature['properties']
                    values = [props[k] for k in ('GID', 'ROW', 'COL', 'X0', 'Y0')]
                    tile = Tile(*values)
                    index[(tile.row, tile.col)] = tile

            self._tileindex = index

        return self._tileindex

    def tilename(self, dataset, row, col, **kwargs):

        args = kwargs.copy()
        args.update(dataset=dataset, row=row, col=col)
        
        return os.path.join(
            self.workspace.workdir,
            self.subdir,
            self.template % args)

class FileParser():
    """
    DOCME
    """

    @classmethod
    def parse(cls, configfile):

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

            items = {key: value for key, value in parser.items(section)}

            if 'type' in items:
                item_type = items['type']
                if item_type == 'datasource':
                    datasources[section] = cls.datasource(section, items)
                elif item_type == 'tileset':
                    tilesets[section] = cls.tileset(section, items)

        while sections:

            section = sections.pop()

            if section == 'Workspace':
                 for key, value in parser.items(section):
                    workspace[key] = value
            elif section == 'DataSources':
                for key, value in parser.items(section):
                    if value in datasources:
                        datasources[key] = datasources[value]
            elif section == 'Tilesets':
                for key, value in parser.items(section):
                    if value in tilesets:
                        tilesets[key] = tilesets[value]

        return Workspace(workspace), datasources, tilesets

    @classmethod
    def datasource(cls, name, items):

        filename = items.get('filename', None)
        resolution = float(items.get('resolution', 0.0))
        return DataSource(name, filename, resolution)

    @classmethod
    def tileset(cls, name, items):

        index = items.get('index', None)
        height = int(items.get('height', 0))
        width = int(items.get('width', 0))
        subdir = items.get('subdir', '')
        template = items.get('tiletemplate', None)
        return Tileset(name, index, height, width, subdir, template)
